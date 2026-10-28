"""Deterministic EvaHan data audit and preparation (Python standard library only).

The original archive, gold files, raw files, and scorer are always read-only.
Run ``python -m gef.data prepare`` from the code directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Iterable
import zipfile


CANONICAL_POS21 = (
    "a", "c", "d", "f", "j", "m", "n", "nr", "ns", "p", "q", "r",
    "s", "t", "u", "v", "vs", "vw", "vy", "w", "y",
)
PAPER_POS15 = (
    "w", "v", "n", "nr", "r", "d", "c", "y", "p", "ns", "u", "a",
    "t", "m", "X",
)
POS_ALIASES = {"sv": "vs", "wv": "vw", "yv": "vy"}
EXPECTED_QUARANTINE_LINES = (
    159, 3110, 3623, 3985, 4166, 4229, 4505, 5053, 5228, 5323,
    6685, 7164, 8293, 8861,
)
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RELATIVE = Path("LT4HALA-master/2022/data_and_doc")
SEED = 2022
SCHEMA_VERSION = 1


def _track(track: str) -> str:
    aliases = {"paper": "paper15", "official21": "native21", "full21": "native21"}
    track = aliases.get(track, track)
    if track not in {"paper15", "native21"}:
        raise ValueError(f"Unknown POS track: {track!r}")
    return track


def pos_tags(track: str = "paper15") -> tuple[str, ...]:
    """Return real POS classes only; model code adds NULL as class zero."""
    return PAPER_POS15 if _track(track) == "paper15" else CANONICAL_POS21


def map_pos(pos: str, track: str = "paper15") -> str:
    """Map derived labels; native21 deliberately preserves anomalous test vn.

    The author-approved fixed paper15 protocol maps every non-retained label
    (including test vn) to X. The original source gold is never rewritten.
    """
    pos = POS_ALIASES.get(pos, pos)
    if _track(track) == "native21":
        return pos
    return pos if pos in PAPER_POS15[:-1] else "X"


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _file_meta(path: Path, data: bytes | None = None) -> dict:
    data = path.read_bytes() if data is None else data
    result = {"path": str(path.resolve()), "bytes": len(data), "sha256": _sha(data)}
    if path.suffix.lower() in {".txt", ".py"}:
        text = data.decode("utf-8-sig")
        result.update({
            "encoding": "utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8",
            "bom": data.startswith(b"\xef\xbb\xbf"),
            "crlf_count": data.count(b"\r\n"), "lf_count": data.count(b"\n"),
            "physical_lines": len(text.splitlines()),
            "nonempty_lines": sum(bool(line.strip()) for line in text.splitlines()),
            "terminal_newline": text.endswith(("\n", "\r")),
        })
    return result


def _parse(text: str, split: str, source: str) -> tuple[list[dict], list[dict]]:
    samples, anomalies = [], []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        words, errors, offset = [], [], 0
        for index, chip in enumerate(line.split()):
            if "/" not in chip:
                word, original_pos = chip, None
                errors.append({"token_index": index, "token": chip, "reason": "missing_slash"})
            else:
                word, original_pos = chip.rsplit("/", 1)
                if not word or not original_pos:
                    errors.append({"token_index": index, "token": chip, "reason": "empty_word_or_pos"})
                elif POS_ALIASES.get(original_pos, original_pos) not in CANONICAL_POS21:
                    errors.append({"token_index": index, "token": chip,
                                   "original_pos": original_pos, "reason": "noncanonical_pos"})
            pos = POS_ALIASES.get(original_pos, original_pos)
            words.append({"text": word, "start": offset, "end": offset + len(word),
                          "pos": pos, "original_pos": original_pos})
            offset += len(word)
        sample = {"id": f"{split}:{number}", "source": source, "line": number,
                  "text": "".join(word["text"] for word in words), "words": words,
                  "split": split, "augmented": False, "augmentation": None}
        samples.append(sample)
        if errors:
            anomalies.append({"id": sample["id"], "source": source, "line": number,
                              "text": sample["text"], "raw_annotated_line": line,
                              "errors": errors})
    return samples, anomalies


def group_split(text: str) -> str:
    """Stable exact-raw-text grouping, independent of all model random seeds."""
    digest = hashlib.sha256(("2022\0" + text).encode("utf-8")).hexdigest()
    return "dev" if int(digest, 16) % 10 == 0 else "train"


def _quantile(values: list[int], q: float) -> float | int:
    if not values:
        return 0
    values = sorted(values)
    at = (len(values) - 1) * q
    low = int(at)
    high = min(low + 1, len(values) - 1)
    return round(values[low] + (values[high] - values[low]) * (at - low), 6)


def _stats(samples: list[dict]) -> dict:
    tags = Counter(word["pos"] for row in samples for word in row["words"] if word["pos"])
    originals = Counter(word["original_pos"] for row in samples for word in row["words"]
                        if word["original_pos"])
    chars = Counter()
    for row in samples:
        for word in row["words"]:
            if word["pos"]:
                chars[word["pos"]] += len(word["text"])
    lengths = [len(row["text"]) for row in samples]
    valid = sum(tags.values())
    char_denominator = sum(chars.values())
    return {
        "nonempty_lines": len(samples), "token_items": sum(len(r["words"]) for r in samples),
        "valid_word_pos_items": valid, "surface_characters": sum(lengths),
        "valid_word_characters": char_denominator,
        "original_pos_counts": dict(sorted(originals.items())),
        "normalized_pos_counts": dict(sorted(tags.items())),
        "word_percent": {p: n / valid * 100 for p, n in sorted(tags.items())} if valid else {},
        "character_percent": {p: n / char_denominator * 100 for p, n in sorted(chars.items())}
            if char_denominator else {},
        "line_length_characters": {
            "min": min(lengths, default=0), "p50": _quantile(lengths, .5),
            "p90": _quantile(lengths, .9), "p95": _quantile(lengths, .95),
            "p99": _quantile(lengths, .99), "max": max(lengths, default=0),
            "over_128": sum(n > 128 for n in lengths),
            "over_256": sum(n > 256 for n in lengths),
            "over_510": sum(n > 510 for n in lengths),
        },
        "longest_lines": [{"id": r["id"], "line": r["line"], "characters": len(r["text"])}
                          for r in sorted(samples, key=lambda r: (-len(r["text"]), r["line"]))[:10]],
    }


def _duplicates(splits: dict[str, list[dict]]) -> dict:
    indices = {}
    within = {}
    for name, samples in splits.items():
        by_text = defaultdict(list)
        for sample in samples:
            by_text[sample["text"]].append(sample)
        indices[name] = by_text
        duplicates = []
        for text, rows in sorted(by_text.items()):
            if len(rows) < 2:
                continue
            variants = defaultdict(list)
            for row in rows:
                key = json.dumps([(w["text"], w["pos"]) for w in row["words"]], ensure_ascii=False)
                variants[key].append(row["id"])
            duplicates.append({"text": text, "characters": len(text),
                               "ids": [r["id"] for r in rows],
                               "annotation_variants": [{"words": json.loads(k), "ids": v}
                                                       for k, v in sorted(variants.items())]})
        within[name] = {"duplicate_unique_texts": len(duplicates),
                        "excess_rows": sum(len(r["ids"]) - 1 for r in duplicates),
                        "conflicting_annotation_texts": sum(len(r["annotation_variants"]) > 1
                                                            for r in duplicates),
                        "items": duplicates}
    across = {}
    for a, b in itertools.combinations(splits, 2):
        common = sorted(set(indices[a]) & set(indices[b]))
        across[f"{a}__{b}"] = {
            "common_unique_texts": len(common),
            "left_occurrences": sum(len(indices[a][t]) for t in common),
            "right_occurrences": sum(len(indices[b][t]) for t in common),
            "max_common_characters": max(map(len, common), default=0),
            "items": [{"text": t, "left_ids": [r["id"] for r in indices[a][t]],
                       "right_ids": [r["id"] for r in indices[b][t]]} for t in common],
        }
    return {"definition": "Exact unnormalized raw text; no fuzzy matching or automatic deduplication.",
            "within": within, "across": across}


def _threshold(stats: dict) -> dict:
    result = {}
    for name, value in stats.items():
        counts = value["normalized_pos_counts"]
        denominator = value["valid_word_pos_items"]
        kept = sorted(p for p, n in counts.items() if n / denominator >= .002)
        merged = sorted(p for p in counts if p not in kept)
        result[name] = {"denominator": denominator, "threshold": .002,
                        "rule": "retain frequency >= 0.2%, without rounding; diagnostic only",
                        "retained_original_labels": kept, "merged_labels": merged,
                        "derived_labels": sorted(kept + (["X"] if merged else [])),
                        "derived_label_count": len(kept) + bool(merged),
                        "f_percent": counts.get("f", 0) / denominator * 100 if denominator else 0,
                        "j_percent": counts.get("j", 0) / denominator * 100 if denominator else 0}
    return result


def _figure4(stats: dict) -> dict:
    targets = {
        "outer_label_train": {"w": 25.5, "v": 23.2, "n": 15.5, "nr": 6.4, "r": 6.1, "d": 5.1, "ns": 3.1},
        "middle_label_test_b": {"w": 26.1, "v": 23.7, "n": 12.9, "nr": 7.3, "r": 6.1, "d": 5.5, "ns": 4.1},
        "inner_label_test_a": {"w": 21.8, "v": 23.6, "n": 13.8, "nr": 7.9, "r": 5.5, "d": 5.2, "ns": 6.2},
    }
    values = {}
    for name in ("original_train", "test_a", "test_b", "retained_train"):
        s = stats[name]
        percentages = {p: 0.0 for p in PAPER_POS15}
        for p, n in s["normalized_pos_counts"].items():
            percentages[map_pos(p, "paper15")] += n / s["valid_word_pos_items"] * 100
        values[name] = {"paper15_word_percent": percentages,
                        "normalized_word_percent": s["word_percent"],
                        "normalized_character_percent": s["character_percent"]}
    return {"fixed_paper15": list(PAPER_POS15), "figure4_displayed_percentages": targets,
            "matching_local_data": {"outer_label_train": "original_train",
                                    "middle_label_test_b": "test_a", "inner_label_test_a": "test_b"},
            "finding": "Displayed rounded values match word frequencies. Figure4 A/B ring labels are swapped relative to local official filenames; fixed 15 classes conflict with a literal 0.2% threshold.",
            "computed": values}


def _augment(train: list[dict]) -> tuple[list[dict], list[dict], dict]:
    pools = defaultdict(set)
    for sample in train:
        for word in sample["words"]:
            if word["pos"] != "w":
                pools[word["pos"]].add(word["text"])
    pools = {pos: sorted(words) for pos, words in sorted(pools.items())}
    rng = random.Random(SEED)
    copies, log = [], []
    for original in train:
        sample = copy.deepcopy(original)
        eligible = [i for i, w in enumerate(sample["words"])
                    if w["pos"] != "w" and any(t != w["text"] for t in pools[w["pos"]])]
        record = {"id": original["id"] + ":aug", "original_id": original["id"],
                  "method": "same_pos_replacement", "seed": SEED, "pool": "train_only",
                  "replaced": bool(eligible)}
        if eligible:
            index = rng.choice(eligible)
            word = sample["words"][index]
            candidates = [t for t in pools[word["pos"]] if t != word["text"]]
            replacement = rng.choice(candidates)
            record.update({"word_index": index, "pos": word["pos"], "old_text": word["text"],
                           "new_text": replacement, "candidate_count": len(candidates)})
            word["text"] = replacement
        else:
            record["reason"] = "no_nonpunctuation_word_with_an_alternative_train_candidate"
        offset = 0
        for word in sample["words"]:
            word["start"], word["end"] = offset, offset + len(word["text"])
            offset = word["end"]
        sample.update({"id": record["id"], "text": "".join(w["text"] for w in sample["words"]),
                       "augmented": True, "augmentation": record})
        copies.append(sample)
        log.append(record)
    pool_report = {"seed": SEED, "source": "train only after quarantine and group split",
                   "candidate_sort": "Unicode code-point order; unique words",
                   "selection": "one eligible token uniformly, then one alternative word uniformly",
                   "original_train_rows": len(train), "copy_rows": len(copies),
                   "replaced_rows": sum(r["replaced"] for r in log),
                   "unchanged_copy_rows": sum(not r["replaced"] for r in log),
                   "pools": pools}
    return list(train) + copies, log, pool_report


def _collect(root: Path, source_dir: Path | None = None) -> tuple[dict, dict]:
    source = source_dir or root / SOURCE_RELATIVE
    source = source.resolve()
    archive = source / "zuozhuan_train_utf8.zip"
    with zipfile.ZipFile(archive) as zf:
        corrupt = zf.testzip()
        if corrupt:
            raise ValueError(f"Corrupt training ZIP entry: {corrupt}")
        training_bytes = zf.read("zuozhuan_train_utf8.txt")
    original_train, quarantine = _parse(training_bytes.decode("utf-8-sig"), "train",
                                        "zuozhuan_train_utf8.txt")
    bad_ids = {row["id"] for row in quarantine}
    retained = [sample for sample in original_train if sample["id"] not in bad_ids]
    train, dev = [], []
    for sample in retained:
        sample["split"] = group_split(sample["text"])
        (dev if sample["split"] == "dev" else train).append(sample)
    samples = {"original_train": original_train, "retained_train": retained,
               "train": train, "dev": dev}
    raw_lines, validation, source_manifest = {}, {}, {}
    source_manifest[archive.name] = _file_meta(archive)
    member_meta = _file_meta(source / "zuozhuan_train_utf8.txt", training_bytes)
    member_meta.update({"path": str(archive) + "!zuozhuan_train_utf8.txt",
                        "archive": str(archive), "member": "zuozhuan_train_utf8.txt"})
    source_manifest["zuozhuan_train_utf8.txt (zip member)"] = member_meta
    for name, letter in (("test_a", "a"), ("test_b", "b")):
        gold_path = source / f"EvaHan_test{letter}_gold.txt"
        raw_path = source / f"EvaHan_test{letter}_raw.txt"
        gold_bytes, raw_bytes = gold_path.read_bytes(), raw_path.read_bytes()
        gold_text, raw_text = gold_bytes.decode("utf-8-sig"), raw_bytes.decode("utf-8-sig")
        rows, errors = _parse(gold_text, name, gold_path.name)
        gold_lines, physical_raw = gold_text.splitlines(), raw_text.splitlines()
        if len(gold_lines) != len(physical_raw):
            raise ValueError(f"Physical raw/gold line count mismatch in {name}")
        by_line = {r["line"]: r["text"] for r in rows}
        mismatches = [i for i, raw in enumerate(physical_raw, 1)
                      if raw != by_line.get(i, "")]
        if mismatches:
            raise ValueError(f"Raw/gold character mismatch in {name}, physical lines {mismatches}")
        samples[name] = rows
        raw_lines[name] = physical_raw
        validation[name] = {"raw_gold_physical_lines_equal": True,
                            "raw_gold_character_mismatches": [], "anomalies": errors,
                            "blank_physical_lines": [i for i, t in enumerate(physical_raw, 1) if not t.strip()],
                            "raw_characters": sum(len(line) for line in physical_raw)}
        source_manifest[gold_path.name] = _file_meta(gold_path, gold_bytes)
        source_manifest[raw_path.name] = _file_meta(raw_path, raw_bytes)
    scorer_zip = source / "EvaHan_scorer.zip"
    with zipfile.ZipFile(scorer_zip) as zf:
        corrupt = zf.testzip()
        if corrupt:
            raise ValueError(f"Corrupt scorer ZIP entry: {corrupt}")
        zip_entries = [{"name": info.filename, "bytes": info.file_size,
                        "sha256": _sha(zf.read(info.filename))} for info in zf.infolist()]
    source_manifest[scorer_zip.name] = _file_meta(scorer_zip)
    for name in ("eval_EvaHan_2022.py", "eval_EvaHan_2022_FINAL.py"):
        source_manifest[name] = _file_meta(source / name)
    stats = {name: _stats(rows) for name, rows in samples.items()}
    duplicate_report = _duplicates({name: samples[name] for name in
                                   ("original_train", "test_a", "test_b", "train", "dev")})
    if duplicate_report["across"]["train__dev"]["common_unique_texts"]:
        raise AssertionError("Exact text leaked across the deterministic train/dev group split")
    report = {
        "schema_version": SCHEMA_VERSION, "source_directory": str(source),
        "source_manifest": source_manifest,
        "zip_crc": {"zuozhuan_train_utf8.zip": "passed", "EvaHan_scorer.zip": "passed"},
        "scorer_zip_entries": zip_entries,
        "quarantine": quarantine,
        "quarantine_protocol": "Normalize only sv/vs, wv/vw, yv/vy aliases; isolate whole training rows with malformed tokens or other noncanonical POS. Do not guess corrections.",
        "expected_quarantine_lines": list(EXPECTED_QUARANTINE_LINES),
        "observed_quarantine_lines": [row["line"] for row in quarantine],
        "expected_quarantine_matches": [row["line"] for row in quarantine] == list(EXPECTED_QUARANTINE_LINES),
        "test_validation": validation, "stats": stats,
        "threshold_diagnostic": _threshold(stats), "figure4_comparison": _figure4(stats),
        "duplicates": duplicate_report,
        "split": {"seed_text": "2022", "group_key": "exact raw text",
                  "rule": "int(SHA256(UTF8('2022\\0' + raw_text)), 16) % 10 == 0 -> dev; otherwise train",
                  "train_ids": [r["id"] for r in train], "dev_ids": [r["id"] for r in dev],
                  "train_rows": len(train), "dev_rows": len(dev),
                  "train_dev_overlap_unique_texts": 0},
        "tracks": {"paper15": list(PAPER_POS15), "native21": list(CANONICAL_POS21),
                   "paper15_mapping": "All labels outside the retained 14, including test vn, map to uppercase X in derived evaluation only.",
                   "native21_test_vn": "Retain original gold vn; a model with 21 classes cannot match this one item.",
                   "NULL": "Added separately as model class zero; absent from these POS inventories."},
        "counting_notes": [
            "Physical lines use UTF-8-sig decoding and splitlines(); blank lines are not training examples.",
            "Word percentages exclude malformed no-slash items. Surface characters include their literal text in original-train diagnostics.",
            "Original train: 8924 CRLF / 8924 splitlines / 8699 nonempty / 225 blank. A terminal split() empty element is not another physical line.",
            "The training BOM adds one code point under plain utf-8; all operational text uses utf-8-sig.",
            "Exact short formulaic overlap is not proof of augmentation leakage; no fuzzy or passage-level overlap claim is made.",
        ],
        "scorer_risks": [
            "FINAL differs from the standalone non-FINAL mainly by utf-8-sig input; ZIP contains another implementation.",
            "Validated original TestA/TestB gold self-evaluation returns all P/R/F = 1.0 (separate read-only audit).",
            "A final multi-character word without a following word can raise IndexError.",
            "Zero correctly tagged words can cause division by zero.",
            "Equal-length different-character inputs may not advance the alignment loop.",
            "Validate per-line exact characters, offsets, full coverage, and allowed predicted POS before invoking the scorer; use an independent exact-span scorer too.",
            "The CLI always writes eval_outcome_*; execute in an experiment directory with a prediction basename, with a timeout.",
        ],
    }
    return report, {"samples": samples, "raw_lines": raw_lines}


def _audit_markdown(report: dict) -> str:
    lines = ["# EvaHan 数据审计", "", "此文件由 `python -m gef.data audit/prepare` 确定性生成；原件保持不变。", "",
             "| 数据 | 非空行 | 合法词/POS | 表面字符 | p50 | p90 | p95 | p99 | 最长 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, stats in report["stats"].items():
        length = stats["line_length_characters"]
        lines.append("| " + " | ".join(str(x) for x in (
            name, stats["nonempty_lines"], stats["valid_word_pos_items"], stats["surface_characters"],
            length["p50"], length["p90"], length["p95"], length["p99"], length["max"])) + " |")
    lines += ["", f"整行隔离训练异常：{len(report['quarantine'])} 行；行号 " +
              ", ".join(str(r["line"]) for r in report["quarantine"]) + "。",
              "", "仅规范官方明确别名 sv→vs、wv→vw、yv→vy。测试原文和原 gold 不改；TestA 的 vn 保留。",
              "", "固定 paper15 对非原14保留标签映射 X（包括测试 vn，仅派生评测）；native21 保留 vn 并正常计错。",
              "", "图4比例匹配词频，图中 TestA/TestB 环标签与本地官方文件对应相反；0.2% 规则仅作诊断，不决定固定15类。",
              "", "训练/开发按完全相同原文分组，以 SHA256('2022\\0'+原文) 大整数模10为0划入开发集；增强词池仅来自训练集。",
              "", "句长为非空物理行字符数；BOM不计入字符；无需修复两个测试集 raw/gold。长句不应截断后评测。",
              "", "完整异常上下文、标签分布、文件SHA256、重复与重叠、逐条划分和scorer边界风险见相邻 JSON 报告。", ""]
    return "\n".join(lines)


def _save_reports(root: Path, report: dict) -> None:
    destination = root / "reports"
    _write_json(destination / "data_audit.json", report)
    for filename, key in (("quarantine.json", "quarantine"), ("split.json", "split"),
                          ("data_stats.json", "stats"), ("figure4_comparison.json", "figure4_comparison"),
                          ("threshold_diagnostic.json", "threshold_diagnostic"), ("duplicates.json", "duplicates")):
        _write_json(destination / filename, report[key])
    (destination / "data_audit.md").write_text(_audit_markdown(report), encoding="utf-8", newline="\n")


def audit(root: str | Path = DEFAULT_ROOT, source_dir: str | Path | None = None) -> dict:
    root = Path(root).resolve()
    report, collected = _collect(root, Path(source_dir) if source_dir else None)
    augmented, _, pool_report = _augment(collected["samples"]["train"])
    report["augmentation"] = {k: v for k, v in pool_report.items() if k != "pools"}
    report["stats"]["train_augmented"] = _stats(augmented)
    _save_reports(root, report)
    return report


def prepare(root: str | Path = DEFAULT_ROOT, source_dir: str | Path | None = None) -> dict:
    root = Path(root).resolve()
    report, collected = _collect(root, Path(source_dir) if source_dir else None)
    samples = collected["samples"]
    augmented, augmentation_log, pool_report = _augment(samples["train"])
    prepared = root / "data" / "prepared"
    for name in ("train", "dev", "test_a", "test_b"):
        _write_jsonl(prepared / f"{name}.jsonl", samples[name])
    _write_jsonl(prepared / "train_augmented.jsonl", augmented)
    for name, lines in collected["raw_lines"].items():
        _write_json(prepared / f"{name}_raw_lines.json", lines)
    _write_jsonl(root / "reports" / "augmentation.jsonl", augmentation_log)
    _write_json(root / "reports" / "augmentation_pools.json", pool_report)
    report["augmentation"] = {k: v for k, v in pool_report.items() if k != "pools"}
    report["stats"]["train_augmented"] = _stats(augmented)
    _save_reports(root, report)
    products = [prepared / f"{name}.jsonl" for name in
                ("train", "dev", "train_augmented", "test_a", "test_b")]
    products += [prepared / f"{name}_raw_lines.json" for name in ("test_a", "test_b")]
    manifest = {
        "schema_version": SCHEMA_VERSION, "seed": SEED,
        "source_manifest": report["source_manifest"],
        "protocols": {"aliases": POS_ALIASES, "quarantine": report["quarantine_protocol"],
                      "split": report["split"], "tracks": report["tracks"],
                      "augmentation": report["augmentation"],
                      "long_lines": "No truncation or segmentation in prepared data; full original lines retained.",
                      "raw_lines": "UTF-8-sig splitlines strings preserving physical blank lines; sample.line is 1-based."},
        "counts": {name: {key: value[key] for key in
                          ("nonempty_lines", "valid_word_pos_items", "surface_characters")}
                   for name, value in report["stats"].items()},
        "products": {str(p.relative_to(root)): {"sha256": _sha(p.read_bytes()), "bytes": p.stat().st_size}
                     for p in products},
        "reports": {str(p.relative_to(root)): {"sha256": _sha(p.read_bytes()), "bytes": p.stat().st_size}
                    for p in sorted((root / "reports").glob("*.json*"))
                    if p.name in {"data_audit.json", "quarantine.json", "split.json", "data_stats.json",
                                  "figure4_comparison.json", "threshold_diagnostic.json", "duplicates.json",
                                  "augmentation.jsonl", "augmentation_pools.json"}},
    }
    _write_json(prepared / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "prepare"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="code root; outputs are data/prepared and reports")
    parser.add_argument("--source-dir", type=Path,
                        help="optional explicit original-data directory (read-only)")
    args = parser.parse_args(argv)
    result = audit(args.root, args.source_dir) if args.command == "audit" else prepare(args.root, args.source_dir)
    if args.command == "audit":
        summary = {"report": str(args.root / "reports/data_audit.json"),
                   "quarantined_rows": len(result["quarantine"]),
                   "stats": {k: v["nonempty_lines"] for k, v in result["stats"].items()}}
    else:
        summary = {"manifest": str(args.root / "data/prepared/manifest.json"), "counts": result["counts"]}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
