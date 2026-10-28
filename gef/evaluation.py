"""Strict line-aware span scoring and an isolated, unmodified EvaHan scorer.

All displayed scores are percentages.  The independent scorer is authoritative
for checkpoint selection; official-script failures are recorded, never repaired.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable


METRIC_NAMES = ("WP", "WR", "WF", "PP", "PR", "PF")


class FormatError(ValueError):
    """A tagged file does not preserve a valid sentence/character sequence."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_tagged_line(line, line_number=1):
    if not line.strip():
        return {"id": str(line_number), "line": line_number, "text": "", "words": []}
    if any(ch.isspace() and ch != " " for ch in line):
        raise FormatError(f"line {line_number}: only ASCII spaces may delimit tagged words")
    words, offset = [], 0
    for token in line.split():
        if token.count("/") != 1:
            raise FormatError(f"line {line_number}: expected one word/POS separator: {token!r}")
        text, pos = token.split("/")
        if not text or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", pos):
            raise FormatError(f"line {line_number}: invalid word/POS token: {token!r}")
        words.append({"text": text, "start": offset, "end": offset + len(text), "pos": pos})
        offset += len(text)
    return {"id": str(line_number), "line": line_number, "text": "".join(w["text"] for w in words), "words": words}


def read_tagged(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return lines, [parse_tagged_line(line, i + 1) for i, line in enumerate(lines)]


def validate_samples(samples, *, prediction=False):
    ids = set()
    for i, sample in enumerate(samples):
        identity = str(sample.get("id", sample.get("line", i + 1)))
        if identity in ids:
            raise FormatError(f"duplicate sample id: {identity}")
        ids.add(identity)
        text, end = sample["text"], 0
        for word in sample["words"]:
            start, stop, pos = word["start"], word["end"], word["pos"]
            if not isinstance(start, int) or not isinstance(stop, int) or start != end or not start < stop <= len(text):
                raise FormatError(f"sample {identity}: spans must partition the text without gaps/overlaps")
            if "text" in word and word["text"] != text[start:stop]:
                raise FormatError(f"sample {identity}: word text and character offsets disagree")
            if not isinstance(pos, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", pos):
                raise FormatError(f"sample {identity}: invalid POS tag {pos!r}")
            if any(ch.isspace() or ch == "/" for ch in text[start:stop]):
                raise FormatError(f"sample {identity}: word cannot be represented in the official format")
            end = stop
        if end != len(text):
            raise FormatError(f"sample {identity}: word spans do not cover all characters")
    return True


def validate_pair(gold, prediction):
    validate_samples(gold)
    validate_samples(prediction, prediction=True)
    if len(gold) != len(prediction):
        raise FormatError(f"physical line count differs: gold={len(gold)}, prediction={len(prediction)}")
    for i, (reference, output) in enumerate(zip(gold, prediction), 1):
        if reference["text"] != output["text"]:
            raise FormatError(f"line {i}: character sequence differs (including physical blank-line order)")


def _prf(correct, predicted, gold):
    precision = 100.0 * correct / predicted if predicted else 0.0
    recall = 100.0 * correct / gold if gold else 0.0
    f1 = 200.0 * correct / (predicted + gold) if predicted + gold else 0.0
    return precision, recall, f1


def _score_subset(gold, prediction, indices):
    gold_spans, predicted_spans, gold_pos, predicted_pos = set(), set(), set(), set()
    for i in indices:
        for word in gold[i]["words"]:
            span = (i, word["start"], word["end"])
            gold_spans.add(span)
            gold_pos.add(span + (word["pos"],))
        for word in prediction[i]["words"]:
            span = (i, word["start"], word["end"])
            predicted_spans.add(span)
            predicted_pos.add(span + (word["pos"],))
    values = _prf(len(gold_spans & predicted_spans), len(predicted_spans), len(gold_spans))
    values += _prf(len(gold_pos & predicted_pos), len(predicted_pos), len(gold_pos))
    return {**dict(zip(METRIC_NAMES, values)), "gold_words": len(gold_spans),
            "predicted_words": len(predicted_spans), "correct_spans": len(gold_spans & predicted_spans),
            "correct_pos_spans": len(gold_pos & predicted_pos), "sentences": len(indices)}


def training_vocabulary(samples):
    return {sample["text"][word["start"]:word["end"]] for sample in samples for word in sample["words"]}


def score_samples(gold, prediction, train_vocabulary=None):
    """Require exact physical alignment, then match end-exclusive word spans."""
    validate_pair(gold, prediction)
    indices = [i for i, sample in enumerate(gold) if sample["text"]]
    result = _score_subset(gold, prediction, indices)
    tags = sorted({w["pos"] for s in gold + prediction for w in s["words"]})
    per_pos = {}
    for tag in tags:
        target = {(i, w["start"], w["end"]) for i, s in enumerate(gold) for w in s["words"] if w["pos"] == tag}
        output = {(i, w["start"], w["end"]) for i, s in enumerate(prediction) for w in s["words"] if w["pos"] == tag}
        pp, pr, pf = _prf(len(target & output), len(output), len(target))
        per_pos[tag] = {"PP": pp, "PR": pr, "PF": pf, "gold_words": len(target),
                        "predicted_words": len(output), "correct_pos_spans": len(target & output)}
    result["per_pos"] = per_pos
    result["by_sentence_length"] = {}
    for name, lower, upper in (("1-64", 1, 64), ("65-128", 65, 128), ("129-256", 129, 256), ("257+", 257, float("inf"))):
        bucket = [i for i in indices if lower <= len(gold[i]["text"]) <= upper]
        result["by_sentence_length"][name] = _score_subset(gold, prediction, bucket)
    if train_vocabulary is not None:
        result["vocabulary"] = {"reference": "original training subset only; excludes dev/test and augmentation", "types": len(train_vocabulary)}
        for name, oov in (("oov", True), ("iv", False)):
            total = correct_word = correct_pos = 0
            for reference, output in zip(gold, prediction):
                output_spans = {(w["start"], w["end"]) for w in output["words"]}
                output_pos = {(w["start"], w["end"], w["pos"]) for w in output["words"]}
                for word in reference["words"]:
                    text = reference["text"][word["start"]:word["end"]]
                    if (text not in train_vocabulary) == oov:
                        total += 1
                        correct_word += (word["start"], word["end"]) in output_spans
                        correct_pos += (word["start"], word["end"], word["pos"]) in output_pos
            result[name] = {"gold_words": total, "correct_spans": correct_word, "correct_pos_spans": correct_pos,
                            "segmentation_recall": 100.0 * correct_word / total if total else 0.0,
                            "pos_recall": 100.0 * correct_pos / total if total else 0.0}
    return result


def write_tagged(path, samples, raw_lines=None):
    """Use sample.line to restore empty physical lines; never concatenate books."""
    validate_samples(samples)
    lines = [] if raw_lines is None else [""] * len(raw_lines)
    occupied = set()
    for sample in samples:
        line = " ".join(f"{sample['text'][w['start']:w['end']]}/{w['pos']}" for w in sample["words"])
        if raw_lines is None:
            lines.append(line)
        else:
            index = int(sample["line"]) - 1
            if index in occupied or not 0 <= index < len(lines) or not raw_lines[index].strip():
                raise FormatError(f"invalid or repeated original physical line: {index + 1}")
            occupied.add(index)
            original_line = raw_lines[index]
            # Prepared raw-lines manifests contain the original unsegmented
            # characters; tagged manifests are accepted for audit roundtrips.
            original_text = parse_tagged_line(original_line, index + 1)["text"] if "/" in original_line else original_line
            if original_text != sample["text"]:
                raise FormatError(f"original characters differ at physical line {index + 1}")
            lines[index] = line
    if raw_lines is not None and occupied != {i for i, line in enumerate(raw_lines) if line.strip()}:
        raise FormatError("prediction omits original nonempty physical lines")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
    return path


def run_official(prediction_path, gold_path, scorer_path, output_dir, timeout=120):
    """Copy the original scorer byte-for-byte; use relative filenames in its own cwd."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "not_run", "scorer": str(Path(scorer_path).resolve()), "timeout_seconds": timeout}
    try:
        report["scorer_sha256"] = sha256_file(scorer_path)
        with tempfile.TemporaryDirectory(prefix="official-", dir=output_dir) as temporary:
            work = Path(temporary)
            shutil.copyfile(scorer_path, work / "eval_EvaHan_2022_FINAL.py")
            shutil.copyfile(prediction_path, work / "prediction.txt")
            shutil.copyfile(gold_path, work / "gold.txt")
            command = [sys.executable, "eval_EvaHan_2022_FINAL.py", "prediction.txt", "gold.txt"]
            report["command"] = command
            environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            completed = subprocess.run(command, cwd=work, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", timeout=timeout, env=environment)
            stdout, stderr = completed.stdout, completed.stderr
            report["returncode"] = completed.returncode
            report["status"] = "ok" if completed.returncode == 0 else "error"
            outcome = work / "eval_outcome_prediction.txt"
            if outcome.exists():
                shutil.copyfile(outcome, output_dir / "eval_outcome_prediction.txt")
            if completed.returncode == 0:
                rows = re.findall(r"\|\s*(?:Word segmentation|Pos tagging)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", stdout)
                if len(rows) == 2:
                    report["metrics"] = dict(zip(METRIC_NAMES, [float(v) for row in rows for v in row]))
                else:
                    report.update(status="unparsed_output", error="official process returned 0 without two parseable metric rows")
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or "", error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        report.update(status="timeout", error=str(error))
    except (OSError, ValueError) as error:
        stdout, stderr = "", str(error)
        report.update(status="error", error=str(error))
    (output_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    report["stdout_path"] = str((output_dir / "stdout.txt").resolve())
    report["stderr_path"] = str((output_dir / "stderr.txt").resolve())
    save_json(output_dir / "result.json", report)
    return report


def evaluate_files(prediction_path, gold_path, output_dir, *, scorer_path=None,
                   train_vocabulary=None, timeout=120, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"prediction": str(Path(prediction_path).resolve()), "gold": str(Path(gold_path).resolve()),
              "metadata": metadata or {}, "status": "invalid_format"}
    try:
        report["prediction_sha256"] = sha256_file(prediction_path)
        report["gold_sha256"] = sha256_file(gold_path)
        _, gold = read_tagged(gold_path)
        _, prediction = read_tagged(prediction_path)
        report["independent"] = score_samples(gold, prediction, train_vocabulary)
        report["status"] = "ok"
    except (FormatError, OSError, UnicodeError) as error:
        report["error"] = str(error)
        report["official"] = {"status": "skipped_invalid_format"}
    else:
        if scorer_path:
            official = run_official(prediction_path, gold_path, scorer_path, output_dir / "official", timeout)
            report["official"] = official
            if official["status"] == "ok":
                differences = {name: official["metrics"][name] - report["independent"][name] for name in METRIC_NAMES}
                report["official_minus_independent"] = differences
                report["official_agrees"] = all(abs(value) <= 0.00011 for value in differences.values())
                if not report["official_agrees"]:
                    report["status"] = "scorer_disagreement"
            else:
                report["status"] = "official_error"
        else:
            report["official"] = {"status": "not_requested"}
    save_json(output_dir / "metrics.json", report)
    return report
