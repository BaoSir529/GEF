"""Data protocol tests, independent of the training stack and network."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from gef import data


class DataProtocolTests(unittest.TestCase):
    def test_fixed_pos_tracks_and_unknown_gold(self):
        self.assertEqual(len(data.CANONICAL_POS21), 21)
        self.assertEqual(len(set(data.CANONICAL_POS21)), 21)
        self.assertEqual(len(data.pos_tags("paper15")), 15)
        self.assertNotIn("NULL", data.pos_tags("paper15"))
        self.assertEqual(data.map_pos("sv", "native21"), "vs")
        self.assertEqual(data.map_pos("wv", "native21"), "vw")
        self.assertEqual(data.map_pos("yv", "native21"), "vy")
        self.assertEqual(data.map_pos("vn", "native21"), "vn")
        self.assertEqual(data.map_pos("vn", "paper15"), "X")
        self.assertEqual(data.map_pos("f", "paper15"), "X")
        self.assertEqual(data.map_pos("j", "paper15"), "X")
        self.assertEqual(data.map_pos("n", "paper15"), "n")
        with self.assertRaises(ValueError):
            data.pos_tags("reestimated15")

    def test_parse_physical_lines_offsets_and_anomalies(self):
        rows, errors = data._parse("\r\n貴/sv 名/n 。/w\r\n禰.r 。\r\n人/mr\r\n", "train", "fixture")
        self.assertEqual([r["line"] for r in rows], [2, 3, 4])
        self.assertEqual(rows[0]["id"], "train:2")
        self.assertEqual(rows[0]["text"], "貴名。")
        self.assertEqual(rows[0]["words"][0],
                         {"text": "貴", "start": 0, "end": 1, "pos": "vs", "original_pos": "sv"})
        self.assertEqual(rows[0]["words"][-1]["end"], len(rows[0]["text"]))
        self.assertEqual(len(errors), 2)
        self.assertEqual(len(errors[0]["errors"]), 2)
        self.assertEqual(errors[0]["raw_annotated_line"], "禰.r 。")
        self.assertEqual(errors[1]["errors"][0]["reason"], "noncanonical_pos")

    def test_split_exact_hash_definition_and_group_integrity(self):
        for text in ("公從之。", "甲", "", "𠀀人"):
            expected = "dev" if int(hashlib.sha256(("2022\0" + text).encode("utf-8")).hexdigest(), 16) % 10 == 0 else "train"
            self.assertEqual(data.group_split(text), expected)
            self.assertEqual(data.group_split(text), data.group_split(text))

    def test_augmentation_train_only_one_word_no_punctuation_and_offsets(self):
        train, _ = data._parse("甲/n 行/v 。/w\n乙/n 止/v 。/w\n長名字/n 行/v 。/w", "train", "fixture")
        original = copy.deepcopy(train)
        augmented, log, pools = data._augment(train)
        self.assertEqual(train, original)
        self.assertEqual(len(augmented), 2 * len(train))
        self.assertNotIn("w", pools["pools"])
        for source, generated, record in zip(train, augmented[len(train):], log):
            differences = [i for i, (a, b) in enumerate(zip(source["words"], generated["words"]))
                           if a["text"] != b["text"]]
            self.assertEqual(len(differences), 1)
            self.assertTrue(record["replaced"])
            self.assertEqual(record["word_index"], differences[0])
            self.assertNotEqual(source["words"][differences[0]]["pos"], "w")
            self.assertEqual(generated["id"], source["id"] + ":aug")
            self.assertEqual(generated["split"], "train")
            offset = 0
            for word in generated["words"]:
                self.assertEqual(word["start"], offset)
                self.assertEqual(generated["text"][word["start"]:word["end"]], word["text"])
                offset = word["end"]
            self.assertEqual(offset, len(generated["text"]))
        self.assertEqual((augmented, log, pools), data._augment(train))

    def test_augmentation_without_alternative_keeps_logged_copy(self):
        train, _ = data._parse("甲/n 。/w", "train", "fixture")
        augmented, log, pools = data._augment(train)
        self.assertEqual(len(augmented), 2)
        self.assertEqual(augmented[1]["text"], train[0]["text"])
        self.assertTrue(augmented[1]["augmented"])
        self.assertFalse(log[0]["replaced"])
        self.assertEqual(pools["unchanged_copy_rows"], 1)

    def test_duplicate_report_retains_annotation_variants(self):
        train, _ = data._parse("甲/n 。/w\n甲/v 。/w\n乙/n 。/w", "train", "fixture")
        test, _ = data._parse("甲/n 。/w", "test_a", "fixture")
        report = data._duplicates({"train": train, "test_a": test})
        self.assertEqual(report["within"]["train"]["duplicate_unique_texts"], 1)
        self.assertEqual(report["within"]["train"]["conflicting_annotation_texts"], 1)
        self.assertEqual(report["across"]["train__test_a"]["left_occurrences"], 2)

    def test_threshold_diagnostic_does_not_choose_paper15(self):
        rows, _ = data._parse("甲/n 乙/f 丙/j 丁/q 。/w", "train", "fixture")
        report = data._threshold({"tiny": data._stats(rows)})["tiny"]
        self.assertIn("f", report["retained_original_labels"])
        self.assertIn("j", report["retained_original_labels"])
        self.assertNotIn("f", data.pos_tags("paper15"))
        self.assertNotIn("j", data.pos_tags("paper15"))
        self.assertEqual(data._threshold({"empty": data._stats([])})["empty"]["derived_label_count"], 0)


class EndToEndDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / data.SOURCE_RELATIVE
        self.source.mkdir(parents=True)
        lines = [f"人{i}/n 行/v 。/w" for i in range(40)]
        lines += [lines[0], "錯/mr 。/w", "禰.r 。/w", "貴/sv 。/w"]
        with zipfile.ZipFile(self.source / "zuozhuan_train_utf8.zip", "w") as archive:
            archive.writestr("zuozhuan_train_utf8.txt", ("\ufeff" + "\r\n".join(lines) + "\r\n\r\n").encode("utf-8"))
        fixtures = {
            "a": ("甲/n 。/w\r\n\r\n福/vn 。/w", "\ufeff甲。\r\n\r\n福。\r\n"),
            "b": ("史/n 記/v 。/w\r\n", "\ufeff史記。\r\n"),
        }
        for letter, (gold, raw) in fixtures.items():
            (self.source / f"EvaHan_test{letter}_gold.txt").write_bytes(gold.encode("utf-8"))
            (self.source / f"EvaHan_test{letter}_raw.txt").write_bytes(raw.encode("utf-8"))
        with zipfile.ZipFile(self.source / "EvaHan_scorer.zip", "w") as archive:
            archive.writestr("eval_EvaHan_2022.py", "# fixture\n")
        for name in ("eval_EvaHan_2022.py", "eval_EvaHan_2022_FINAL.py"):
            (self.source / name).write_bytes(b"# fixture\n")

    def tearDown(self):
        self.temporary.cleanup()

    def test_prepare_is_reproducible_preserves_sources_and_line_numbers(self):
        source_bytes = {p.name: p.read_bytes() for p in self.source.iterdir()}
        first = data.prepare(self.root)
        prepared = self.root / "data/prepared"
        products = {p.name: p.read_bytes() for p in prepared.iterdir()}
        reports = {p.name: p.read_bytes() for p in (self.root / "reports").iterdir()}
        self.assertEqual(first, data.prepare(self.root))
        self.assertEqual(products, {p.name: p.read_bytes() for p in prepared.iterdir()})
        self.assertEqual(reports, {p.name: p.read_bytes() for p in (self.root / "reports").iterdir()})
        self.assertEqual(source_bytes, {p.name: p.read_bytes() for p in self.source.iterdir()})
        train, dev = data.load_jsonl(prepared / "train.jsonl"), data.load_jsonl(prepared / "dev.jsonl")
        self.assertTrue(train)
        self.assertTrue(dev)
        self.assertFalse({r["text"] for r in train} & {r["text"] for r in dev})
        self.assertFalse(any(w["pos"] == "mr" for r in train + dev for w in r["words"]))
        self.assertTrue(any(w["pos"] == "vs" for r in train + dev for w in r["words"]))
        self.assertEqual(len(data.load_jsonl(prepared / "train_augmented.jsonl")), 2 * len(train))
        test_a = data.load_jsonl(prepared / "test_a.jsonl")
        self.assertEqual([r["line"] for r in test_a], [1, 3])
        self.assertEqual(test_a[1]["words"][0]["pos"], "vn")
        self.assertEqual(json.loads((prepared / "test_a_raw_lines.json").read_text(encoding="utf-8")), ["甲。", "", "福。"])
        for path, meta in first["products"].items():
            self.assertEqual(hashlib.sha256((self.root / path).read_bytes()).hexdigest(), meta["sha256"])
        pools = json.loads((self.root / "reports/augmentation_pools.json").read_text(encoding="utf-8"))["pools"]
        self.assertEqual(set(pools["n"]), {w["text"] for r in train for w in r["words"] if w["pos"] == "n"})
        self.assertNotIn("甲", pools["n"])
        data.audit(self.root)
        self.assertEqual(reports, {p.name: p.read_bytes() for p in (self.root / "reports").iterdir()})

    def test_raw_gold_mismatch_fails_before_product_writes(self):
        (self.source / "EvaHan_testa_raw.txt").write_bytes("甲。\r\n\r\n錯。\r\n".encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "character mismatch"):
            data.prepare(self.root)
        self.assertFalse((self.root / "data/prepared").exists())


if __name__ == "__main__":
    unittest.main()
