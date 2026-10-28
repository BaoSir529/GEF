import json
from pathlib import Path
import tempfile
import unittest

from gef.evaluation import (FormatError, evaluate_files, parse_tagged_line,
                            read_tagged, score_samples, write_tagged)


class EvaluationTests(unittest.TestCase):
    def test_segmentation_and_pos_have_distinct_exact_span_counts(self):
        gold = [parse_tagged_line("仁/n 義/n 。/w")]
        output = [parse_tagged_line("仁義/nr 。/w")]
        score = score_samples(gold, output, {"仁", "。"})
        self.assertEqual(score["correct_spans"], 1)
        self.assertEqual(score["gold_words"], 3)
        self.assertEqual(score["predicted_words"], 2)
        self.assertAlmostEqual(score["WF"], 40.0)
        self.assertAlmostEqual(score["PF"], 40.0)
        self.assertEqual(score["oov"]["gold_words"], 1)
        self.assertEqual(score["oov"]["correct_pos_spans"], 0)

    def test_pos_error_keeps_segmentation_correct(self):
        gold = [parse_tagged_line("民/n 力/n")]
        output = [parse_tagged_line("民/v 力/n")]
        score = score_samples(gold, output)
        self.assertEqual(score["WF"], 100)
        self.assertEqual(score["PF"], 50)
        self.assertEqual(score["per_pos"]["v"]["gold_words"], 0)

    def test_character_and_blank_line_alignment_are_strict(self):
        for gold, prediction in ((["甲/n", "", "乙/n"], ["甲/n", "乙/n"]),
                                 (["甲/n"], ["乙/n"]),
                                 (["甲/n", "", "乙/n"], ["甲/n", "乙/n", ""])):
            with self.assertRaises(FormatError):
                score_samples([parse_tagged_line(s, i + 1) for i, s in enumerate(gold)],
                              [parse_tagged_line(s, i + 1) for i, s in enumerate(prediction)])

    def test_invalid_tokens_are_rejected_before_official_script(self):
        for text in ("甲", "/n", "甲/", "甲/n/v", "甲/n\t乙/v"):
            with self.subTest(text=text), self.assertRaises(FormatError):
                parse_tagged_line(text)

    def test_physical_blanks_and_end_exclusive_spans_roundtrip(self):
        raw = ["", "甲/n 乙/v", "", "。/w", ""]
        samples = [parse_tagged_line(raw[i], i + 1) for i in (1, 3)]
        with tempfile.TemporaryDirectory() as temporary:
            path = write_tagged(Path(temporary) / "prediction.txt", samples, raw)
            lines, parsed = read_tagged(path)
            self.assertEqual(lines, raw)
            self.assertEqual(parsed[1]["words"][1]["start"], 1)
            self.assertEqual(parsed[1]["words"][1]["end"], 2)
            write_tagged(path, samples, ["", "甲乙", "", "。", ""])
            self.assertEqual(read_tagged(path)[0], raw)
            with self.assertRaises(FormatError):
                write_tagged(path, samples[:1], raw)

    def test_official_original_success_and_failure_are_preserved(self):
        original = Path(__file__).resolve().parents[1] / "data/official_snapshot/eval_EvaHan_2022_FINAL.py"
        if not original.is_file():
            self.skipTest("Download the external official scorer with scripts/fetch_assets.py first")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            gold, prediction = folder / "gold.txt", folder / "prediction.txt"
            gold.write_text("甲/n 。/w\n", encoding="utf-8")
            prediction.write_text("甲/n 。/w\n", encoding="utf-8")
            good = evaluate_files(prediction, gold, folder / "ok", scorer_path=original)
            self.assertEqual(good["status"], "ok")
            self.assertTrue(good["official_agrees"])
            # The original has an out-of-range access for a final multi-char
            # word. Preserve the failure and the valid independent score.
            gold.write_text("甲乙/n\n", encoding="utf-8")
            prediction.write_text("甲乙/n\n", encoding="utf-8")
            failed = evaluate_files(prediction, gold, folder / "failed", scorer_path=original)
            self.assertEqual(failed["independent"]["PF"], 100)
            self.assertEqual(failed["status"], "official_error")
            self.assertNotEqual(failed["official"]["returncode"], 0)
            self.assertIn("IndexError", Path(failed["official"]["stderr_path"]).read_text(encoding="utf-8"))

    def test_invalid_format_saves_flag_and_skips_official(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            gold, prediction = folder / "gold.txt", folder / "prediction.txt"
            gold.write_text("甲/n\n", encoding="utf-8")
            prediction.write_text("乙/n\n", encoding="utf-8")
            report = evaluate_files(prediction, gold, folder / "out", scorer_path="missing.py")
            self.assertEqual(report["official"]["status"], "skipped_invalid_format")
            self.assertEqual(json.loads((folder / "out/metrics.json").read_text(encoding="utf-8"))["status"], "invalid_format")


if __name__ == "__main__":
    unittest.main()
