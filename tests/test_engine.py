import importlib.util
import argparse
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from gef.engine import (DEFAULT_CONFIG, capture_rng, character_ids, is_better,
                        load_config, lr_factor, optimizer_groups, parser,
                        restore_rng, seed_everything, train, _load_checkpoint, select_overfit_samples, main, input_signatures, _tests, select_device, _version_info)
from gef.matrix import CONFIG_NAMES, SEEDS, completed_run, tasks, aggregate_completed, validate_preflight
from gef.evaluation import sha256_file


class EngineProtocolTests(unittest.TestCase):
    def test_matrix_preflight_rejects_missing_failed_or_stale_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config = load_config(overrides={"encoder_path": str(folder / "encoder"), "data_dir": str(folder / "data")})
            profile, overfit = folder / "profile.json", folder / "overfit.json"
            with self.assertRaises(RuntimeError):
                validate_preflight(profile, overfit, config)
            from gef.engine import stage_evidence
            evidence = stage_evidence(config)
            first = {"gate_passed": True, "all_required_lengths_tested": True, "tested_lengths_passed": True,
                     "config": config, "evidence": evidence}
            second = {"gate_passed": True, "config": dict(config, seed=42), "samples": list(range(32)),
                      "steps": 100, "evidence": evidence}
            profile.write_text(json.dumps(first), encoding="utf-8")
            overfit.write_text(json.dumps(second), encoding="utf-8")
            self.assertIn("evidence", validate_preflight(profile, overfit, config))
            second["evidence"] = dict(evidence, encoder_metadata_hashes={"vocab.txt": "changed"})
            overfit.write_text(json.dumps(second), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_preflight(profile, overfit, config)
            second["gate_passed"] = False
            overfit.write_text(json.dumps(second), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_preflight(profile, overfit, config)

    def test_resume_signature_detects_same_size_vocabulary_reordering(self):
        with tempfile.TemporaryDirectory() as temporary:
            vocabulary = Path(temporary) / "vocab.txt"
            vocabulary.write_text("甲\n乙\n", encoding="utf-8")
            first = input_signatures({"encoder_path": temporary})
            vocabulary.write_text("乙\n甲\n", encoding="utf-8")
            second = input_signatures({"encoder_path": temporary})
            self.assertNotEqual(first["encoder_metadata_hashes"], second["encoder_metadata_hashes"])
            self.assertEqual(first["implementation_hashes"], second["implementation_hashes"])

    def test_failed_resource_and_overfit_gates_return_nonzero(self):
        for command, status in (("profile", "resource_gate_failed"), ("overfit", "overfit_gate_failed")):
            with patch(f"gef.engine.{command}", return_value={"status": status, "gate_passed": False}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([command]), 2)

    def test_overfit_selection_excludes_trivial_punctuation_and_covers_boundaries(self):
        def sample(identity, text, spans):
            return {"id": identity, "text": text, "words": [{"start": start, "end": end, "pos": pos} for start, end, pos in spans]}
        trivial = sample("punct", "。。。。", [(i, i + 1, "w") for i in range(4)])
        short = sample("short", "甲乙", [(0, 1, "n"), (1, 2, "v")])
        same = sample("same", "仁義民力", [(0, 2, "n"), (2, 4, "n")])
        varied = sample("varied", "孔子曰之", [(0, 2, "nr"), (2, 3, "v"), (3, 4, "r")])
        selected = select_overfit_samples([trivial, short, same, varied], 2)
        self.assertEqual({row["id"] for row in selected}, {"same", "varied"})
        self.assertEqual(selected, select_overfit_samples([varied, same, short, trivial], 2))

    def test_selection_never_uses_test_and_retains_earliest_exact_tie(self):
        best = {"PF": 90.0, "WF": 95.0}
        self.assertTrue(is_better({"PF": 90.01, "WF": 80}, best))
        self.assertTrue(is_better({"PF": 90.0, "WF": 95.01}, best))
        self.assertFalse(is_better({"PF": 90.0, "WF": 95.0}, best))
        self.assertFalse(is_better({"PF": 89.99, "WF": 100}, best))

    def test_matrix_is_exactly_seven_configs_five_seeds(self):
        schedule = tasks()
        self.assertEqual(len(schedule), 35)
        self.assertEqual(len({task["id"] for task in schedule}), 35)
        self.assertEqual(tuple(sorted({task["seed"] for task in schedule})), tuple(sorted(SEEDS)))
        self.assertEqual(schedule[0]["id"], "paper15_gef_aug/seed_42")
        configs = [load_config(Path("configs") / f"{name}.json") for name in CONFIG_NAMES]
        self.assertEqual(sum(not config["augmentation"] for config in configs), 1)
        self.assertEqual({(c["track"], c["model_type"]) for c in configs if c["augmentation"]},
                         {(track, model) for track in ("paper15", "native21") for model in ("gef", "grid_softmax", "sequence_crf")})

    def test_matrix_summary_uses_sample_std_without_cross_track_pooling(self):
        completed = {f"paper15_gef_aug/seed_{seed}": {"metrics": {split: {key: score for key in ("WP", "WR", "WF", "PP", "PR", "PF")}
                                                                                for split in ("test_a", "test_b")}}
                     for seed, score in ((13, 90), (42, 92))}
        summary = aggregate_completed(completed)
        self.assertEqual(summary["paper15_gef_aug"]["test_a"]["PF"]["mean"], 91)
        self.assertAlmostEqual(summary["paper15_gef_aug"]["test_a"]["PF"]["sample_std_ddof1"], 2 ** .5)
        self.assertEqual(summary["native21_gef_aug"]["completed_seeds"], 0)
        self.assertFalse(summary["paper15_gef_aug"]["complete"])

    def test_smoke_result_cannot_be_accepted_as_full_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = {"id": "x/seed_13", "output_dir": temporary}
            Path(temporary, "run_complete.json").write_text(json.dumps({"status": "complete", "subset_smoke_only": True}), encoding="utf-8")
            self.assertIsNone(completed_run(task))

    def test_completion_checks_current_protocol_and_evaluated_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            scorer = folder / "fixture_scorer.py"
            scorer.write_text("# hash-only test fixture; never executed\n", encoding="utf-8")
            config = load_config(overrides={"encoder_path": str(folder / "encoder"),
                                           "data_dir": str(folder / "data"), "seed": 42,
                                           "official_scorer": str(scorer)})
            config.update(input_signatures(config), dataset_hashes={})
            resolved = folder / "resolved_config.json"
            resolved.write_text(json.dumps(config), encoding="utf-8")
            for name in ("best.pt",):
                (folder / name).write_bytes(b"test fixture")
            tests = {}
            for split in ("test_a", "test_b"):
                output = folder / "evaluation" / split
                output.mkdir(parents=True)
                for name in ("prediction", "gold"):
                    (output / f"{name}.txt").write_text("甲/n 。/w\n", encoding="utf-8")
                tests[split] = {"status": "ok", "independent": {key: 100 for key in ("WP", "WR", "WF", "PP", "PR", "PF")},
                                "metadata": {"track": "paper15", "model_type": "gef", "seed": 42,
                                             "subset_smoke_only": False, "test_used_for_selection": False,
                                             "physical_blank_lines_preserved": True},
                                "official": {"scorer_sha256": sha256_file(config["official_scorer"])}}
                for name in ("prediction", "gold"):
                    tests[split][name] = str(output / f"{name}.txt")
                    tests[split][name + "_sha256"] = sha256_file(tests[split][name])
                (output / "metrics.json").write_text(json.dumps(tests[split]), encoding="utf-8")
            result = {"status": "complete", "epochs": 10, "best_checkpoint_reloaded_for_test": True,
                      "resolved_config_sha256": sha256_file(resolved), "best_epoch": 1,
                      "best_dev": {"PF": 99, "WF": 99}, "tests": tests}
            (folder / "run_complete.json").write_text(json.dumps(result), encoding="utf-8")
            task = {"id": "paper15_gef_aug/seed_42", "seed": 42, "output_dir": str(folder)}
            self.assertIsNotNone(completed_run(task, config))
            self.assertIsNone(completed_run(task, dict(config, encoder_window=256)))
            self.assertIsNone(completed_run(task, dict(config, data_dir=str(folder / "other_data"))))
            Path(tests["test_a"]["prediction"]).write_text("甲/v 。/w\n", encoding="utf-8")
            self.assertIsNone(completed_run(task, config))

    def test_native21_evaluation_uses_original_gold_and_preserves_vn(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            sample = {"id": "1", "line": 1, "text": "徼福", "words": [{"start": 0, "end": 2, "pos": "vn"}]}
            for split in ("test_a", "test_b"):
                (folder / f"{split}.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
                (folder / f"{split}_raw_lines.json").write_text(json.dumps(["徼福"], ensure_ascii=False), encoding="utf-8")
            config = load_config(overrides={"data_dir": str(folder), "track": "native21"})
            def fake_decode(model, tokenizer, samples, device, config):
                self.assertEqual(samples[0]["words"][0]["pos"], "vn")
                return [dict(samples[0], words=[{"start": 0, "end": 2, "pos": "n"}])], {}
            with patch("gef.engine._decode", fake_decode), patch("gef.engine.evaluate_files", side_effect=lambda prediction, gold, *args, **kwargs: str(gold)):
                reports = _tests(None, None, None, config, folder / "out", set())
            for split, letter in (("test_a", "a"), ("test_b", "b")):
                expected = Path(config["official_scorer"]).parent / f"EvaHan_test{letter}_gold.txt"
                self.assertEqual(reports[split], str(expected))

    def test_cli_exposes_five_commands_and_explicit_gpu(self):
        for command in ("profile", "overfit", "train"):
            args = parser().parse_args([command, "--device", "cuda:1"])
            self.assertEqual(args.device, "cuda:1")
        self.assertEqual(DEFAULT_CONFIG["device"], "cuda:0")
        with self.assertRaises(ValueError):
            load_config(overrides={"micro_batch_size": 2})

    def test_warmup_and_linear_decay_endpoints(self):
        self.assertEqual(lr_factor(0, 100, 10), 0)
        self.assertEqual(lr_factor(10, 100, 10), 1)
        self.assertAlmostEqual(lr_factor(55, 100, 10), 0.5)
        self.assertEqual(lr_factor(100, 100, 10), 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable in this interpreter")
class EngineTorchTests(unittest.TestCase):
    def test_completed_formal_run_keeps_inference_best_and_removes_only_latest(self):
        import torch
        import torch.nn.functional as F
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.2))
            def forward(self, ids):
                return torch.stack((self.weight.expand(len(ids)), -self.weight.expand(len(ids))), dim=-1)
            def loss_from_emissions(self, emissions, words):
                return F.cross_entropy(emissions, torch.tensor([0 if word["pos"] == "n" else 1 for word in words]))
            def decode_emissions(self, emissions):
                return {"words": [{"start": i, "end": i + 1, "pos": ("n", "v")[int(tag)]}
                                  for i, tag in enumerate(emissions.argmax(-1))]}
        class Vocabulary:
            def convert_tokens_to_ids(self, characters):
                return [1] * len(characters)
        def runtime(config, checkpoint=None):
            model = Tiny()
            if checkpoint:
                model.load_state_dict(checkpoint["model"])
            return model, Vocabulary(), torch.device("cpu"), ["n", "v"]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data = folder / "data"
            data.mkdir()
            samples = [{"id": "train:1", "line": 1, "text": "甲乙",
                        "words": [{"start": 0, "end": 1, "pos": "n"}, {"start": 1, "end": 2, "pos": "v"}]}]
            for name in ("train", "dev"):
                (data / f"{name}.jsonl").write_text(json.dumps(samples[0], ensure_ascii=False) + "\n", encoding="utf-8")
            output = folder / "run"
            output.mkdir()
            unrelated = output / "unrelated.pt"
            unrelated.write_bytes(b"keep")
            config = load_config(overrides={"device": "cpu", "amp": False, "augmentation": False, "epochs": 1,
                                           "effective_batch_size": 1, "data_dir": str(data), "output_dir": str(output),
                                           "encoder_path": "unused"})
            args = argparse.Namespace(train_limit=None, eval_limit=None, resume=None, max_steps=None, skip_test=False, log_every=999)
            def full_tests(model, *unused):
                selected = _load_checkpoint(output / "best.pt")
                torch.testing.assert_close(model.weight, selected["model"]["weight"], rtol=0, atol=0)
                self.assertTrue((output / "latest.pt").exists())
                return {split: {"status": "ok", "independent": {"PF": 100}} for split in ("test_a", "test_b")}
            with patch("gef.engine._runtime", runtime), patch("gef.engine._tests", full_tests):
                result = train(config, args)
            self.assertTrue(result["checkpoint_storage"]["latest_checkpoint_removed"])
            self.assertFalse((output / "latest.pt").exists())
            self.assertEqual(unrelated.read_bytes(), b"keep")
            best = _load_checkpoint(output / "best.pt")
            self.assertEqual(best["checkpoint_type"], "inference")
            self.assertEqual(best["best_epoch"], 1)
            self.assertTrue({"optimizer", "scheduler", "scaler", "rng", "state"}.isdisjoint(best))
            restored, _, _, _ = runtime(config, best)
            torch.testing.assert_close(restored.weight, best["model"]["weight"], rtol=0, atol=0)
            self.assertTrue(json.loads((output / "checkpoint_storage.json").read_text(encoding="utf-8"))["latest_checkpoint_removed"])
            args.resume = str(output / "best.pt")
            with self.assertRaisesRegex(ValueError, "inference checkpoint and cannot resume"):
                train(config, args)

    def test_seed_enforces_strict_determinism_without_tf32(self):
        import torch
        seed_everything(42)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertTrue(torch.backends.cuda.math_sdp_enabled())
        self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
        self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
        self.assertFalse(torch.backends.cuda.cudnn_sdp_enabled())
        version = _version_info()
        self.assertEqual(version["cublas_workspace_config"], ":4096:8")
        self.assertFalse(version["deterministic_algorithms_warn_only"])

    def test_physical_uuid_is_checked_before_cuda_device_binding(self):
        from types import SimpleNamespace
        properties = SimpleNamespace(name="RTX 3080 Ti", uuid=DEFAULT_CONFIG["gpu_uuid"])
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2), \
             patch("torch.cuda.get_device_properties", return_value=properties), patch("torch.cuda.set_device") as bind:
            self.assertEqual(str(select_device(DEFAULT_CONFIG)), "cuda:0")
            bind.assert_called_once()
            bind.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "physical GPU UUID"):
                select_device(dict(DEFAULT_CONFIG, gpu_uuid="GPU-other"))
            bind.assert_not_called()

    def test_training_resume_matches_uninterrupted_weights_scheduler_and_rng(self):
        import torch
        import torch.nn.functional as F
        class Vocabulary:
            def convert_tokens_to_ids(self, characters):
                return [ord(character) % 64 for character in characters]
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Embedding(64, 4)
                self.dropout = torch.nn.Dropout(.3)
                self.head = torch.nn.Linear(4, 2)
            def forward(self, ids):
                return self.head(self.dropout(self.encoder(ids)))
            def loss_from_emissions(self, emissions, words):
                labels = torch.tensor([0 if word["pos"] == "n" else 1 for word in words])
                return F.cross_entropy(emissions, labels)
            def decode_emissions(self, emissions):
                return {"words": [{"start": i, "end": i + 1, "pos": ("n", "v")[int(tag)]}
                                  for i, tag in enumerate(emissions.argmax(-1))]}
        def fake_runtime(config, checkpoint=None):
            model = TinyModel()
            if checkpoint:
                model.load_state_dict(checkpoint["model"])
            return model, Vocabulary(), torch.device("cpu"), ["n", "v"]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data_dir = folder / "data"
            data_dir.mkdir()
            samples = [{"id": f"train:{i}", "line": i + 1, "text": "甲乙",
                        "words": [{"start": 0, "end": 1, "text": "甲", "pos": "n"},
                                  {"start": 1, "end": 2, "text": "乙", "pos": "v"}]} for i in range(6)]
            for name, rows in (("train", samples), ("dev", samples[:2])):
                (data_dir / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            config = load_config(overrides={"device": "cpu", "amp": False, "augmentation": False,
                                           "retain_completed_resume": True,
                                           "epochs": 2, "effective_batch_size": 2, "data_dir": str(data_dir),
                                           "output_dir": str(folder / "continuous"), "encoder_path": "unused"})
            args = argparse.Namespace(train_limit=None, eval_limit=None, resume=None, max_steps=None, skip_test=True, log_every=999)
            with patch("gef.engine._runtime", fake_runtime):
                full = train(config, args)
                interrupted_config = dict(config, output_dir=str(folder / "resumed"))
                args.max_steps = 2
                stopped = train(interrupted_config, args)
                self.assertEqual(stopped["status"], "max_steps")
                args.max_steps, args.resume = None, "auto"
                resumed = train(interrupted_config, args)
            reference = _load_checkpoint(folder / "continuous/latest.pt")
            actual = _load_checkpoint(folder / "resumed/latest.pt")
            for key, value in reference["model"].items():
                torch.testing.assert_close(value, actual["model"][key], rtol=0, atol=0)
            self.assertEqual(reference["scheduler"], actual["scheduler"])
            self.assertEqual(full["best_dev"], resumed["best_dev"])
            self.assertEqual(actual["state"]["optimizer_steps"], 6)
            torch.testing.assert_close(reference["rng"]["torch"], actual["rng"]["torch"], rtol=0, atol=0)

    def test_character_mapping_does_not_call_tokenizer_segmentation(self):
        class VocabularyOnly:
            def convert_tokens_to_ids(self, characters):
                self.characters = characters
                return list(range(len(characters)))
            def __call__(self, *args, **kwargs):
                raise AssertionError("text segmentation must never be called")
        vocabulary = VocabularyOnly()
        ids = character_ids(vocabulary, "臺甲，A", "cpu")
        self.assertEqual(vocabulary.characters, ["臺", "甲", "，", "A"])
        self.assertEqual(ids.tolist(), [0, 1, 2, 3])

    def test_rng_roundtrip_restores_python_numpy_torch(self):
        import numpy as np
        import torch
        seed_everything(13)
        saved = capture_rng()
        expected = (random.random(), float(np.random.random()), torch.rand(4))
        restore_rng(saved)
        actual = (random.random(), float(np.random.random()), torch.rand(4))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_optimizer_separates_encoder_head_and_layernorm_decay(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.ModuleDict({"dense": torch.nn.Linear(2, 2), "LayerNorm": torch.nn.LayerNorm(2)})
                self.head = torch.nn.Linear(2, 2)
        model = Tiny()
        assigned = {id(parameter): (group["lr"], group["weight_decay"])
                    for group in optimizer_groups(model, DEFAULT_CONFIG) for parameter in group["params"]}
        self.assertEqual(assigned[id(model.encoder["dense"].weight)], (2e-5, .01))
        self.assertEqual(assigned[id(model.encoder["dense"].bias)], (2e-5, 0))
        self.assertEqual(assigned[id(model.encoder["LayerNorm"].weight)], (2e-5, 0))
        self.assertEqual(assigned[id(model.head.weight)], (1e-3, .01))


if __name__ == "__main__":
    unittest.main()
