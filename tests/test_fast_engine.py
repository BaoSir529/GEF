"""CPU checks for the opt-in batching engine and its resume protocol."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from gef import fast_engine as fast
from gef.engine import _load_checkpoint


class TinyVocabulary:
    def convert_tokens_to_ids(self, characters):
        return [ord(character) % 64 for character in characters]


class TinyModel(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.encoder = torch.nn.Embedding(64, 4)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(4, 2)

    def forward(self, ids):
        return self.head(self.dropout(self.encoder(ids)))

    def loss_from_emissions(self, emissions, words):
        labels = torch.tensor([0 if word["pos"] == "n" else 1 for word in words], device=emissions.device)
        return F.cross_entropy(emissions, labels)

    def decode_emissions(self, emissions):
        return {"words": [{"start": i, "end": i + 1, "pos": ("n", "v")[int(tag)]}
                          for i, tag in enumerate(emissions.argmax(-1))]}


def samples(lengths=(2, 3, 4, 2, 3, 1)):
    return [{"id": f"train:{i}", "line": i + 1, "text": "甲乙丙丁戊己"[:length],
             "words": [{"start": j, "end": j + 1, "pos": ("n", "v")[(i + j) % 2]} for j in range(length)]}
            for i, length in enumerate(lengths)]


def fake_forward_batch(model, inputs):
    return [model(ids) for ids in inputs]


def fake_runtime(config, checkpoint=None):
    model = TinyModel(dropout=.3)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    return model, TinyVocabulary(), torch.device("cpu"), ["n", "v"]


def args(**changes):
    result = argparse.Namespace(train_limit=None, eval_limit=None, resume=None,
                                max_steps=None, skip_test=True, log_every=999)
    vars(result).update(changes)
    return result


def prepared(folder):
    data = folder / "data"
    data.mkdir()
    rows = samples()
    for name, values in (("train", rows), ("dev", rows[:2])):
        (data / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
    return fast.load_config(overrides={"device": "cpu", "amp": False, "augmentation": False,
                                      "retain_completed_resume": True, "epochs": 2,
                                      "effective_batch_size": 4, "micro_batch_size": 3,
                                      "data_dir": str(data), "encoder_path": "unused"})


def test_fast_loader_records_actual_microbatch_and_rejects_invalid_values():
    config = fast.load_config("configs/paper15_gef_aug.json")
    assert config["micro_batch_size"] == 4
    assert config["effective_batch_size"] == 16
    assert config["compute_protocol"] == fast.COMPUTE_PROTOCOL
    for key in ("micro_batch_size", "max_batch_chars", "max_batch_cells"):
        with pytest.raises(ValueError):
            fast.load_config(overrides={key: 0})
    assert fast.load_config(overrides={"micro_batch_size": 2})["micro_batch_size"] == 2


def test_microbatch_budgets_preserve_order_and_keep_long_samples_complete():
    rows = [{"text": "甲" * length} for length in (4, 5, 100, 511, 825, 2)]
    config = fast.load_config(overrides={"max_batch_chars": 12})
    batches = list(fast.microbatches(rows, list(range(6)), config))
    assert batches == [[0, 1], [2], [3], [4], [5]]
    assert [i for batch in batches for i in batch] == list(range(6))
    config.update(max_batch_chars=5000, max_batch_cells=10000000)
    assert list(fast.microbatches(rows, list(range(6)), config)) == [[0, 1, 2], [3], [4], [5]]


def test_microbatch_loss_gradients_equal_per_sample_mean_including_partial_group():
    torch.manual_seed(13)
    model = TinyModel().double()
    reference = copy.deepcopy(model)
    rows = samples((1, 5, 2, 6, 1))
    config = fast.load_config(overrides={"device": "cpu", "amp": False, "micro_batch_size": 4})
    vocabulary = TinyVocabulary()
    expected = []
    for row in rows:
        ids = fast.character_ids(vocabulary, row["text"], "cpu")
        expected.append(reference.loss_from_emissions(reference(ids), row["words"]))
    torch.stack(expected).mean().backward()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with patch("gef.fast_engine.forward_batch", fake_forward_batch):
        total, groups = fast.train_effective_batch(model, vocabulary, rows, list(range(5)), torch.device("cpu"), config, scaler)
    assert groups == [4, 1]
    assert total / len(rows) == pytest.approx(float(torch.stack(expected).mean().detach()), abs=1e-12)
    for actual, target in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, target.grad, rtol=1e-12, atol=1e-12)


def test_fast_resume_matches_uninterrupted_rng_optimizer_and_weights():
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        config = prepared(folder)
        with patch("gef.fast_engine._runtime", fake_runtime), patch("gef.fast_engine.forward_batch", fake_forward_batch):
            full = fast.train(dict(config, output_dir=str(folder / "full")), args())
            split_config = dict(config, output_dir=str(folder / "resumed"))
            stopped = fast.train(split_config, args(max_steps=1))
            assert stopped["status"] == "max_steps"
            resumed = fast.train(split_config, args(resume="auto"))
        expected = _load_checkpoint(folder / "full/latest.pt")
        actual = _load_checkpoint(folder / "resumed/latest.pt")
        assert actual["config"]["micro_batch_size"] == 3
        assert set(actual["config"]["fast_implementation_hashes"]) == {"fast_engine.py", "batching.py"}
        assert actual["state"]["optimizer_steps"] == 4
        assert expected["scheduler"] == actual["scheduler"]
        assert full["best_dev"] == resumed["best_dev"]
        for key, value in expected["model"].items():
            torch.testing.assert_close(value, actual["model"][key], rtol=0, atol=0)
        torch.testing.assert_close(expected["rng"]["torch"], actual["rng"]["torch"], rtol=0, atol=0)
        assert expected["rng"]["python"] == actual["rng"]["python"]
        for name, value in expected["optimizer"]["state"].items():
            for key, item in value.items():
                torch.testing.assert_close(item, actual["optimizer"]["state"][name][key], rtol=0, atol=0)


def test_fast_resume_rejects_legacy_or_changed_compute_budget():
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        config = prepared(folder)
        config["output_dir"] = str(folder / "run")
        with patch("gef.fast_engine._runtime", fake_runtime), patch("gef.fast_engine.forward_batch", fake_forward_batch):
            fast.train(config, args(max_steps=1))
            with pytest.raises(ValueError, match="resume configuration/data changed"):
                fast.train(dict(config, micro_batch_size=2), args(resume="auto"))
            checkpoint = _load_checkpoint(folder / "run/latest.pt")
            del checkpoint["config"]["compute_protocol"]
            torch.save(checkpoint, folder / "legacy.pt")
            with pytest.raises(ValueError, match="cannot be migrated"):
                fast.train(config, args(resume=str(folder / "legacy.pt")))


def test_stop_request_pauses_only_after_completed_optimizer_step_and_can_resume():
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        config = prepared(folder)
        config["output_dir"] = str(folder / "run")
        calls = []
        def request_after_first_forward(model, inputs):
            calls.append(len(inputs))
            if len(calls) == 1:
                (folder / "run/stop_request.json").write_text('{"reason":"test pause"}', encoding="utf-8")
            return fake_forward_batch(model, inputs)
        with patch("gef.fast_engine._runtime", fake_runtime), patch("gef.fast_engine.forward_batch", request_after_first_forward):
            result = fast.train(config, args())
        assert result["status"] == "paused"
        saved = _load_checkpoint(folder / "run/latest.pt")
        assert saved["state"]["global_step"] == 1
        assert saved["state"]["offset"] == 4
        assert not (folder / "run/stop_request.json").exists()
        assert (folder / "run/stop_request.acknowledged.json").exists()
        assert not (folder / "run/run_complete.json").exists()
        with patch("gef.fast_engine._runtime", fake_runtime), patch("gef.fast_engine.forward_batch", fake_forward_batch):
            assert fast.train(config, args(resume="auto"))["status"] == "complete"
    with patch("gef.fast_engine.train", return_value={"status": "paused"}):
        assert fast.main(["train"]) == 75
