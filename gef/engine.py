"""Reproducible single-GPU EvaHan training, prediction, profiling and scoring.

Run from code/: python -m gef.engine --help. No work starts upon import.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

# cuBLAS requires this before its first CUDA context is created. The launcher
# sets the same value before Python starts; this also covers standalone CLI use.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .evaluation import (evaluate_files, save_json, score_samples, sha256_file,
                         training_vocabulary, validate_samples, write_tagged)


DEFAULT_CONFIG = {
    "track": "paper15", "model_type": "gef", "augmentation": True,
    "encoder_path": "models/sikuroberta", "data_dir": "data/prepared", "output_dir": "runs/manual",
    "device": "cuda:0", "gpu_uuid": None,
    "seed": 13, "epochs": 10, "d": 128, "t": 64, "dropout": 0.1,
    "encoder_lr": 2e-5, "head_lr": 1e-3, "weight_decay": 0.01, "warmup_ratio": 0.1,
    "grad_clip": 1.0, "effective_batch_size": 16, "micro_batch_size": 1, "amp": True,
    "encoder_window": 510, "encoder_overlap": 128, "crf_chunk_size": 256,
    "feature_chunk_size": 4096, "gradient_checkpointing": True,
    "retain_completed_resume": False,
    "official_scorer": "data/official_snapshot/eval_EvaHan_2022_FINAL.py",
    "official_timeout": 120,
}


def load_config(path=None, overrides=None):
    result = dict(DEFAULT_CONFIG)
    if path:
        result.update(json.loads(Path(path).read_text(encoding="utf-8-sig")))
    result.update({key: value for key, value in (overrides or {}).items() if value is not None})
    if result["micro_batch_size"] != 1:
        raise ValueError("variable-size exact grids currently require micro_batch_size=1")
    for key in ("epochs", "effective_batch_size", "d", "t"):
        if result[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if result["track"] not in ("paper15", "native21"):
        raise ValueError("track must be paper15 or native21")
    if result["model_type"] not in ("gef", "grid_softmax", "sequence_crf"):
        raise ValueError("unknown model type")
    if not 0 <= result["warmup_ratio"] <= 1:
        raise ValueError("warmup_ratio must lie in [0, 1]")
    return result


def is_better(metrics, best):
    """Full-precision dev PF, then WF; exact ties retain the earliest epoch."""
    return best is None or (metrics["PF"], metrics["WF"]) > (best["PF"], best["WF"])


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # BERT uses SDPA in recent Transformers. Its fused CUDA backward can be
    # nondeterministic, so select the deterministic math implementation.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def capture_rng():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(torch.cuda.current_device()) if torch.cuda.is_initialized() else None}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_initialized():
        # Only the explicitly selected GPU is touched, including when a UUID
        # launcher remaps physical GPU 1 to logical cuda:0 during a resume.
        torch.cuda.set_rng_state(state["cuda"].cpu(), torch.cuda.current_device())


def optimizer_groups(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        encoder = name.startswith("encoder.")
        no_decay = name.endswith("bias") or "layernorm" in name.lower() or "layer_norm" in name.lower()
        key = (encoder, no_decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": config["encoder_lr"] if encoder else config["head_lr"],
                           "weight_decay": 0.0 if no_decay else config["weight_decay"]}
        groups[key]["params"].append(parameter)
    return list(groups.values())


def lr_factor(step, total_steps, warmup_steps):
    if warmup_steps and step < warmup_steps:
        return float(step) / warmup_steps
    return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))


def character_ids(tokenizer, text, device):
    import torch
    ids = tokenizer.convert_tokens_to_ids(list(text))
    if len(ids) != len(text) or any(value is None for value in ids):
        raise ValueError("tokenizer must map exactly one original character to one vocabulary id")
    return torch.tensor(ids, dtype=torch.long, device=device)


def _mapped_samples(path, track):
    from .data import load_jsonl, map_pos
    samples = load_jsonl(path)
    samples = list(samples)
    output = []
    for sample in samples:
        transformed = dict(sample)
        transformed["words"] = [dict(word, pos=map_pos(word["pos"], track)) for word in sample["words"]]
        output.append(transformed)
    validate_samples(output)
    return output


def _limit(samples, limit):
    return samples[:limit] if limit is not None else samples


def input_signatures(config):
    """Catch a reordered same-size vocabulary and implementation drift on resume."""
    encoder = Path(config["encoder_path"])
    metadata = {name: sha256_file(encoder / name)
                for name in ("config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json", "download_manifest.json")
                if (encoder / name).is_file()}
    package = Path(__file__).resolve().parent
    implementation = {name: sha256_file(package / name) for name in ("data.py", "model.py", "grid.py", "crf.py", "engine.py", "evaluation.py")
                      if (package / name).is_file()}
    return {"encoder_metadata_hashes": metadata, "implementation_hashes": implementation}


def stage_evidence(config):
    data_dir = Path(config["data_dir"])
    paths = [data_dir / name for name in ("train.jsonl", "train_augmented.jsonl", "dev.jsonl", "manifest.json",
             "test_a.jsonl", "test_b.jsonl", "test_a_raw_lines.json", "test_b_raw_lines.json")]
    scorer = Path(config["official_scorer"])
    paths += [scorer, scorer.parent / "EvaHan_testa_gold.txt", scorer.parent / "EvaHan_testb_gold.txt"]
    return {**input_signatures(config), "dataset_hashes": {str(path.resolve()): sha256_file(path) for path in paths if path.is_file()}}


def select_device(config):
    """Validate physical GPU identity before model allocation; never change masks."""
    import torch
    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            raise RuntimeError(f"requested {device}, available CUDA devices={torch.cuda.device_count()}; no CPU/GPU fallback")
        properties = torch.cuda.get_device_properties(device)
        expected, actual = config.get("gpu_uuid"), str(getattr(properties, "uuid", ""))
        normalize = lambda value: value.lower().removeprefix("gpu-")
        if expected and normalize(str(expected)) != normalize(actual):
            raise RuntimeError(f"requested physical GPU UUID {expected}, but {device} is {properties.name} ({actual}); "
                               "set CUDA_VISIBLE_DEVICES before launching Python, or choose the matching logical index")
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("only explicit cpu or cuda devices are supported")
    return device


def _runtime(config, *, checkpoint=None):
    from transformers import AutoTokenizer
    from .data import pos_tags
    from .model import GEFModel
    device = select_device(config)
    tokenizer = AutoTokenizer.from_pretrained(config["encoder_path"], local_files_only=True, use_fast=False)
    tags = list(pos_tags(config["track"]))
    keys = ("model_type", "d", "t", "dropout", "encoder_window", "encoder_overlap", "crf_chunk_size", "feature_chunk_size", "gradient_checkpointing")
    model = GEFModel(config["encoder_path"], tags, **{key: config[key] for key in keys}).to(device)
    if checkpoint:
        if list(checkpoint["pos_tags"]) != tags:
            raise ValueError("checkpoint POS inventory differs from the requested protocol")
        model.load_state_dict(checkpoint["model"])
    return model, tokenizer, device, tags


def _autocast(config, device):
    import torch
    return torch.autocast(device_type="cuda", dtype=torch.float16) if config["amp"] and device.type == "cuda" else contextlib.nullcontext()


def _decode(model, tokenizer, samples, device, config, include_loss=False):
    import torch
    model.eval()
    predictions, diagnostics = [], {"conflict_cells": 0, "grid_cells": 0, "conflict_sentences": 0, "sentences": len(samples)}
    losses = []
    with torch.no_grad():
        for sample in samples:
            ids = character_ids(tokenizer, sample["text"], device)
            with _autocast(config, device):
                emissions = model(ids)
            decoded = model.decode_emissions(emissions)
            if include_loss:
                losses.append(float(model.loss_from_emissions(emissions, sample["words"])))
            output = dict(sample, words=[dict(word, text=sample["text"][word["start"]:word["end"]]) for word in decoded["words"]],
                          decoding={key: int(decoded.get(key, 0)) for key in ("conflict_cells", "grid_cells", "conflict_sentence")})
            predictions.append(output)
            diagnostics["conflict_cells"] += int(decoded.get("conflict_cells", 0))
            diagnostics["grid_cells"] += int(decoded.get("grid_cells", 0))
            diagnostics["conflict_sentences"] += int(bool(decoded.get("conflict_sentence", False)))
    validate_samples(predictions, prediction=True)
    diagnostics["conflict_cell_rate"] = diagnostics["conflict_cells"] / max(1, diagnostics["grid_cells"])
    diagnostics["conflict_sentence_rate"] = diagnostics["conflict_sentences"] / max(1, diagnostics["sentences"])
    if include_loss:
        diagnostics["mean_normalized_loss"] = sum(losses) / len(losses) if losses else None
    return predictions, diagnostics


def _jsonl(path, entry):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def _write_jsonl(path, entries):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _save_checkpoint(path, model, optimizer, scheduler, scaler, config, tags, state, dataset_hashes, rng=None):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"format_version": 1, "checkpoint_type": "training_resume", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "config": config,
             "pos_tags": tags, "state": dict(state), "dataset_hashes": dataset_hashes,
             "rng": rng or capture_rng(), "model_config": getattr(model, "model_config", {}),
             "selection_rule": "dev independent exact-span PF, then WF, then earliest epoch"}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def inference_checkpoint(checkpoint):
    """Keep the selected weights and protocol; drop bulky training-only states."""
    keys = ("format_version", "model", "config", "pos_tags", "dataset_hashes", "model_config", "selection_rule")
    result = {key: checkpoint[key] for key in keys if key in checkpoint}
    result.update(checkpoint_type="inference", resumable=False,
                  best_epoch=checkpoint.get("state", {}).get("best_epoch", checkpoint.get("best_epoch")),
                  best_dev=checkpoint.get("state", {}).get("best_metrics", checkpoint.get("best_dev")))
    return result


def require_training_checkpoint(checkpoint, path):
    required = ("optimizer", "scheduler", "scaler", "rng", "state")
    if checkpoint.get("checkpoint_type") == "inference" or any(key not in checkpoint for key in required):
        raise ValueError(f"{path} is an inference checkpoint and cannot resume training; use a full latest.pt "
                         "from an unfinished run, or set retain_completed_resume=True before completing training")


def _atomic_torch_save(path, value):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _version_info():
    packages = {}
    for name in ("torch", "transformers", "numpy", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    import torch
    gpu = None
    if torch.cuda.is_initialized():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {"logical_index": index, "name": properties.name, "uuid": str(getattr(properties, "uuid", "")),
               "total_memory_bytes": properties.total_memory}
    return {"python": sys.version, "executable": sys.executable, "packages": packages,
            "cuda_runtime": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "selected_gpu": gpu, "cudnn": torch.backends.cudnn.version(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "sdpa": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                     "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                     "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
                     "math": torch.backends.cuda.math_sdp_enabled()}}


def _tests(model, tokenizer, device, config, output_dir, vocabulary, eval_limit=None):
    reports = {}
    data_dir, output_dir = Path(config["data_dir"]), Path(output_dir)
    for split in ("test_a", "test_b"):
        samples = _limit(_mapped_samples(data_dir / f"{split}.jsonl", config["track"]), eval_limit)
        predictions, diagnostics = _decode(model, tokenizer, samples, device, config)
        folder = output_dir / split
        raw_path = data_dir / f"{split}_raw_lines.json"
        raw_lines = json.loads(raw_path.read_text(encoding="utf-8")) if eval_limit is None else None
        prediction = write_tagged(folder / "prediction.txt", predictions, raw_lines)
        _write_jsonl(folder / "prediction.jsonl", predictions)
        if config["track"] == "native21" and eval_limit is None:
            letter = "a" if split == "test_a" else "b"
            gold = Path(config["official_scorer"]).parent / f"EvaHan_test{letter}_gold.txt"
        else:
            gold = write_tagged(folder / f"gold.{config['track']}.txt", samples, raw_lines)
        metadata = {"track": config["track"], "model_type": config["model_type"], "seed": config["seed"],
                    "subset_smoke_only": eval_limit is not None, "sample_limit": eval_limit,
                    "test_used_for_selection": False, "physical_blank_lines_preserved": eval_limit is None,
                    "diagnostics": diagnostics}
        reports[split] = evaluate_files(prediction, gold, folder, scorer_path=config["official_scorer"],
                                        train_vocabulary=vocabulary, timeout=config["official_timeout"], metadata=metadata)
    return reports


def train(config, args):
    import torch
    config = dict(config, **{key: str(Path(config[key]).resolve()) for key in ("encoder_path", "data_dir", "output_dir", "official_scorer")})
    seed_everything(config["seed"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    data_dir = Path(config["data_dir"])
    train_name = "train_augmented.jsonl" if config["augmentation"] else "train.jsonl"
    train_path, dev_path = data_dir / train_name, data_dir / "dev.jsonl"
    samples = _limit(_mapped_samples(train_path, config["track"]), args.train_limit)
    dev = _limit(_mapped_samples(dev_path, config["track"]), args.eval_limit)
    if not samples or not dev:
        raise ValueError("training and dev must both contain samples")
    original_train = _mapped_samples(data_dir / "train.jsonl", config["track"])
    vocabulary = training_vocabulary(original_train)
    evidence = stage_evidence(config)
    hashes = evidence["dataset_hashes"]
    config = dict(config, train_limit=args.train_limit, eval_limit=args.eval_limit,
                  encoder_metadata_hashes=evidence["encoder_metadata_hashes"], implementation_hashes=evidence["implementation_hashes"])
    resume_path = (output / "latest.pt") if args.resume == "auto" else (Path(args.resume) if args.resume else None)
    checkpoint = _load_checkpoint(resume_path) if resume_path and resume_path.exists() else None
    if args.resume and args.resume != "auto" and checkpoint is None:
        raise FileNotFoundError(resume_path)
    if checkpoint:
        require_training_checkpoint(checkpoint, resume_path)
        essential = ("track", "model_type", "augmentation", "encoder_path", "epochs", "seed", "d", "t", "dropout", "encoder_lr", "head_lr", "weight_decay", "warmup_ratio", "grad_clip", "effective_batch_size", "micro_batch_size", "train_limit", "eval_limit", "amp", "encoder_window", "encoder_overlap", "crf_chunk_size", "feature_chunk_size", "gradient_checkpointing", "encoder_metadata_hashes", "implementation_hashes")
        changed = [key for key in essential if checkpoint["config"].get(key) != config.get(key)]
        if changed or checkpoint["dataset_hashes"] != hashes:
            raise ValueError(f"resume configuration/data changed: {changed}; dataset hashes must match")
    elif (output / "latest.pt").exists() or (output / "epochs.jsonl").exists():
        raise FileExistsError(f"run exists at {output}; pass --resume auto or use a new output directory")
    model, tokenizer, device, tags = _runtime(config, checkpoint=checkpoint)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config))
    total_steps = math.ceil(len(samples) / config["effective_batch_size"]) * config["epochs"]
    warmup = int(total_steps * config["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, total_steps, warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] and device.type == "cuda")
    state = {"epoch": 1, "offset": 0, "order": None, "global_step": 0, "optimizer_steps": 0, "amp_skips": 0, "best_metrics": None,
             "best_epoch": None, "epoch_loss_sum": 0.0, "epoch_samples": 0}
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint["state"]
        restore_rng(checkpoint["rng"])
    del checkpoint
    save_json(output / "resolved_config.json", {**config, "total_optimizer_steps": total_steps, "warmup_steps": warmup,
                                                "pos_tags": tags, "dataset_hashes": hashes,
                                                "provenance_sha256": sha256_file("reports/provenance.json") if Path("reports/provenance.json").is_file() else None,
                                                "environment": _version_info()})
    started = time.monotonic()
    boundary_rng = capture_rng()
    stopped_early = False
    try:
        while state["epoch"] <= config["epochs"]:
            epoch = state["epoch"]
            if state["order"] is None:
                state["order"] = list(range(len(samples)))
                random.shuffle(state["order"])
                state.update(offset=0, epoch_loss_sum=0.0, epoch_samples=0)
                boundary_rng = capture_rng()
            model.train()
            while state["offset"] < len(samples):
                if args.max_steps is not None and state["global_step"] >= args.max_steps:
                    stopped_early = True
                    break
                indices = state["order"][state["offset"]:state["offset"] + config["effective_batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for index in indices:
                    sample = samples[index]
                    ids = character_ids(tokenizer, sample["text"], device)
                    with _autocast(config, device):
                        emissions = model(ids)
                    # Exact CRF numerics are FP32 (with internal long-chain FP64 accumulation).
                    loss = model.loss_from_emissions(emissions, sample["words"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"nonfinite loss at sample {sample['id']}")
                    loss_sum += float(loss.detach())
                    scaler.scale(loss / len(indices)).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=not scaler.is_enabled())
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                    state["optimizer_steps"] += 1
                else:
                    state["amp_skips"] += 1
                state["global_step"] += 1
                state["offset"] += len(indices)
                state["epoch_loss_sum"] += loss_sum
                state["epoch_samples"] += len(indices)
                boundary_rng = capture_rng()
                if state["global_step"] == 1 or state["global_step"] % args.log_every == 0:
                    progress = {"event": "train_step", "epoch": epoch, "step": state["global_step"],
                                "offset": state["offset"], "samples": len(samples), "loss": loss_sum / len(indices),
                                "grad_norm": float(grad_norm) if torch.isfinite(grad_norm) else None,
                                "optimizer_steps": state["optimizer_steps"], "amp_skips": state["amp_skips"],
                                "elapsed_seconds": time.monotonic() - started}
                    _jsonl(output / "steps.jsonl", progress)
                    print(json.dumps(progress), flush=True)
            if stopped_early:
                _save_checkpoint(output / "latest.pt", model, optimizer, scheduler, scaler, config, tags, state, hashes, boundary_rng)
                save_json(output / "smoke_stop.json", {"status": "max_steps", "state": state, "complete_training": False})
                return {"status": "max_steps", "global_step": state["global_step"], "output_dir": str(output)}
            predictions, diagnostics = _decode(model, tokenizer, dev, device, config)
            metrics = score_samples(dev, predictions, vocabulary)
            better = is_better(metrics, state["best_metrics"])
            if better:
                state["best_metrics"] = {name: metrics[name] for name in ("PF", "WF")}
                state["best_epoch"] = epoch
            record = {"epoch": epoch, "global_step": state["global_step"], "seed": config["seed"],
                      "train_normalized_loss": state["epoch_loss_sum"] / state["epoch_samples"], "dev": metrics,
                      "dev_diagnostics": diagnostics, "is_best": better, "best_epoch": state["best_epoch"],
                      "learning_rates": scheduler.get_last_lr(), "optimizer_steps": state["optimizer_steps"],
                      "amp_skips": state["amp_skips"], "elapsed_seconds": time.monotonic() - started,
                      "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
            _jsonl(output / "epochs.jsonl", record)
            print(json.dumps({"event": "epoch", "epoch": epoch, "dev_PF": metrics["PF"], "dev_WF": metrics["WF"], "is_best": better}), flush=True)
            state.update(epoch=epoch + 1, offset=0, order=None, epoch_loss_sum=0.0, epoch_samples=0)
            boundary_rng = capture_rng()
            if better:
                _save_checkpoint(output / "best.pt", model, optimizer, scheduler, scaler, config, tags, state, hashes, boundary_rng)
                save_json(output / "best_dev.json", {"epoch": epoch, "metrics": metrics, "diagnostics": diagnostics})
            _save_checkpoint(output / "latest.pt", model, optimizer, scheduler, scaler, config, tags, state, hashes, boundary_rng)
    except KeyboardInterrupt:
        _save_checkpoint(output / "latest.pt", model, optimizer, scheduler, scaler, config, tags, state, hashes, boundary_rng)
        raise
    best = _load_checkpoint(output / "best.pt")
    model.load_state_dict(best["model"])
    compact_best = inference_checkpoint(best)
    del best
    reports = {} if args.skip_test else _tests(model, tokenizer, device, config, output / "evaluation", vocabulary, args.eval_limit)
    complete_tests = (set(reports) == {"test_a", "test_b"}
                      and all(value.get("status") != "invalid_format" and "independent" in value for value in reports.values()))
    archive = (not config["retain_completed_resume"] and args.train_limit is None
               and args.eval_limit is None and not args.skip_test and complete_tests)
    if archive:
        # Test the reloaded best model first. An interrupted conversion still
        # leaves the separately saved full latest checkpoint available.
        _atomic_torch_save(output / "best.pt", compact_best)
    del compact_best
    storage = {"policy": "inference_only_after_full_training_and_test" if archive else "retain_full_resume",
               "best_checkpoint_type": "inference" if archive else "training_resume",
               "retain_completed_resume_requested": config["retain_completed_resume"],
               "latest_checkpoint_path": str((output / "latest.pt").resolve()),
               "latest_checkpoint_removed": False}
    result = {"status": "complete", "epochs": config["epochs"], "global_step": state["global_step"],
              "optimizer_steps": state["optimizer_steps"], "amp_skips": state["amp_skips"],
              "best_epoch": state["best_epoch"], "best_dev": state["best_metrics"], "tests": reports,
              "test_skipped": args.skip_test, "subset_smoke_only": args.train_limit is not None or args.eval_limit is not None,
              "best_checkpoint_reloaded_for_test": True, "output_dir": str(output.resolve()),
              "resolved_config_sha256": sha256_file(output / "resolved_config.json"),
              "checkpoint_storage": storage}
    save_json(output / "run_complete.json", result)
    if archive:
        # Only this run's generated literal latest.pt is removed, and only
        # after complete evaluation evidence and run_complete are durable.
        try:
            (output / "latest.pt").unlink()
            storage["latest_checkpoint_removed"] = True
        except OSError as error:
            storage["cleanup_error"] = str(error)
        save_json(output / "run_complete.json", result)
    save_json(output / "checkpoint_storage.json", storage)
    return result


def profile(config, args):
    import torch
    seed_everything(config["seed"])
    data_dir = Path(config["data_dir"])
    samples = _mapped_samples(data_dir / "train_augmented.jsonl", config["track"])
    longest = max(samples, key=lambda value: len(value["text"]))
    lengths = [int(value) for value in args.lengths.split(",")]
    if not args.skip_maximum:
        lengths.append(len(longest["text"]))
    model, tokenizer, device, tags = _runtime(config)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config))
    evidence = stage_evidence(config)
    records = []
    for length in dict.fromkeys(lengths):
        record = {"characters": length, "is_actual_augmented_maximum": length == len(longest["text"]),
                  "actual_augmented_maximum": len(longest["text"]), "synthetic_target_for_memory_only": True,
                  "status": "running", "phase": "forward", "started_unix": time.time()}
        print(json.dumps({"event": "profile_start", "characters": length}), flush=True)
        save_json(Path(config["output_dir"]) / "profile.json", {"status": "running", "gate_passed": False,
                                                                 "config": config, "records": records + [record]})
        text = (longest["text"] * math.ceil(length / len(longest["text"])))[:length]
        words = [{"start": i, "end": i + 1, "pos": tags[0]} for i in range(length)]
        model.train()
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.monotonic()
        ids = emissions = loss = None
        try:
            ids = character_ids(tokenizer, text, device)
            phase_started = time.monotonic()
            with _autocast(config, device):
                emissions = model(ids)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record["forward_seconds"] = time.monotonic() - phase_started
            record["phase"] = "loss"
            save_json(Path(config["output_dir"]) / "profile.json", {"status": "running", "gate_passed": False,
                                                                     "config": config, "records": records + [record]})
            phase_started = time.monotonic()
            loss = model.loss_from_emissions(emissions, words)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite profile loss")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record["loss_seconds"] = time.monotonic() - phase_started
            record["phase"] = "backward"
            save_json(Path(config["output_dir"]) / "profile.json", {"status": "running", "gate_passed": False,
                                                                     "config": config, "records": records + [record]})
            phase_started = time.monotonic()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record["backward_seconds"] = time.monotonic() - phase_started
            phase_started = time.monotonic()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            optimizer.step()  # Include AdamW moments in the training memory gate.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record["optimizer_seconds"] = time.monotonic() - phase_started
            record["includes_adamw_states"] = True
            record["phase"] = "decode"
            save_json(Path(config["output_dir"]) / "profile.json", {"status": "running", "gate_passed": False,
                                                                     "config": config, "records": records + [record]})
            phase_started = time.monotonic()
            decoded = model.decode_emissions(emissions.detach())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record["decode_seconds"] = time.monotonic() - phase_started
            validate_samples([{"id": "profile", "text": text, "words": decoded["words"]}])
            record["decode_covers_all_characters"] = True
            record.update(status="ok", phase="complete", normalized_loss=float(loss.detach()), cells=int(length * (length + 1) // 2) if config["model_type"] != "sequence_crf" else length)
        except (torch.OutOfMemoryError, RuntimeError, ValueError, FloatingPointError) as error:
            record.update(status="oom" if "memory" in str(error).lower() else "error", error=str(error))
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            if device.type == "cuda":
                record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            model.zero_grad(set_to_none=True)
            del ids, emissions, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
            records.append(record)
            print(json.dumps(record), flush=True)
            save_json(Path(config["output_dir"]) / "profile.json", {"status": "running", "config": config, "records": records, "environment": _version_info()})
    successful = all(record["status"] == "ok" for record in records)
    required_covered = {64, 128, 256, len(longest["text"])}.issubset(set(lengths))
    result = {"status": "profiled" if successful else "resource_gate_failed", "config": config, "records": records,
              "tested_lengths_passed": successful, "all_required_lengths_tested": required_covered,
              "gate_passed": successful and required_covered, "evidence": evidence, "environment": _version_info()}
    save_json(Path(config["output_dir"]) / "profile.json", result)
    return result


def select_overfit_samples(samples, count=32, max_length=64):
    """Deterministically cover POS, multi-character words and same-POS boundaries."""
    candidates = sorted((sample for sample in samples if 4 <= len(sample["text"]) <= max_length
                         and len(sample["words"]) >= 2 and any(word["pos"] != "w" for word in sample["words"])),
                        key=lambda sample: str(sample["id"]))
    def features(sample):
        words = sample["words"]
        tags = {"pos:" + word["pos"] for word in words if word["pos"] != "w"}
        if any(word["end"] - word["start"] > 1 and word["pos"] != "w" for word in words):
            tags.add("multi_character_word")
        if any(left["pos"] == right["pos"] != "w" for left, right in zip(words, words[1:])):
            tags.add("adjacent_same_pos_words")
        return tags
    selected, covered = [], set()
    while candidates and len(selected) < count:
        def priority(index):
            sample = candidates[index]
            found = features(sample)
            return (len(found - covered), int("multi_character_word" in found),
                    int("adjacent_same_pos_words" in found), len(found), -abs(len(sample["text"]) - 24), -index)
        index = max(range(len(candidates)), key=priority)
        sample = candidates.pop(index)
        selected.append(sample)
        covered.update(features(sample))
    return selected


def overfit(config, args):
    import torch
    seed_everything(config["seed"])
    samples = _mapped_samples(Path(config["data_dir"]) / "train.jsonl", config["track"])
    samples = select_overfit_samples(samples, args.samples, args.max_length)
    if not samples:
        raise ValueError("no short samples satisfy the overfit filter")
    model, tokenizer, device, tags = _runtime(config)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, args.steps, int(args.steps * config["warmup_ratio"])))
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] and device.type == "cuda")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    predictions, diagnostics = _decode(model, tokenizer, samples, device, config, include_loss=True)
    baseline = {"step": 0, "loss": diagnostics["mean_normalized_loss"], "metrics": score_samples(samples, predictions), "diagnostics": diagnostics}
    records = [baseline]
    _jsonl(output / "overfit.jsonl", baseline)
    for step in range(1, args.steps + 1):
        model.train()
        chosen = [samples[(step - 1) * config["effective_batch_size"] % len(samples)]] if len(samples) == 1 else [samples[((step - 1) * config["effective_batch_size"] + i) % len(samples)] for i in range(min(config["effective_batch_size"], len(samples)))]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for sample in chosen:
            ids = character_ids(tokenizer, sample["text"], device)
            with _autocast(config, device):
                emissions = model(ids)
            loss = model.loss_from_emissions(emissions, sample["words"])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite overfit loss")
            losses.append(float(loss.detach()))
            scaler.scale(loss / len(chosen)).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=not scaler.is_enabled())
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= before:
            scheduler.step()
        if step == 1 or step % args.check_every == 0 or step == args.steps:
            predictions, diagnostics = _decode(model, tokenizer, samples, device, config, include_loss=True)
            record = {"step": step, "loss": diagnostics["mean_normalized_loss"], "train_batch_loss": sum(losses) / len(losses), "metrics": score_samples(samples, predictions), "diagnostics": diagnostics}
            records.append(record)
            _jsonl(output / "overfit.jsonl", record)
            print(json.dumps({"step": step, "loss": record["loss"], "PF": record["metrics"]["PF"], "WF": record["metrics"]["WF"]}), flush=True)
    last = records[-1]
    gate_passed = (last["metrics"]["WF"] >= 95 and last["metrics"]["PF"] >= 95
                   and last["loss"] <= baseline["loss"] * .8)
    result = {"status": "complete" if gate_passed else "overfit_gate_failed", "gate_passed": gate_passed,
              "gate_criteria": {"minimum_final_WF": 95, "minimum_final_PF": 95, "maximum_final_to_initial_loss_ratio": .8},
              "config": config, "samples": [sample["id"] for sample in samples], "steps": args.steps,
              "sample_selection": "4..max_length characters; >=2 words; >=1 nonpunctuation word; deterministic POS/multi-character/same-POS coverage",
              "records": records, "evidence": stage_evidence(config),
              "purpose": "implementation diagnostic; no requirement to reproduce historical headline scores"}
    save_json(output / "overfit_summary.json", result)
    return result


def predict(config, args):
    checkpoint = _load_checkpoint(args.checkpoint)
    saved_config = checkpoint["config"]
    # Architecture and label inventory always come from the saved checkpoint.
    config = dict(saved_config, device=config["device"], output_dir=config["output_dir"],
                  encoder_path=args.encoder_path or saved_config["encoder_path"], data_dir=args.data_dir or saved_config["data_dir"])
    signatures = input_signatures(config)
    if any(saved_config.get(key) != value for key, value in signatures.items()):
        raise ValueError("checkpoint tokenizer/implementation signatures differ; prediction would change the recorded protocol")
    model, tokenizer, device, _ = _runtime(config, checkpoint=checkpoint)
    path = Path(args.input) if args.input else Path(config["data_dir"]) / f"{args.split}.jsonl"
    samples = _limit(_mapped_samples(path, config["track"]), args.eval_limit)
    predictions, diagnostics = _decode(model, tokenizer, samples, device, config)
    raw_path = Path(args.raw_lines) if args.raw_lines else path.with_name(path.stem + "_raw_lines.json")
    raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() and args.eval_limit is None else None
    destination = Path(args.prediction) if args.prediction else Path(config["output_dir"]) / "prediction.txt"
    write_tagged(destination, predictions, raw)
    _write_jsonl(destination.with_suffix(".jsonl"), predictions)
    result = {"status": "predicted", "prediction": str(destination.resolve()), "diagnostics": diagnostics,
              "subset_smoke_only": args.eval_limit is not None, "physical_blank_lines_preserved": raw is not None}
    save_json(destination.with_suffix(".diagnostics.json"), result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("profile", "overfit", "train", "predict", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--config")
        for option in ("encoder-path", "data-dir", "output-dir", "device", "gpu-uuid", "official-scorer"):
            sub.add_argument("--" + option)
        sub.add_argument("--track", choices=("paper15", "native21"))
        sub.add_argument("--model-type", choices=("gef", "grid_softmax", "sequence_crf"))
        sub.add_argument("--seed", type=int)
        sub.add_argument("--epochs", type=int)
        sub.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
        sub.add_argument("--augmentation", action=argparse.BooleanOptionalAction, default=None)
        sub.add_argument("--eval-limit", type=int, help="explicit subset smoke only; never a final full-test report")
        if name == "profile":
            sub.add_argument("--lengths", default="64,128,256")
            sub.add_argument("--skip-maximum", action="store_true")
        elif name == "overfit":
            sub.add_argument("--steps", type=int, default=100)
            sub.add_argument("--samples", type=int, default=32)
            sub.add_argument("--max-length", type=int, default=64)
            sub.add_argument("--check-every", type=int, default=10)
        elif name == "train":
            sub.add_argument("--retain-completed-resume", action=argparse.BooleanOptionalAction, default=None)
            sub.add_argument("--resume", nargs="?", const="auto")
            sub.add_argument("--train-limit", "--max-samples", dest="train_limit", type=int)
            sub.add_argument("--max-steps", type=int)
            sub.add_argument("--skip-test", action="store_true")
            sub.add_argument("--log-every", type=int, default=20)
        elif name == "predict":
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--input")
            sub.add_argument("--split", choices=("test_a", "test_b", "dev"), default="test_a")
            sub.add_argument("--raw-lines")
            sub.add_argument("--prediction")
        elif name == "evaluate":
            sub.add_argument("--prediction", required=True)
            sub.add_argument("--gold", required=True, help="already mapped gold for the stated track")
            sub.add_argument("--train-jsonl")
            sub.add_argument("--no-official", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    keys = set(DEFAULT_CONFIG)
    config = load_config(args.config, {key: value for key, value in vars(args).items() if key in keys})
    for key in ("train_limit", "eval_limit", "max_steps", "steps", "samples", "max_length", "check_every", "log_every"):
        value = getattr(args, key, None)
        if value is not None and value <= 0:
            raise ValueError(f"{key} must be positive")
    if args.command == "evaluate":
        vocabulary = training_vocabulary(_mapped_samples(args.train_jsonl, config["track"])) if args.train_jsonl else None
        report = evaluate_files(args.prediction, args.gold, config["output_dir"],
                                scorer_path=None if args.no_official else config["official_scorer"],
                                train_vocabulary=vocabulary, timeout=config["official_timeout"], metadata={"track": config["track"]})
    else:
        report = globals()[args.command](config, args)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] in ("ok", "complete", "predicted", "profiled", "max_steps") else 2


if __name__ == "__main__":
    raise SystemExit(main())
