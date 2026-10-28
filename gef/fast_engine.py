"""Batched encoder execution with the frozen EvaHan statistical protocol.

This is an opt-in execution engine for not-yet-started runs. It never migrates
legacy engine checkpoints. Effective batches and original sample order stay
unchanged; each sample-normalized loss contributes 1/effective_batch_count.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

from . import engine as legacy
from .engine import (_atomic_torch_save, _autocast, _decode, _jsonl, _limit,
                     _load_checkpoint, _mapped_samples, _runtime, _save_checkpoint,
                     _tests, _version_info, capture_rng, character_ids,
                     inference_checkpoint, is_better, lr_factor, optimizer_groups,
                     require_training_checkpoint, restore_rng, seed_everything)
from .evaluation import save_json, score_samples, sha256_file, training_vocabulary


COMPUTE_PROTOCOL = "gef-padded-encoder-microbatch-v1"
DEFAULT_CONFIG = dict(legacy.DEFAULT_CONFIG, micro_batch_size=4,
                      max_batch_chars=512, max_batch_cells=131072,
                      compute_protocol=COMPUTE_PROTOCOL)


def load_config(path=None, overrides=None):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {}
    raw.update({key: value for key, value in (overrides or {}).items() if value is not None})
    # Old config files describe the old physical micro-batch of one. An explicit
    # fast override wins; otherwise this engine defaults to four.
    micro = (overrides or {}).get("micro_batch_size")
    if micro is None:
        micro = raw.get("micro_batch_size", 4) if raw.get("compute_protocol") else 4
    config = legacy.load_config(overrides=dict(raw, micro_batch_size=1))
    config.update(micro_batch_size=micro,
                  max_batch_chars=raw.get("max_batch_chars", DEFAULT_CONFIG["max_batch_chars"]),
                  max_batch_cells=raw.get("max_batch_cells", DEFAULT_CONFIG["max_batch_cells"]),
                  compute_protocol=COMPUTE_PROTOCOL)
    for key in ("micro_batch_size", "max_batch_chars", "max_batch_cells"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if raw.get("compute_protocol", COMPUTE_PROTOCOL) != COMPUTE_PROTOCOL:
        raise ValueError("unsupported fast compute protocol")
    return config


def input_signatures(config):
    result = legacy.input_signatures(config)
    package = Path(__file__).resolve().parent
    result["fast_implementation_hashes"] = {name: sha256_file(package / name)
                                              for name in ("fast_engine.py", "batching.py")}
    return result


def stage_evidence(config):
    return dict(legacy.stage_evidence(config), **input_signatures(config))


def forward_batch(model, inputs):
    from .batching import forward_batch as implementation
    return implementation(model, inputs)


def microbatches(samples, indices, config):
    """Preserve order; over-budget single samples always run without truncation."""
    batch, characters, cells = [], 0, 0
    for index in indices:
        length = len(samples[index]["text"])
        if length > config["encoder_window"]:
            if batch:
                yield batch
                batch, characters, cells = [], 0, 0
            yield [index]
            continue
        cost = length if config["model_type"] == "sequence_crf" else length * (length + 1) // 2
        if batch and (len(batch) >= config["micro_batch_size"]
                      or characters + length > config["max_batch_chars"]
                      or cells + cost > config["max_batch_cells"]):
            yield batch
            batch, characters, cells = [], 0, 0
        batch.append(index)
        characters += length
        cells += cost
    if batch:
        yield batch


def train_effective_batch(model, tokenizer, samples, indices, device, config, scaler):
    """Accumulate the same per-sample objective; optimizer stepping is separate."""
    import torch
    if not indices:
        raise ValueError("an effective batch must contain at least one sample")
    loss_sum, sizes = 0.0, []
    for batch in microbatches(samples, indices, config):
        inputs = [character_ids(tokenizer, samples[index]["text"], device) for index in batch]
        with _autocast(config, device):
            outputs = forward_batch(model, inputs)
        if len(outputs) != len(batch):
            raise ValueError("forward_batch must return one emission tensor per sample in input order")
        losses = [model.loss_from_emissions(emission, samples[index]["words"])
                  for index, emission in zip(batch, outputs)]
        if any(not torch.isfinite(loss) for loss in losses):
            raise FloatingPointError(f"nonfinite micro-batch loss: {[samples[index]['id'] for index in batch]}")
        loss_sum += sum(float(loss.detach()) for loss in losses)
        scaler.scale(torch.stack(losses).sum() / len(indices)).backward()
        sizes.append(len(batch))
    return loss_sum, sizes


def _stop_request(output):
    path = Path(output) / "stop_request.json"
    return path if path.is_file() else None


def pause_at_boundary(output, model, optimizer, scheduler, scaler, config, tags, state, hashes, rng):
    request = _stop_request(output)
    if request is None:
        return None
    _save_checkpoint(output / "latest.pt", model, optimizer, scheduler, scaler, config, tags, state, hashes, rng)
    result = {"status": "paused", "reason": "stop_request", "global_step": state["global_step"],
              "offset": state["offset"], "epoch": state["epoch"], "output_dir": str(output)}
    save_json(output / "pause_acknowledged.json", result)
    os.replace(request, output / "stop_request.acknowledged.json")
    return result


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
                  encoder_metadata_hashes=evidence["encoder_metadata_hashes"], implementation_hashes=evidence["implementation_hashes"],
                  fast_implementation_hashes=evidence["fast_implementation_hashes"],
                  execution_engine="gef.fast_engine", compute_protocol=COMPUTE_PROTOCOL)
    resume_path = (output / "latest.pt") if args.resume == "auto" else (Path(args.resume) if args.resume else None)
    checkpoint = _load_checkpoint(resume_path) if resume_path and resume_path.exists() else None
    if args.resume and args.resume != "auto" and checkpoint is None:
        raise FileNotFoundError(resume_path)
    if checkpoint:
        require_training_checkpoint(checkpoint, resume_path)
        if checkpoint["config"].get("compute_protocol") != COMPUTE_PROTOCOL:
            raise ValueError("legacy/other-engine checkpoints cannot be migrated into fast_engine; resume with their original engine")
        essential = ("track", "model_type", "augmentation", "encoder_path", "epochs", "seed", "d", "t", "dropout", "encoder_lr", "head_lr", "weight_decay", "warmup_ratio", "grad_clip", "effective_batch_size", "micro_batch_size", "train_limit", "eval_limit", "amp", "encoder_window", "encoder_overlap", "crf_chunk_size", "feature_chunk_size", "gradient_checkpointing", "encoder_metadata_hashes", "implementation_hashes", "max_batch_chars", "max_batch_cells", "compute_protocol", "fast_implementation_hashes")
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
                paused = pause_at_boundary(output, model, optimizer, scheduler, scaler, config, tags, state, hashes, boundary_rng)
                if paused:
                    return paused
                indices = state["order"][state["offset"]:state["offset"] + config["effective_batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                loss_sum, batch_sizes = train_effective_batch(model, tokenizer, samples, indices, device, config, scaler)
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
                                "micro_batch_sizes": batch_sizes, "compute_protocol": COMPUTE_PROTOCOL,
                                "elapsed_seconds": time.monotonic() - started}
                    _jsonl(output / "steps.jsonl", progress)
                    print(json.dumps(progress), flush=True)
                paused = pause_at_boundary(output, model, optimizer, scheduler, scaler, config, tags, state, hashes, boundary_rng)
                if paused:
                    return paused
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
              "execution_engine": "gef.fast_engine", "compute_protocol": COMPUTE_PROTOCOL,
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


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True).add_parser("train")
    sub.add_argument("--config")
    for option in ("encoder-path", "data-dir", "output-dir", "device", "gpu-uuid", "official-scorer"):
        sub.add_argument("--" + option)
    sub.add_argument("--track", choices=("paper15", "native21"))
    sub.add_argument("--model-type", choices=("gef", "grid_softmax", "sequence_crf"))
    for option in ("seed", "epochs", "micro-batch-size", "max-batch-chars", "max-batch-cells"):
        sub.add_argument("--" + option, type=int)
    for option in ("amp", "augmentation", "retain-completed-resume"):
        sub.add_argument("--" + option, action=argparse.BooleanOptionalAction, default=None)
    sub.add_argument("--resume", nargs="?", const="auto")
    sub.add_argument("--train-limit", "--max-samples", dest="train_limit", type=int)
    sub.add_argument("--eval-limit", type=int)
    sub.add_argument("--max-steps", type=int)
    sub.add_argument("--skip-test", action="store_true")
    sub.add_argument("--log-every", type=int, default=20)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.config, {key: value for key, value in vars(args).items() if key in DEFAULT_CONFIG})
    for key in ("train_limit", "eval_limit", "max_steps", "log_every"):
        value = getattr(args, key, None)
        if value is not None and value < 1:
            raise ValueError(f"{key} must be positive")
    report = train(config, args)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report["status"] == "paused":
        return 75
    return 0 if report["status"] in ("complete", "max_steps") else 2


if __name__ == "__main__":
    raise SystemExit(main())
