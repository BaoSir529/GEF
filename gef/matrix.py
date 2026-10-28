"""Sequential, resumable launcher for the seven configurations x five seeds.

This module only starts training when invoked explicitly. Acceptance executes
paper15 GEF+augmentation at seed 42 first; remaining requires that run to have
finished full training and full Test-A/B evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import statistics
import sys
import time

from .evaluation import save_json, sha256_file
from .engine import load_config, stage_evidence, input_signatures


CONFIG_NAMES = (
    "paper15_gef_aug", "native21_gef_aug", "paper15_grid_softmax_aug",
    "native21_grid_softmax_aug", "paper15_sequence_crf_aug", "native21_sequence_crf_aug",
    "paper15_gef_noaug",
)
SEEDS = (13, 21, 42, 87, 100)


def aggregate_completed(completed):
    summary = {}
    for config in CONFIG_NAMES:
        records = [record for identity, record in completed.items() if identity.split("/")[0] == config]
        grouped = {"completed_seeds": len(records), "expected_seeds": len(SEEDS), "complete": len(records) == len(SEEDS)}
        for split in ("test_a", "test_b"):
            grouped[split] = {}
            for metric in ("WP", "WR", "WF", "PP", "PR", "PF"):
                values = [record["metrics"][split][metric] for record in records]
                grouped[split][metric] = {"n": len(values), "mean": statistics.mean(values) if values else None,
                                          "sample_std_ddof1": statistics.stdev(values) if len(values) > 1 else None,
                                          "min": min(values) if values else None, "max": max(values) if values else None}
        summary[config] = grouped
    return summary


def tasks(config_dir="configs", output_root="runs/matrix"):
    schedule = [{"id": f"{name}/seed_{seed}", "config": str(Path(config_dir) / f"{name}.json"),
             "seed": seed, "output_dir": str(Path(output_root) / name / f"seed_{seed}")}
            for seed in SEEDS for name in CONFIG_NAMES]
    acceptance = next(task for task in schedule if task["id"] == "paper15_gef_aug/seed_42")
    schedule.remove(acceptance)
    return [acceptance] + schedule


def completed_run(task, expected_config=None):
    path = Path(task["output_dir"]) / "run_complete.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("subset_smoke_only") or result.get("test_skipped"):
        return None
    if result.get("epochs") != 10 or not result.get("best_checkpoint_reloaded_for_test"):
        return None
    resolved_path = Path(task["output_dir"]) / "resolved_config.json"
    if not resolved_path.exists() or result.get("resolved_config_sha256") != sha256_file(resolved_path):
        return None
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if resolved.get("seed") != task["seed"]:
        return None
    requested = expected_config or load_config(task["config"])
    protocol_keys = ("track", "model_type", "augmentation", "epochs", "d", "t", "dropout", "encoder_lr", "head_lr",
                     "weight_decay", "warmup_ratio", "grad_clip", "effective_batch_size", "amp",
                     "encoder_window", "encoder_overlap")
    if any(resolved.get(key) != requested[key] for key in protocol_keys):
        return None
    if any(Path(resolved.get(key, "")).resolve() != Path(requested[key]).resolve()
           for key in ("encoder_path", "data_dir", "official_scorer")):
        return None
    if any(resolved.get(key) != value for key, value in input_signatures(resolved).items()):
        return None
    # Padding/microbatch/checkpoint chunk sizes change computation, not the
    # statistical protocol above. New fast-engine runs must additionally prove
    # the exact added implementation, while original-engine evidence stays valid.
    fast_protocol = resolved.get("compute_protocol")
    fast_hashes = resolved.get("fast_implementation_hashes")
    if fast_protocol is not None or fast_hashes is not None:
        if fast_protocol != "gef-padded-encoder-microbatch-v1":
            return None
        package = Path(__file__).resolve().parent
        names = ("fast_engine.py", "batching.py")
        if not all((package / name).is_file() for name in names):
            return None
        if fast_hashes != {name: sha256_file(package / name) for name in names}:
            return None
    if not (Path(task["output_dir"]) / "best.pt").is_file():
        return None
    if any(not Path(path).is_file() or sha256_file(path) != digest for path, digest in resolved.get("dataset_hashes", {}).items()):
        return None
    tests = result.get("tests", {})
    if set(tests) != {"test_a", "test_b"} or any(test.get("status") == "invalid_format" or "independent" not in test
                                               or test.get("metadata", {}).get("subset_smoke_only") for test in tests.values()):
        return None
    for split, test in tests.items():
        if test.get("status") not in ("ok", "official_error", "scorer_disagreement"):
            return None
        if test.get("official", {}).get("scorer_sha256") != sha256_file(resolved["official_scorer"]):
            return None
        metadata = test.get("metadata", {})
        if any(metadata.get(key) != resolved[key] for key in ("track", "model_type", "seed")):
            return None
        if metadata.get("test_used_for_selection") is not False or metadata.get("physical_blank_lines_preserved") is not True:
            return None
        # A result remains complete only while its actual evaluated files and
        # saved metrics agree with the run summary. This catches accidental
        # output edits, stale copied reports, and cross-track result mixing.
        for key in ("prediction", "gold"):
            evaluated = Path(test.get(key, ""))
            if not evaluated.is_file() or sha256_file(evaluated) != test.get(key + "_sha256"):
                return None
        metric_path = Path(task["output_dir"]) / "evaluation" / split / "metrics.json"
        if not metric_path.is_file() or json.loads(metric_path.read_text(encoding="utf-8")) != test:
            return None
        if resolved["track"] == "native21":
            letter = "a" if split == "test_a" else "b"
            original_gold = Path(resolved["official_scorer"]).parent / f"EvaHan_test{letter}_gold.txt"
            if Path(test["gold"]).resolve() != original_gold.resolve():
                return None
    # An original official-scorer failure is a completed evaluation with a flag,
    # not grounds to silently rewrite that scorer or to rerun expensive training.
    return {"id": task["id"], "output_dir": str(Path(task["output_dir"]).resolve()),
            "compute_protocol": fast_protocol or "original-single-sample-v1",
            "micro_batch_size": resolved.get("micro_batch_size", 1),
            "effective_batch_size": resolved["effective_batch_size"],
            "gradient_checkpointing": resolved.get("gradient_checkpointing"),
            "best_epoch": result["best_epoch"], "best_dev": result["best_dev"],
            "evaluation_status": {key: value["status"] for key, value in tests.items()},
            "metrics": {key: {name: value["independent"][name] for name in ("WP", "WR", "WF", "PP", "PR", "PF")}
                        for key, value in tests.items()}}


def validate_preflight(profile_path, overfit_path, config):
    """Both passed stages must refer to the current data, tokenizer and code."""
    reports = []
    for name, path in (("profile", profile_path), ("overfit", overfit_path)):
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(f"missing {name} gate report: {path}; run the preflight first")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("gate_passed"):
            raise RuntimeError(f"{name} gate has not passed: {path}")
        reports.append(report)
    profile, overfit = reports
    if not profile.get("all_required_lengths_tested") or not profile.get("tested_lengths_passed"):
        raise RuntimeError("profile must cover 64,128,256 and actual augmented maximum with valid decoding")
    if overfit.get("config", {}).get("seed") != 42 or len(overfit.get("samples", [])) < 32 or overfit.get("steps", 0) < 100:
        raise RuntimeError("overfit gate requires seed 42, at least 32 representative samples and 100 steps")
    protocol_keys = ("track", "model_type", "augmentation", "d", "t", "dropout", "encoder_lr", "head_lr", "weight_decay",
                     "warmup_ratio", "grad_clip", "effective_batch_size", "micro_batch_size", "amp", "encoder_window",
                     "encoder_overlap", "crf_chunk_size", "feature_chunk_size", "gradient_checkpointing")
    current_evidence = stage_evidence(config)
    for name, report in zip(("profile", "overfit"), reports):
        if any(report.get("config", {}).get(key) != config[key] for key in protocol_keys):
            raise RuntimeError(f"{name} configuration differs from acceptance protocol")
        if report.get("evidence") != current_evidence:
            raise RuntimeError(f"{name} data/tokenizer/implementation hashes changed; rerun preflight")
    return {"profile": str(Path(profile_path).resolve()), "overfit": str(Path(overfit_path).resolve()), "evidence": current_evidence}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("acceptance", "remaining", "all"), default="acceptance")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output-root", default="runs/matrix")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-uuid", default="")
    parser.add_argument("--encoder-path", default="models/sikuroberta")
    parser.add_argument("--data-dir", default="data/prepared")
    parser.add_argument("--profile-report")
    parser.add_argument("--overfit-report")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--timeout", type=float, default=None, help="optional seconds per run; leaves latest checkpoint resumable")
    args = parser.parse_args(argv)
    schedule = tasks(args.config_dir, args.output_root)
    if args.phase == "acceptance":
        selected = schedule[:1]
    elif args.phase == "remaining":
        selected = schedule[1:]
    else:
        selected = schedule
    root = Path(args.output_root)
    registry_path = root / "completed.json"
    registry = {"protocol": "7_configs_x_5_seeds", "seeds": list(SEEDS), "configs": list(CONFIG_NAMES), "completed": {}}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for task in schedule:
        record = completed_run(task, load_config(task["config"], {"encoder_path": args.encoder_path, "data_dir": args.data_dir}))
        if record:
            registry["completed"][task["id"]] = record
        else:
            registry["completed"].pop(task["id"], None)
    if args.dry_run:
        print(json.dumps({"phase": args.phase, "tasks": selected, "completed": list(registry["completed"])}, ensure_ascii=False, indent=2))
        return 0
    verifier = Path(__file__).resolve().parents[1] / "scripts/verify_inputs.py"
    verified = subprocess.run([sys.executable, str(verifier)])
    if verified.returncode:
        raise RuntimeError("official input/model verification failed; no training was dispatched")
    acceptance_config = load_config(schedule[0]["config"], {"encoder_path": args.encoder_path, "data_dir": args.data_dir})
    gates = validate_preflight(args.profile_report or root.parent / "profile/profile.json",
                              args.overfit_report or root.parent / "overfit/overfit_summary.json", acceptance_config)
    registry["preflight"] = gates
    if args.phase == "remaining" and schedule[0]["id"] not in registry["completed"]:
        raise RuntimeError("acceptance run is not complete: run --phase acceptance first; smoke/subset runs do not qualify")
    root.mkdir(parents=True, exist_ok=True)
    registry["summary"] = aggregate_completed(registry["completed"])
    save_json(registry_path, registry)
    save_json(root / "summary.json", registry["summary"])
    started = 0
    for task in selected:
        if task["id"] in registry["completed"]:
            continue
        if args.max_runs is not None and started >= args.max_runs:
            break
        config = Path(task["config"])
        if not config.is_file():
            raise FileNotFoundError(config)
        command = [sys.executable, "-m", "gef.engine", "train", "--config", str(config),
                   "--seed", str(task["seed"]), "--output-dir", task["output_dir"],
                   "--device", args.device, "--gpu-uuid", args.gpu_uuid,
                   "--encoder-path", args.encoder_path, "--data-dir", args.data_dir,
                   "--resume", "auto"]
        folder = Path(task["output_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        save_json(folder / "launch.json", {"command": command, "cwd": str(Path.cwd()), "started_unix": time.time()})
        print(json.dumps({"event": "launch", "id": task["id"], "command": command}), flush=True)
        with (folder / "console.log").open("a", encoding="utf-8") as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                save_json(folder / "launch_failure.json", {"reason": "timeout", "timeout_seconds": args.timeout})
                return 2
        started += 1
        record = completed_run(task, load_config(task["config"], {"encoder_path": args.encoder_path, "data_dir": args.data_dir}))
        if returncode != 0 or record is None:
            save_json(folder / "launch_failure.json", {"reason": "process_or_acceptance_failure", "returncode": returncode})
            return returncode if returncode else 2
        registry["completed"][task["id"]] = record
        registry["summary"] = aggregate_completed(registry["completed"])
        save_json(registry_path, registry)
        save_json(root / "summary.json", registry["summary"])
        # `all` is still sequential: seed-42 GEF acceptance is completed before
        # the remaining configurations are dispatched.
    print(json.dumps({"completed": len(registry["completed"]), "total": 35, "registry": str(registry_path.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
