"""Fetch and verify fixed official inputs; never overwrite supplied originals."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen, build_opener, install_opener, ProxyHandler

REPO_REVISION = "2ac2af22127c33fbf07cf1bf2ae2055415228f3e"
MODEL_REVISION = "bb25260d5c321924fe4fb353c09191c0aaf5c5c6"
MODEL_ID = "SIKU-BERT/sikuroberta"
FILES = (
    "zuozhuan_train_utf8.zip", "EvaHan_testa_raw.txt", "EvaHan_testa_gold.txt",
    "EvaHan_testb_raw.txt", "EvaHan_testb_gold.txt", "eval_EvaHan_2022_FINAL.py",
    "eval_EvaHan_2022.py", "EvaHan_scorer.zip", "EvaHan_guidelines_v1_training.pdf",
)


def get(url: str):
    return urlopen(Request(url, headers={"User-Agent": "GEF-EvaHan-Reproduction/1.0"}), timeout=90)


def get_json(url: str):
    with get(url) as response:
        return json.load(response)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def download(url: str, path: Path, expected_size=None, expected_sha=None):
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        if expected_sha and sha256(path) == expected_sha:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    for attempt in range(3):
        try:
            with get(url) as response, partial.open("wb") as output:
                received = 0
                next_notice = 16 * 1024 * 1024
                while block := response.read(1024 * 1024):
                    output.write(block)
                    received += len(block)
                    if received >= next_notice:
                        print(f"{path.name}: {received / 1048576:.0f} MiB", flush=True)
                        next_notice += 16 * 1024 * 1024
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise ValueError(f"Size mismatch for {path.name}")
            if expected_sha and sha256(partial) != expected_sha:
                raise ValueError(f"SHA256 mismatch for {path.name}")
            os.replace(partial, path)
            return
        except Exception as error:
            print(f"Download attempt {attempt + 1}: {type(error).__name__}: {error}", flush=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--proxy", help="Optional process-local HTTP proxy; does not change Windows settings")
    args = parser.parse_args()
    if args.proxy:
        install_opener(build_opener(ProxyHandler({"http": args.proxy, "https": args.proxy})))
    root = args.root.resolve()
    report_path = root / "reports" / "provenance.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    original = root / "LT4HALA-master" / "2022" / "data_and_doc"
    entries = get_json("https://api.github.com/repos/CIRCSE/LT4HALA/contents/2022/data_and_doc?ref=" + REPO_REVISION)
    entries = {entry["name"]: entry for entry in entries}
    report = {"official_repo": "https://github.com/CIRCSE/LT4HALA", "repo_revision": REPO_REVISION,
              "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "data": [], "model": []}
    for name in FILES:
        entry = entries[name]
        path = original / name
        matches = path.exists() and git_blob(path) == entry["sha"]
        if not matches:
            path = root / "data" / "official_snapshot" / name
            url = f"https://raw.githubusercontent.com/CIRCSE/LT4HALA/{REPO_REVISION}/2022/data_and_doc/{name}"
            download(url, path, entry["size"])
            if git_blob(path) != entry["sha"]:
                raise ValueError("Official git blob mismatch: " + name)
        report["data"].append({"name": name, "path": str(path.relative_to(root)), "size": path.stat().st_size,
                               "sha256": sha256(path), "git_blob_sha1": entry["sha"], "supplied_copy_matches": matches})
        print(f"Official data verified: {name} (supplied={matches})", flush=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.data_only:
        return
    metadata = get_json(f"https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REVISION}?blobs=true")
    siblings = {item["rfilename"]: item for item in metadata["siblings"]}
    model_dir = root / "models" / "sikuroberta"
    for name in ("config.json", "vocab.txt", "README.md", "pytorch_model.bin"):
        item = siblings[name]
        lfs = item.get("lfs") or {}
        expected_sha = lfs.get("sha256")
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
        path = model_dir / name
        download(url, path, item.get("size"), expected_sha)
        blob_id = item.get("blobId")
        if not lfs and blob_id and git_blob(path) != blob_id:
            raise ValueError("Model git blob mismatch: " + name)
        report["model"].append({"name": name, "size": path.stat().st_size, "sha256": sha256(path),
                                "url": url, "lfs_sha256": expected_sha, "git_blob_sha1": blob_id})
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Model file verified: {name}", flush=True)
    (model_dir / "revision.json").write_text(json.dumps({"model_id": MODEL_ID, "revision": MODEL_REVISION}, indent=2) + "\n", encoding="utf-8")
    print(f"Provenance: {report_path}", flush=True)


if __name__ == "__main__":
    main()
