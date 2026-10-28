"""Offline gate: require the pinned official model and verified original data."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "eede4853e143e741fb65cca1607c453116e172517ea31a7d9a3556d31d64e81f"
MODEL_SIZE = 435860951


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    errors = []
    path = ROOT / "reports/provenance.json"
    if path.exists():
        provenance = json.loads(path.read_text(encoding="utf-8"))
        for item in provenance.get("data", []):
            source = ROOT / item["path"]
            if not source.exists() or digest(source) != item["sha256"]:
                errors.append("Original data missing/changed: " + item["name"])
        if len(provenance.get("data", [])) != 9:
            errors.append("Expected nine verified official files")
        for item in provenance.get("model", []):
            source = ROOT / "models/sikuroberta" / item["name"]
            if not source.exists() or digest(source) != item["sha256"]:
                errors.append("Pinned model file missing/changed: " + item["name"])
        if not {"config.json", "vocab.txt", "README.md"}.issubset({item["name"] for item in provenance.get("model", [])}):
            errors.append("Model metadata provenance is incomplete")
    else:
        errors.append("Official provenance.json is missing")
    model = ROOT / "models/sikuroberta/pytorch_model.bin"
    if not model.exists():
        errors.append("Pinned SIKU-RoBERTa weights have not been downloaded")
    elif model.stat().st_size != MODEL_SIZE or digest(model) != MODEL_SHA:
        errors.append("Model does not match the pinned official SHA256/size")
    report = {"status": "ready" if not errors else "blocked", "errors": errors,
              "required_model_sha256": MODEL_SHA, "required_model_bytes": MODEL_SIZE}
    (ROOT / "reports/input_gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
