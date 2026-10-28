# GEF for EvaHan2022

Source code for joint ancient Chinese word segmentation and part-of-speech (POS) tagging with a grid-based encoding framework (GEF).

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the implementation, training configurations, input-download utilities, and unit tests. It does **not** distribute experimental results, analysis reports, training logs, figures, corpora, pretrained models, or trained checkpoints. Such artifacts are generated locally and excluded by `.gitignore`.

## Implementation

- Character representations from SIKU-RoBERTa, projected to 128 dimensions.
- Character-pair features combining endpoints, interval max-pooling, and a bilinear relation.
- Upper-triangular grid labeling with a CRF and a deterministic full-coverage word-partition decoder.
- Two controls: grid-cell softmax without a CRF, and character-level BMES–POS CRF sequence labeling.
- Training-only same-POS word substitution, deterministic grouped train/development splitting, and official/independent span-based scoring.
- Sliding encoder windows for long lines and optional encoder micro-batching.

The code is a reconstruction of the method. Implementation details that were not specified in the original manuscript are explicit in the source and configurations; this release does not claim to recover undocumented historical code or guarantee identical historical scores.

## Installation

Python 3.11 is recommended. From the repository root:

```shell
conda env create -f environment.yml
conda activate gef-evahan
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

Alternatively, create a Python 3.11 environment, install the appropriate PyTorch build for your machine, then run `python -m pip install -r requirements.txt`. Transformers is pinned to 4.56.2. CUDA 12.6 wheels require a compatible NVIDIA driver. CPU execution is available via `--device cpu` for small checks; full-grid training is intended for a CUDA GPU.

## Obtain external inputs

Data and the official scorer come from [CIRCSE/LT4HALA](https://github.com/CIRCSE/LT4HALA/tree/master/2022), revision `2ac2af22127c33fbf07cf1bf2ae2055415228f3e`. The encoder is [SIKU-BERT/sikuroberta](https://huggingface.co/SIKU-BERT/sikuroberta), revision `bb25260d5c321924fe4fb353c09191c0aaf5c5c6`.

```shell
python scripts/fetch_assets.py
python scripts/verify_inputs.py
python -m gef.data prepare --source-dir data/official_snapshot
```

The downloader obtains and verifies the fixed revisions. On a fresh clone, official files are stored in `data/official_snapshot/` and model files in `models/sikuroberta/`. The provenance manifest is generated locally in `reports/`. Downloading the model requires approximately 436 MB for its weight file. Use `--data-only` to fetch only official inputs, or `--proxy http://host:port` for an optional process-local proxy. External data, scorer, and model terms remain applicable; they are not relicensed by this repository.

Preparation preserves original characters and physical line identities, normalizes documented POS aliases, quarantines malformed training lines, groups identical source texts before splitting, and derives augmentation pools from training data only. Official test gold is not rewritten.

Configuration identifiers `native21` and `paper15` designate, respectively, the official POS inventory and the fixed merged inventory. Both have augmented-training configurations; `paper15_gef_noaug.json` disables augmentation. These identifiers describe label protocols, not additional downloadable datasets.

## Tests and preliminary checks

```shell
python -m pytest tests -q
python -m gef.engine profile --config configs/paper15_gef_aug.json --output-dir runs/profile
python -m gef.engine overfit --config configs/paper15_gef_aug.json --seed 42 --output-dir runs/overfit --samples 32 --steps 300
```

Tests use tiny synthetic fixtures and do not require downloaded model weights. The official-scorer integration test is skipped until the external scorer has been downloaded. Profiling and overfitting require the external inputs; their outputs stay local.

## Training

```shell
python -m gef.engine train --config configs/native21_gef_aug.json --seed 42 --output-dir runs/native21_gef/seed_42 --resume auto
```

Select another JSON file to run the softmax, sequence-labeling, merged-inventory, or no-augmentation setting. Each configuration trains for 10 epochs, with effective batch size 16 and development PF/WF-based checkpoint selection. The standard engine uses micro-batch size 1. To use the batched encoder engine:

```shell
python -m gef.fast_engine train --config configs/native21_gef_aug.json --micro-batch-size 4 --seed 42 --output-dir runs/native21_gef_fast/seed_42 --resume auto
```

The effective batch remains 16. Micro-batching may change floating-point reductions and dropout draws; use one execution protocol consistently when comparing runs. The two engines use different resume protocols: do not exchange their training checkpoints.

GPU selection is portable: no local GPU UUID is required. Use `--device cuda:0` or set `CUDA_VISIBLE_DEVICES` before starting Python. For two GPUs, launch independent commands in separate terminals with `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`, assigning different output directories and seeds. In PowerShell, set the variable with `$env:CUDA_VISIBLE_DEVICES = '0'` (or `'1'`).

`gef.matrix` additionally provides a sequential seven-configuration/five-seed scheduler. Its preflight requires successful local profile/overfit records. Run `python -m gef.matrix --help` for its options; it does not include any precomputed results or machine-specific background launchers.

## Prediction and scoring

```shell
python -m gef.engine predict --config configs/native21_gef_aug.json --checkpoint runs/native21_gef/seed_42/best.pt --split test_a --output-dir runs/predict_test_a
python -m gef.engine evaluate --config configs/native21_gef_aug.json --prediction runs/predict_test_a/prediction.txt --gold data/official_snapshot/EvaHan_testa_gold.txt --output-dir runs/score_test_a
```

For a merged-inventory evaluation, use a matching configuration and gold with the same label mapping; do not directly compare its POS score with official-inventory scores. Training performs full Test-A/Test-B evaluation after development-based selection. Independent span/POS scores and the official scorer status are recorded separately; a scorer failure is preserved rather than silently replaced.

## Source layout

| Path | Purpose |
|---|---|
| `gef/data.py` | Parsing, grouped splitting, POS mapping, augmentation |
| `gef/grid.py`, `gef/crf.py`, `gef/model.py` | Grid representation, structured decoding, neural model |
| `gef/engine.py`, `gef/fast_engine.py`, `gef/batching.py` | Training, prediction, optional micro-batching |
| `gef/evaluation.py` | Character-preserving output checks and span/POS scoring |
| `gef/matrix.py` | Sequential experiment scheduling and completion checks |
| `configs/` | Seven training configurations, without stored scores |
| `scripts/` | Pinned external-input download and verification |
| `tests/` | Unit and optional official-scorer integration tests |

Local experiment summaries, plotting/analysis tools, simulation utilities, workstation launchers, and research notes are intentionally omitted. Only this README is included as Markdown documentation.

## License

The original code and documentation in this repository are open-source licensed under the **MIT License**. See [LICENSE](LICENSE) for the complete terms.

You may use, modify, redistribute, sublicense, and use this code commercially, provided that the copyright and license notices are retained in copies or substantial portions of the software. The software is provided without warranty.

The MIT License applies only to material authored for this project. EvaHan2022 data and official scoring scripts, SIKU-RoBERTa weights and associated files, and third-party dependencies remain subject to their respective licenses and usage terms. Downloading them with this project's utilities does not relicense them under MIT.

If you use this implementation in academic work, please cite the accompanying GEF paper and acknowledge the original dataset and pretrained-model publications. This is a scholarly request, not an additional restriction on the MIT License.
