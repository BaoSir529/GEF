"""Word/grid codecs and an explicit, deterministic consistency projection.

The projection is an implementation completion, not a recovered original GEF
decoder. It finds the legal contiguous word partition closest in Hamming
distance to the predicted upper-triangular grid.
"""
from __future__ import annotations

from math import isqrt
from typing import Sequence

import numpy as np
import torch


NULL_ID = 0


def upper_indices(n: int, device=None) -> torch.Tensor:
    if n < 1:
        raise ValueError("A grid requires at least one character")
    return torch.triu_indices(n, n, device=device)


def flattened_index(i, j, n: int):
    """Zero-based upper-row-major index, valid for 0 <= i <= j < n."""
    return i * n - i * (i - 1) // 2 + (j - i)


def length_from_grid_cells(cells: int) -> int:
    n = (isqrt(8 * cells + 1) - 1) // 2
    if n < 1 or n * (n + 1) // 2 != cells:
        raise ValueError(f"{cells} is not a positive triangular grid length")
    return n


def validate_words(words: Sequence[dict], n: int) -> None:
    cursor = 0
    for word in words:
        start, end = int(word["start"]), int(word["end"])
        if start != cursor or end <= start or end > n:
            raise ValueError("Words must be ordered, nonempty and cover all characters")
        if not isinstance(word["pos"], str) or not word["pos"]:
            raise ValueError("Each word must have a nonempty POS string")
        cursor = end
    if cursor != n:
        raise ValueError(f"Word coverage ends at {cursor}, expected {n}")


def grid_labels_numpy(words: Sequence[dict], n: int, pos_tags: Sequence[str]) -> np.ndarray:
    """POS iff the two characters belong to the *same word*, else NULL.

    Equal POS on different words never creates a link. End offsets are exclusive.
    """
    validate_words(words, n)
    mapping = {p: i + 1 for i, p in enumerate(pos_tags)}
    word_ids = np.empty(n, dtype=np.int64)
    character_pos = np.empty(n, dtype=np.int64)
    for wi, word in enumerate(words):
        if word["pos"] not in mapping:
            raise ValueError(f"POS {word['pos']!r} is outside the model tag inventory")
        start, end = int(word["start"]), int(word["end"])
        word_ids[start:end] = wi
        character_pos[start:end] = mapping[word["pos"]]
    rows, cols = np.triu_indices(n)
    return np.where(word_ids[rows] == word_ids[cols], character_pos[rows], NULL_ID)


def grid_gold(words, n: int, pos_tags: Sequence[str], device=None) -> torch.Tensor:
    return torch.as_tensor(grid_labels_numpy(words, n, pos_tags), dtype=torch.long, device=device)


def project_grid(predictions, n: int, pos_tags: Sequence[str]) -> dict:
    """Nearest legal word-block grid, using every predicted upper-triangle cell.

    A span tagged p changes its internal cells from the all-NULL reference to p.
    Its gain in number of matching cells is sum(1[y=p] - 1[y=NULL]). A prefix DP
    maximizes this gain. Ties prefer the shortest final word, then the lowest POS
    ID. This is not constrained MAP under the original CRF distribution.

    Complexity O(n^2 * P), auxiliary memory O(n * P), excluding the input/output
    grid. No maximum word length or confidence threshold is introduced.
    """
    if not pos_tags:
        raise ValueError("At least one POS is required")
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().to(device="cpu", dtype=torch.long).numpy()
    labels = np.asarray(predictions, dtype=np.int64)
    expected = n * (n + 1) // 2
    if n < 1 or labels.shape != (expected,):
        raise ValueError(f"Expected {expected} grid predictions, got {labels.shape}")
    p_count = len(pos_tags)
    if labels.min() < 0 or labels.max() > p_count:
        raise ValueError("Grid prediction is outside the POS+NULL vocabulary")
    scores = np.zeros((n, p_count), dtype=np.int64)
    best = np.zeros(n + 1, dtype=np.int64)
    back_start = np.zeros(n + 1, dtype=np.int64)
    back_pos = np.zeros(n + 1, dtype=np.int64)
    for end in range(n):
        starts = np.arange(end + 1, dtype=np.int64)
        column = labels[flattened_index(starts, end, n)]
        gains = np.zeros((end + 1, p_count), dtype=np.int64)
        gains[column == NULL_ID, :] = -1
        nonnull = np.flatnonzero(column != NULL_ID)
        gains[nonnull, column[nonnull] - 1] = 1
        scores[:end + 1] += np.cumsum(gains[::-1], axis=0)[::-1]
        candidates = scores[:end + 1] + best[:end + 1, None]
        optimum = int(candidates.max())
        # Latest start is the shortest final word; POS ties use model order.
        start = int(np.flatnonzero((candidates == optimum).any(axis=1))[-1])
        pos = int(np.flatnonzero(candidates[start] == optimum)[0])
        best[end + 1] = optimum
        back_start[end + 1], back_pos[end + 1] = start, pos
    words = []
    end = n
    while end:
        start = int(back_start[end])
        words.append({"start": start, "end": end, "pos": pos_tags[int(back_pos[end])]})
        end = start
    words.reverse()
    repaired = grid_labels_numpy(words, n, pos_tags)
    conflicts = int(np.count_nonzero(repaired != labels))
    return {"words": words, "conflict_cells": conflicts, "grid_cells": expected,
            "conflict_sentence": int(conflicts > 0)}


def sequence_gold(words, n: int, pos_tags: Sequence[str], device=None) -> torch.Tensor:
    """BMES-POS: tag IDs are POS-major [B, M, E, S]."""
    validate_words(words, n)
    mapping = {p: i for i, p in enumerate(pos_tags)}
    labels = torch.empty(n, dtype=torch.long, device=device)
    for word in words:
        if word["pos"] not in mapping:
            raise ValueError(f"POS {word['pos']!r} is outside the model tag inventory")
        start, end = int(word["start"]), int(word["end"])
        base = 4 * mapping[word["pos"]]
        if end - start == 1:
            labels[start] = base + 3
        else:
            labels[start], labels[end - 1] = base, base + 2
            if end - start > 2:
                labels[start + 1:end - 1] = base + 1
    return labels


def bmes_constraints(pos_count: int):
    count = 4 * pos_count
    transitions = torch.zeros((count, count), dtype=torch.bool)
    starts = torch.zeros(count, dtype=torch.bool)
    ends = torch.zeros(count, dtype=torch.bool)
    for p in range(pos_count):
        b, m, e, s = (4 * p + i for i in range(4))
        starts[b] = starts[s] = True
        ends[e] = ends[s] = True
        transitions[b, m] = transitions[b, e] = True
        transitions[m, m] = transitions[m, e] = True
        for q in range(pos_count):
            transitions[e, 4 * q] = transitions[e, 4 * q + 3] = True
            transitions[s, 4 * q] = transitions[s, 4 * q + 3] = True
    return transitions, starts, ends


def sequence_to_words(labels: Sequence[int], pos_tags: Sequence[str]) -> dict:
    words = []
    start = None
    active_pos = None
    for i, label in enumerate(labels):
        pos, kind = divmod(int(label), 4)
        if pos < 0 or pos >= len(pos_tags):
            raise ValueError("Sequence label is outside the model inventory")
        if kind == 3:
            if start is not None:
                raise ValueError("Invalid BMES path")
            words.append({"start": i, "end": i + 1, "pos": pos_tags[pos]})
        elif kind == 0:
            if start is not None:
                raise ValueError("Invalid BMES path")
            start, active_pos = i, pos
        elif start is None or pos != active_pos:
            raise ValueError("Invalid BMES path")
        elif kind == 2:
            words.append({"start": start, "end": i + 1, "pos": pos_tags[pos]})
            start, active_pos = None, None
    if start is not None:
        raise ValueError("Unclosed BMES word")
    validate_words(words, len(labels))
    return {"words": words, "conflict_cells": 0, "grid_cells": 0, "conflict_sentence": 0}
