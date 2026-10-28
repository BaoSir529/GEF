import itertools

import numpy as np
import pytest
import torch

from gef.grid import (bmes_constraints, flattened_index, grid_gold, grid_labels_numpy,
                      length_from_grid_cells, project_grid, sequence_gold,
                      sequence_to_words, upper_indices)


POS = ("n", "v")


def partitions(n, start=0):
    if start == n:
        yield []
    for end in range(start + 1, n + 1):
        for pos in POS:
            for rest in partitions(n, end):
                yield [{"start": start, "end": end, "pos": pos}] + rest


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
def test_upper_index_bijection(n):
    rows, cols = upper_indices(n)
    assert torch.equal(flattened_index(rows, cols, n), torch.arange(n * (n + 1) // 2))
    assert length_from_grid_cells(len(rows)) == n


def test_same_pos_on_distinct_words_does_not_link():
    words = [{"start": 0, "end": 1, "pos": "n"}, {"start": 1, "end": 3, "pos": "n"}]
    labels = grid_labels_numpy(words, 3, POS)
    assert labels[flattened_index(0, 1, 3)] == 0
    assert labels[flattened_index(1, 2, 3)] == 1
    assert project_grid(labels, 3, POS)["words"] == words


def test_every_small_gold_partition_roundtrips():
    for n in range(1, 5):
        for words in partitions(n):
            labels = grid_gold(words, n, POS)
            result = project_grid(labels, n, POS)
            assert result["words"] == words
            assert result["conflict_cells"] == 0
            assert result["conflict_sentence"] == 0


def test_projection_matches_brute_force_hamming_optimum():
    rng = np.random.default_rng(2022)
    for n in range(1, 5):
        candidates = list(partitions(n))
        for _ in range(12):
            predicted = rng.integers(0, len(POS) + 1, n * (n + 1) // 2)
            result = project_grid(predicted, n, POS)
            best = min(int(np.count_nonzero(grid_labels_numpy(words, n, POS) != predicted))
                       for words in candidates)
            assert result["conflict_cells"] == best
            assert result["words"][0]["start"] == 0
            assert result["words"][-1]["end"] == n


def test_null_prediction_has_deterministic_complete_fallback():
    result = project_grid(np.zeros(6, dtype=int), 3, POS)
    assert result["words"] == [{"start": i, "end": i + 1, "pos": "n"} for i in range(3)]
    assert result["conflict_cells"] == 3
    assert result["conflict_sentence"] == 1


def test_equal_hamming_scores_prefer_shorter_final_word():
    # Both one n-word and two n/v singletons need one cell repair.
    result = project_grid(np.array([1, 1, 2]), 2, POS)
    assert result["words"] == [{"start": 0, "end": 1, "pos": "n"},
                               {"start": 1, "end": 2, "pos": "v"}]
    assert result["conflict_cells"] == 1


def test_bmes_gold_and_constraints():
    words = [{"start": 0, "end": 3, "pos": "n"}, {"start": 3, "end": 4, "pos": "v"}]
    labels = sequence_gold(words, 4, POS)
    assert labels.tolist() == [0, 1, 2, 7]
    assert sequence_to_words(labels.tolist(), POS)["words"] == words
    transitions, starts, ends = bmes_constraints(2)
    assert starts[labels[0]] and ends[labels[-1]]
    assert transitions[labels[:-1], labels[1:]].all()
    assert not transitions[0, 6]  # B-n cannot finish with E-v.


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        length_from_grid_cells(5)
    with pytest.raises(ValueError):
        grid_gold([{"start": 1, "end": 2, "pos": "n"}], 2, POS)
    with pytest.raises(ValueError):
        project_grid([1, 7, 1], 2, POS)
