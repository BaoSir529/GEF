import itertools

import numpy as np
import pytest
import torch

from gef.crf import LinearChainCRF
from gef.grid import bmes_constraints, sequence_gold


def brute_scores(crf, emissions):
    scores = []
    paths = []
    transitions, start, end = [x.detach().double() for x in crf._scores()]
    for path in itertools.product(range(crf.num_tags), repeat=len(emissions)):
        score = start[path[0]] + end[path[-1]]
        for i, tag in enumerate(path):
            score = score + emissions[i, tag].double()
            if i:
                score = score + transitions[path[i - 1], tag]
        scores.append(score)
        paths.append(list(path))
    return torch.stack(scores), paths


@pytest.mark.parametrize("length", [1, 2, 4])
def test_partition_gold_and_viterbi_match_enumeration(length):
    torch.manual_seed(11)
    crf = LinearChainCRF(3, chunk_size=2)
    emissions = torch.randn(length, 3)
    scores, paths = brute_scores(crf, emissions)
    assert torch.allclose(crf.log_partition(emissions), torch.logsumexp(scores, dim=0), atol=2e-6)
    optimum = int(scores.argmax())
    assert crf.decode(emissions) == paths[optimum]
    tags = torch.tensor(paths[optimum])
    assert torch.allclose(crf.gold_score(emissions, tags), scores[optimum], atol=1e-7)


@pytest.mark.parametrize("chunk_size", [1, 4, 7])
def test_checkpoint_chunks_preserve_loss_and_gradients(chunk_size):
    torch.manual_seed(17)
    reference = LinearChainCRF(3, chunk_size=0)
    chunked = LinearChainCRF(3, chunk_size=chunk_size)
    chunked.load_state_dict(reference.state_dict())
    first = torch.randn(19, 3, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    tags = torch.randint(0, 3, (19,))
    loss_a = reference(first, tags)
    loss_b = chunked(second, tags)
    loss_a.backward()
    loss_b.backward()
    assert torch.allclose(loss_a, loss_b, atol=1e-7)
    assert torch.allclose(first.grad, second.grad, atol=2e-6)
    for a, b in zip(reference.parameters(), chunked.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=2e-6)


def test_long_chain_is_finite_and_does_not_reset_chunks():
    torch.manual_seed(23)
    crf = LinearChainCRF(2, chunk_size=64)
    emissions = (torch.randn(1025, 2) * 5).requires_grad_(True)
    tags = torch.zeros(1025, dtype=torch.long)
    loss = crf(emissions, tags)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(emissions.grad).all()
    # Every original position remains in the objective, including the last one.
    assert emissions.grad[0].abs().sum() > 0
    assert emissions.grad[-1].abs().sum() > 0


def test_bmes_constraints_and_invalid_gold():
    crf = LinearChainCRF(8, 3, *bmes_constraints(2))
    emissions = torch.randn(4, 8)
    labels = sequence_gold([{"start": 0, "end": 3, "pos": "n"},
                            {"start": 3, "end": 4, "pos": "v"}], 4, ("n", "v"))
    assert torch.isfinite(crf(emissions, labels))
    path = crf.decode(emissions)
    assert crf.allowed_starts[path[0]] and crf.allowed_ends[path[-1]]
    assert crf.allowed_transitions[path[:-1], path[1:]].all()
    with pytest.raises(ValueError):
        crf(emissions, torch.tensor([2, 3, 3, 3]))


def test_partition_gradients_match_differentiable_exhaustive_sum():
    torch.manual_seed(47)
    crf = LinearChainCRF(3, chunk_size=2)
    emissions = torch.randn(4, 3, requires_grad=True)
    transitions, start, end = crf._scores()
    enumerated = []
    for path in itertools.product(range(3), repeat=4):
        score = start[path[0]] + end[path[-1]]
        score = score + sum(emissions[i, tag] for i, tag in enumerate(path))
        score = score + sum(transitions[path[i - 1], path[i]] for i in range(1, 4))
        enumerated.append(score)
    expected = torch.logsumexp(torch.stack(enumerated), dim=0)
    actual = crf.log_partition(emissions)
    parameters = (emissions, crf.transitions, crf.start_transitions, crf.end_transitions)
    expected_grad = torch.autograd.grad(expected, parameters, retain_graph=True)
    actual_grad = torch.autograd.grad(actual, parameters)
    for got, want in zip(actual_grad, expected_grad):
        assert torch.allclose(got, want, atol=2e-6, rtol=2e-6)


def test_constrained_bmes_gradients_are_finite_for_one_character():
    crf = LinearChainCRF(8, 2, *bmes_constraints(2))
    emissions = torch.randn(1, 8, requires_grad=True)
    crf(emissions, torch.tensor([3])).backward()
    assert torch.isfinite(emissions.grad).all()
    assert torch.isfinite(crf.start_transitions.grad).all()
    assert torch.isfinite(crf.end_transitions.grad).all()


@pytest.mark.parametrize("length", [1, 2, 3, 18, 19, 65])
@pytest.mark.parametrize("constrained", [False, True])
def test_balanced_semiring_partition_and_all_gradients_match_reference(length, constrained):
    torch.manual_seed(61)
    constraints = bmes_constraints(2) if constrained else (None, None, None)
    tags = 8 if constrained else 3
    crf = LinearChainCRF(tags, 4, *constraints)
    emissions = torch.randn(length, tags, requires_grad=True)
    reference = crf.log_partition_reference(emissions)
    parallel = crf.log_partition_parallel(emissions)
    assert torch.allclose(reference, parallel, atol=2e-5, rtol=2e-6)
    parameters = (emissions, crf.transitions, crf.start_transitions, crf.end_transitions)
    expected = torch.autograd.grad(reference, parameters, retain_graph=True, allow_unused=True)
    actual = torch.autograd.grad(parallel, parameters, allow_unused=True)
    for a, b in zip(actual, expected):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert torch.isfinite(a).all()
            assert torch.allclose(a, b, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("constrained", [False, True])
def test_cuda_balanced_partition_and_all_gradients_match_cpu_reference(constrained):
    torch.manual_seed(67)
    constraints = bmes_constraints(2) if constrained else (None, None, None)
    tags = 8 if constrained else 16
    cpu = LinearChainCRF(tags, 16, *constraints)
    gpu = LinearChainCRF(tags, 16, *constraints).cuda()
    gpu.load_state_dict(cpu.state_dict())
    first = torch.randn(137, tags, requires_grad=True)
    second = first.detach().cuda().requires_grad_(True)
    reference, parallel = cpu.log_partition_reference(first), gpu.log_partition(second)
    assert torch.allclose(reference, parallel.cpu(), atol=1e-4, rtol=2e-6)
    reference.backward()
    parallel.backward()
    assert torch.allclose(first.grad, second.grad.cpu(), atol=8e-5, rtol=8e-5)
    for p, q in zip(cpu.parameters(), gpu.parameters()):
        assert torch.isfinite(q.grad).all()
        assert torch.allclose(p.grad, q.grad.cpu(), atol=1e-4, rtol=1e-4)


def test_long_balanced_chain_matches_closed_form_and_expected_gradients():
    length, count = 10001, 2
    crf = LinearChainCRF(count, 128)
    for parameter in crf.parameters():
        parameter.detach().zero_()
    emissions = torch.zeros(length, count, requires_grad=True)
    partition = crf.log_partition_parallel(emissions)
    assert abs(float(partition.detach()) - length * np.log(count)) < 1e-3
    partition.backward()
    assert torch.allclose(emissions.grad, torch.full_like(emissions, 1 / count), atol=2e-5)
    assert torch.allclose(crf.transitions.grad, torch.full_like(crf.transitions, (length - 1) / count ** 2),
                          atol=.02, rtol=1e-5)
    assert torch.allclose(crf.start_transitions.grad, torch.full_like(crf.start_transitions, .5), atol=2e-5)
    assert torch.allclose(crf.end_transitions.grad, torch.full_like(crf.end_transitions, .5), atol=2e-5)
