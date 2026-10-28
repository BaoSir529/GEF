"""Single-sequence linear-chain CRF with exact long-chain training.

Messages are normalized at every timestep in float32 and their removed scales
are accumulated in float64. CPU uses the sequential reference recurrence; CUDA
uses a balanced product of log-semiring transition matrices. Both compute the
same partition function. Checkpoint/reduction boundaries never reset the chain.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int, chunk_size: int = 256,
                 allowed_transitions=None, allowed_starts=None, allowed_ends=None):
        super().__init__()
        if num_tags < 1:
            raise ValueError("num_tags must be positive")
        self.num_tags, self.chunk_size = int(num_tags), int(chunk_size)
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.register_buffer("allowed_transitions", torch.ones(num_tags, num_tags, dtype=torch.bool)
                             if allowed_transitions is None else allowed_transitions.bool())
        self.register_buffer("allowed_starts", torch.ones(num_tags, dtype=torch.bool)
                             if allowed_starts is None else allowed_starts.bool())
        self.register_buffer("allowed_ends", torch.ones(num_tags, dtype=torch.bool)
                             if allowed_ends is None else allowed_ends.bool())
        for parameter in [self.transitions, self.start_transitions, self.end_transitions]:
            nn.init.uniform_(parameter, -0.1, 0.1)

    def _scores(self):
        return (self.transitions.float().masked_fill(~self.allowed_transitions, -torch.inf),
                self.start_transitions.float().masked_fill(~self.allowed_starts, -torch.inf),
                self.end_transitions.float().masked_fill(~self.allowed_ends, -torch.inf))

    def _validate(self, emissions, tags=None):
        if emissions.ndim != 2 or emissions.shape[0] < 1 or emissions.shape[1] != self.num_tags:
            raise ValueError(f"Expected nonempty emissions [T,{self.num_tags}]")
        if tags is not None:
            if tags.shape != (emissions.shape[0],):
                raise ValueError("Tags and emissions lengths differ")
            if tags.min().item() < 0 or tags.max().item() >= self.num_tags:
                raise ValueError("Tags are outside the CRF vocabulary")
            if not bool(self.allowed_starts[tags[0]]) or not bool(self.allowed_ends[tags[-1]]):
                raise ValueError("Gold path violates CRF start/end constraints")
            if len(tags) > 1 and not bool(self.allowed_transitions[tags[:-1], tags[1:]].all()):
                raise ValueError("Gold path violates CRF transition constraints")

    @staticmethod
    def _advance(alpha, offset, emissions, transitions):
        for emission in emissions.unbind(0):
            alpha = torch.logsumexp(alpha[:, None] + transitions, dim=0) + emission
            scale = torch.logsumexp(alpha, dim=0)
            alpha = alpha - scale
            offset = offset + scale.double()
        return alpha, offset

    def log_partition_reference(self, emissions):
        """Sequential O(T K^2) reference, also used for CPU training."""
        self._validate(emissions)
        transitions, start, end = self._scores()
        values = emissions.float()
        alpha = start + values[0]
        scale = torch.logsumexp(alpha, dim=0)
        alpha, offset = alpha - scale, scale.double()
        chunk = self.chunk_size if self.chunk_size > 0 else len(values)
        for begin in range(1, len(values), chunk):
            piece = values[begin:begin + chunk]
            if self.training and torch.is_grad_enabled() and self.chunk_size > 0:
                alpha, offset = checkpoint(self._advance, alpha, offset, piece, transitions,
                                          use_reentrant=False)
            else:
                alpha, offset = self._advance(alpha, offset, piece, transitions)
        return offset + torch.logsumexp(alpha + end, dim=0).double()

    @staticmethod
    def _safe_logsumexp(values, dim):
        # A forbidden BMES endpoint can have no incoming path in an intermediate
        # matrix. logsumexp([-inf,...]) has undefined gradients in torch. Mask
        # that reduction before evaluating it, then restore its exact -inf value.
        active = torch.isfinite(values).any(dim=dim, keepdim=True)
        safe = torch.where(active, values, torch.zeros_like(values))
        return torch.logsumexp(safe, dim=dim).masked_fill(~active.squeeze(dim), -torch.inf)

    @classmethod
    def _leaf_matrices(cls, values, transitions):
        matrices = transitions[None, :, :] + values[:, None, :]
        scales = torch.logsumexp(matrices.flatten(1), dim=1)
        return matrices - scales[:, None, None], scales.double()

    @classmethod
    def _merge_matrices(cls, left, right):
        # C[i,j] = logsumexp_k(A[i,k] + B[k,j]); associative, order preserving.
        matrices = cls._safe_logsumexp(left[:, :, :, None] + right[:, None, :, :], dim=2)
        scales = torch.logsumexp(matrices.flatten(1), dim=1)
        return matrices - scales[:, None, None], scales.double()

    def log_partition_parallel(self, emissions):
        """Exact log-semiring balanced reduction, differentiable on CPU/CUDA.

        Arithmetic cost is O(T K^3), trading extra small-matrix work for O(log T)
        dependency depth. Pair batches are bounded to avoid a full [T,K,K,K]
        tensor. Every merge carries its float64 normalization offset, including
        an unpaired final matrix. CPU callers retain the cheaper reference.
        """
        self._validate(emissions)
        transitions, start, end = self._scores()
        values = emissions.float()
        alpha = start + values[0]
        scale = torch.logsumexp(alpha, dim=0)
        alpha, offset = alpha - scale, scale.double()
        if len(values) == 1:
            return offset + self._safe_logsumexp(alpha + end, dim=0).double()
        recompute = self.training and torch.is_grad_enabled() and self.chunk_size > 0
        if recompute:
            matrices, offsets = checkpoint(self._leaf_matrices, values[1:], transitions,
                                            use_reentrant=False)
        else:
            matrices, offsets = self._leaf_matrices(values[1:], transitions)
        # Bound each broadcast temporary to <= 8M float32 elements (~32 MiB).
        batch = min(max(self.chunk_size, 1), max(1, 8_388_608 // self.num_tags ** 3))
        while len(matrices) > 1:
            pairs = len(matrices) // 2
            merged, merged_offsets = [], []
            for begin in range(0, pairs, batch):
                stop = min(begin + batch, pairs)
                left, right = matrices[2 * begin:2 * stop:2], matrices[2 * begin + 1:2 * stop:2]
                if recompute:
                    output, scales = checkpoint(self._merge_matrices, left, right, use_reentrant=False)
                else:
                    output, scales = self._merge_matrices(left, right)
                merged.append(output)
                merged_offsets.append(offsets[2 * begin:2 * stop:2]
                                      + offsets[2 * begin + 1:2 * stop:2] + scales)
            if len(matrices) % 2:
                merged.append(matrices[-1:])
                merged_offsets.append(offsets[-1:])
            matrices, offsets = torch.cat(merged), torch.cat(merged_offsets)
        final = self._safe_logsumexp(alpha[:, None] + matrices[0], dim=0) + end
        return offset + offsets[0] + self._safe_logsumexp(final, dim=0).double()

    def log_partition(self, emissions):
        # CUDA launch overhead dominates the sequential recurrence for grids.
        # The reference remains independently callable for numerical audits.
        if emissions.is_cuda and len(emissions) > 1:
            return self.log_partition_parallel(emissions)
        return self.log_partition_reference(emissions)

    def gold_score(self, emissions, tags):
        self._validate(emissions, tags)
        transitions, start, end = self._scores()
        values = emissions.float()
        score = start[tags[0]].double() + end[tags[-1]].double()
        score = score + values.gather(1, tags[:, None]).squeeze(1).double().sum()
        if len(tags) > 1:
            score = score + transitions[tags[:-1], tags[1:]].double().sum()
        return score

    def neg_log_likelihood(self, emissions, tags, reduction="token_mean"):
        nll = self.log_partition(emissions) - self.gold_score(emissions, tags)
        if reduction == "token_mean":
            nll = nll / len(emissions)
        elif reduction not in ("none", "sum"):
            raise ValueError("reduction must be token_mean, none, or sum")
        return nll.float()

    def forward(self, emissions, tags, reduction="token_mean"):
        return self.neg_log_likelihood(emissions, tags, reduction)

    @torch.no_grad()
    def decode(self, emissions):
        self._validate(emissions)
        transitions, start, end = (x.detach().cpu().numpy().astype(np.float64)
                                   for x in self._scores())
        values = emissions.detach().float().cpu().numpy()
        time, count = values.shape
        pointers = np.zeros((time, count), dtype=np.int16 if count < 32768 else np.int32)
        alpha = start + values[0]
        for index in range(1, time):
            candidates = alpha[:, None] + transitions
            pointers[index] = np.argmax(candidates, axis=0)
            alpha = candidates[pointers[index], np.arange(count)] + values[index]
            alpha -= np.max(alpha)
        final = alpha + end
        if not np.isfinite(final).any():
            raise ValueError("No valid CRF path")
        tag = int(np.argmax(final))
        path = [tag]
        for index in range(time - 1, 0, -1):
            tag = int(pointers[index, tag])
            path.append(tag)
        return list(reversed(path))
