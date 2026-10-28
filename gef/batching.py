"""True encoder minibatching without changing the frozen GEF implementation.

Each sample keeps its original character positions, complete grid, loss and CRF
chain. Only encoder windows share a padded BERT call. Position IDs restart at
zero for each window, and overlap averaging follows GEFModel.encode_characters.

With dropout disabled, outputs/gradients agree with sequential calls within
floating-point tolerances. Training dropout consumes RNG in a different order
when windows are batched; accelerated runs must record this protocol and this
module's source hash rather than promise bitwise replay of legacy training.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

from .grid import upper_indices


BATCHING_PROTOCOL_VERSION = 1


def source_sha256() -> str:
    """Fingerprint the independent accelerated implementation for checkpoints."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_inputs(input_ids_list: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    inputs = list(input_ids_list)
    if not inputs:
        raise ValueError("input_ids_list must contain at least one sample")
    for ids in inputs:
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or len(ids) < 1:
            raise ValueError("every sample must be a nonempty 1-D character-ID tensor")
        if ids.dtype != torch.long:
            raise ValueError("character IDs must use torch.long")
        if ids.device != inputs[0].device:
            raise ValueError("all samples in an encoder batch must use the same device")
    return inputs


def encode_characters_batch(model, input_ids_list, *, encoder_batch_size=None, pad_token_id=None):
    """Return a projected, unpadded [characters,d] tensor for every input.

    ``encoder_batch_size`` bounds the number of windows in one encoder call;
    its default is the number of input samples. A single long input therefore
    stays memory conservative by default. A sample-major window queue preserves
    the original window starts/order, including the final short overlapping one.
    No padding/special-token position enters a grid or any loss.
    """
    inputs = _validate_inputs(input_ids_list)
    batch_size = len(inputs) if encoder_batch_size is None else encoder_batch_size
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("encoder_batch_size must be a positive integer")
    if pad_token_id is None:
        pad_token_id = getattr(model.encoder.config, "pad_token_id", None)
    pad_token_id = 0 if pad_token_id is None else int(pad_token_id)
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be nonnegative")
    device = inputs[0].device
    totals = [None] * len(inputs)
    counts = [torch.zeros(len(ids), device=device, dtype=torch.float32) for ids in inputs]
    windows, locations = [], []
    stride = model.encoder_window - model.encoder_overlap
    for sample, ids in enumerate(inputs):
        for start in range(0, len(ids), stride):
            stop = min(start + model.encoder_window, len(ids))
            windows.append(torch.cat((ids.new_tensor([model.cls_token_id]), ids[start:stop],
                                      ids.new_tensor([model.sep_token_id]))))
            locations.append((sample, start, stop))
            if stop == len(ids):
                break
    for begin in range(0, len(windows), batch_size):
        pieces = windows[begin:begin + batch_size]
        padded = pad_sequence(pieces, batch_first=True, padding_value=pad_token_id)
        lengths = torch.tensor([len(piece) for piece in pieces], device=device, dtype=torch.long)
        mask = (torch.arange(padded.shape[1], device=device)[None, :] < lengths[:, None]).long()
        hidden = model.encoder(input_ids=padded, attention_mask=mask).last_hidden_state
        for offset, (sample, start, stop) in enumerate(locations[begin:begin + batch_size]):
            original = hidden[offset, 1:stop - start + 1]
            if totals[sample] is None:
                totals[sample] = original.new_zeros((len(inputs[sample]), original.shape[-1]))
            indices = torch.arange(start, stop, device=device)
            # Each index_add follows the original per-sample window order.
            totals[sample] = totals[sample].index_add(0, indices, original)
            counts[sample].index_add_(0, indices, torch.ones(stop - start, device=device))
    return [model.char_projection(total / count[:, None].to(total.dtype))
            for total, count in zip(totals, counts)]


def emissions_from_hidden(model, hidden):
    """Apply the unchanged per-sample head to one complete character sequence."""
    if model.model_type == "sequence_crf":
        return model.emission(model.sequence_projection(hidden))
    n = len(hidden)
    rows, cols = upper_indices(n, hidden.device)
    context = torch.cat([torch.cummax(hidden[i:], dim=0).values for i in range(n)], dim=0)
    bilinear_left = torch.einsum("nd,kde->nke", hidden, model.bilinear)
    outputs = []
    for begin in range(0, len(rows), model.feature_chunk_size):
        stop = begin + model.feature_chunk_size
        args = (hidden, bilinear_left, context[begin:stop], rows[begin:stop], cols[begin:stop])
        if model.gradient_checkpointing and model.training and torch.is_grad_enabled():
            output = checkpoint(model._feature_chunk, *args, use_reentrant=False)
        else:
            output = model._feature_chunk(*args)
        outputs.append(output)
    return torch.cat(outputs, dim=0)


def forward_batch(model, input_ids_list, *, encoder_batch_size=None, pad_token_id=None):
    """Return emissions in input order; every grid/CRF remains sample-specific."""
    representations = encode_characters_batch(model, input_ids_list,
                                               encoder_batch_size=encoder_batch_size,
                                               pad_token_id=pad_token_id)
    return [emissions_from_hidden(model, hidden) for hidden in representations]


def loss_batch(model, input_ids_list, words_list, *, encoder_batch_size=None, pad_token_id=None):
    """Mean of original per-sample token-normalized losses, never a padded mean.

    An effective-batch trainer can instead call ``forward_batch`` and sum each
    ``model.loss_from_emissions`` divided by its actual effective sample count.
    """
    inputs, words = list(input_ids_list), list(words_list)
    if len(inputs) != len(words):
        raise ValueError("input_ids_list and words_list must have the same length")
    outputs = forward_batch(model, inputs, encoder_batch_size=encoder_batch_size, pad_token_id=pad_token_id)
    return torch.stack([model.loss_from_emissions(output, gold) for output, gold in zip(outputs, words)]).mean()
