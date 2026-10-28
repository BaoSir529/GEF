"""A documented GEF reimplementation; omitted paper details are explicit choices."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoTokenizer

from .crf import LinearChainCRF
from .grid import (bmes_constraints, grid_gold, length_from_grid_cells, project_grid,
                   sequence_gold, sequence_to_words, upper_indices)


class GEFModel(nn.Module):
    def __init__(self, encoder_path, pos_tags, model_type="gef", d=128, t=64, dropout=.1,
                 encoder_window=510, encoder_overlap=128, crf_chunk_size=256,
                 feature_chunk_size=4096, gradient_checkpointing=True, *, encoder=None,
                 cls_token_id=None, sep_token_id=None):
        super().__init__()
        if model_type not in {"gef", "grid_softmax", "sequence_crf"}:
            raise ValueError(f"Unknown model type {model_type!r}")
        if not pos_tags or len(set(pos_tags)) != len(pos_tags):
            raise ValueError("pos_tags must be nonempty and unique")
        if "__NULL__" in pos_tags:
            raise ValueError("__NULL__ is reserved for grid label 0")
        self.pos_tags, self.model_type = tuple(pos_tags), model_type
        self.d, self.t = int(d), int(t)
        self.encoder_window, self.encoder_overlap = int(encoder_window), int(encoder_overlap)
        self.feature_chunk_size = int(feature_chunk_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(
            str(encoder_path), local_files_only=True, trust_remote_code=False,
            add_pooling_layer=False)
        if cls_token_id is None:
            cls_token_id = getattr(self.encoder.config, "cls_token_id", None)
        if sep_token_id is None:
            sep_token_id = getattr(self.encoder.config, "sep_token_id", None)
        if (cls_token_id is None or sep_token_id is None) and encoder_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(str(encoder_path), local_files_only=True,
                                                       trust_remote_code=False)
            cls_token_id = tokenizer.cls_token_id if cls_token_id is None else cls_token_id
            sep_token_id = tokenizer.sep_token_id if sep_token_id is None else sep_token_id
        if cls_token_id is None or sep_token_id is None:
            raise ValueError("CLS/SEP IDs must come from the local tokenizer/config or explicit test injection")
        self.cls_token_id, self.sep_token_id = int(cls_token_id), int(sep_token_id)
        max_window = int(getattr(self.encoder.config, "max_position_embeddings", 512)) - 2
        if not 0 <= self.encoder_overlap < self.encoder_window <= max_window:
            raise ValueError("Require 0 <= overlap < encoder_window <= max positions - 2")
        if self.feature_chunk_size < 1 or self.d < 1 or self.t < 1:
            raise ValueError("Projection dimensions and feature chunk size must be positive")
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.char_projection = nn.Linear(self.encoder.config.hidden_size, d)
        self.activation = nn.GELU()
        if model_type == "sequence_crf":
            self.num_tags = 4 * len(pos_tags)
            self.sequence_projection = nn.Sequential(nn.Linear(d, d), nn.Dropout(dropout), nn.GELU())
            self.emission = nn.Linear(d, self.num_tags)
            constraints = bmes_constraints(len(pos_tags))
            self.crf = LinearChainCRF(self.num_tags, crf_chunk_size, *constraints)
        else:
            self.num_tags = len(pos_tags) + 1
            self.bilinear = nn.Parameter(torch.empty(t, d, d))
            nn.init.xavier_uniform_(self.bilinear)
            self.grid_feature_projection = nn.Linear(3 * d + t, 4 * d)
            # Formula 7's nested affine notation is resolved as ONE affine map.
            self.grid_reduction = nn.Sequential(nn.Linear(4 * d, d), nn.Dropout(dropout), nn.GELU())
            self.emission = nn.Linear(d, self.num_tags)
            self.crf = LinearChainCRF(self.num_tags, crf_chunk_size) if model_type == "gef" else None
        self.model_config = {"encoder_path": None if encoder_path is None else str(Path(encoder_path)),
                             "model_type": model_type, "d": d, "t": t, "dropout": dropout,
                             "encoder_window": encoder_window, "encoder_overlap": encoder_overlap,
                             "crf_chunk_size": crf_chunk_size, "feature_chunk_size": feature_chunk_size,
                             "gradient_checkpointing": gradient_checkpointing, "pos_tags": list(pos_tags),
                             "encoder_hidden_size": self.encoder.config.hidden_size,
                             "num_tags": self.num_tags,
                             "cls_token_id": self.cls_token_id, "sep_token_id": self.sep_token_id,
                             "flatten_order": "upper_row_major", "null_id": 0,
                             "decoder": ("bmes_constrained_viterbi" if model_type == "sequence_crf"
                                         else "minimum_hamming_valid_word_partition"),
                             "loss_reduction": "token_mean", "formula7_affine_count": 1}

    def encode_characters(self, input_ids):
        if input_ids.ndim != 1 or len(input_ids) < 1:
            raise ValueError("input_ids must be a nonempty 1-D character-ID tensor")
        n, total = len(input_ids), None
        counts = torch.zeros(n, device=input_ids.device, dtype=torch.float32)
        step = self.encoder_window - self.encoder_overlap
        for start in range(0, n, step):
            stop = min(start + self.encoder_window, n)
            chars = input_ids[start:stop]
            ids = torch.cat((chars.new_tensor([self.cls_token_id]), chars,
                             chars.new_tensor([self.sep_token_id])))[None, :]
            hidden = self.encoder(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0, 1:-1]
            indices = torch.arange(start, stop, device=input_ids.device)
            if total is None:
                total = hidden.new_zeros((n, hidden.shape[-1]))
            total = total.index_add(0, indices, hidden)
            counts.index_add_(0, indices, torch.ones(stop - start, device=counts.device))
            if stop == n:
                break
        return self.char_projection(total / counts[:, None].to(total.dtype))

    def _feature_chunk(self, hidden, bilinear_left, context, rows, cols):
        left, right = hidden.index_select(0, rows), hidden.index_select(0, cols)
        relation = torch.bmm(bilinear_left.index_select(0, rows), right[:, :, None]).squeeze(-1)
        features = torch.cat((left, right, context, relation), dim=-1)
        grid = self.activation(self.grid_feature_projection(features))
        return self.emission(self.grid_reduction(grid))

    def forward(self, input_ids):
        hidden = self.encode_characters(input_ids)
        if self.model_type == "sequence_crf":
            return self.emission(self.sequence_projection(hidden))
        n = len(hidden)
        rows, cols = upper_indices(n, hidden.device)
        # Exact interval max-pooling. No [n,n,n,d] tensor is constructed.
        context = torch.cat([torch.cummax(hidden[i:], dim=0).values for i in range(n)], dim=0)
        bilinear_left = torch.einsum("nd,kde->nke", hidden, self.bilinear)
        chunks = []
        for begin in range(0, len(rows), self.feature_chunk_size):
            stop = begin + self.feature_chunk_size
            arguments = (hidden, bilinear_left, context[begin:stop], rows[begin:stop], cols[begin:stop])
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                output = checkpoint(self._feature_chunk, *arguments, use_reentrant=False)
            else:
                output = self._feature_chunk(*arguments)
            chunks.append(output)
        return torch.cat(chunks, dim=0)

    def loss_from_emissions(self, emissions, words):
        if emissions.ndim != 2 or emissions.shape[1] != self.num_tags:
            raise ValueError("Unexpected emissions shape")
        if self.model_type == "sequence_crf":
            gold = sequence_gold(words, len(emissions), self.pos_tags, emissions.device)
        else:
            n = length_from_grid_cells(len(emissions))
            gold = grid_gold(words, n, self.pos_tags, emissions.device)
        if self.crf is not None:
            return self.crf.neg_log_likelihood(emissions.float(), gold, reduction="token_mean")
        return F.cross_entropy(emissions.float(), gold, reduction="mean")

    def loss(self, input_ids, words):
        return self.loss_from_emissions(self(input_ids), words)

    @torch.no_grad()
    def decode_emissions(self, emissions):
        if emissions.ndim != 2 or emissions.shape[1] != self.num_tags:
            raise ValueError("Unexpected emissions shape")
        predictions = self.crf.decode(emissions) if self.crf is not None else emissions.argmax(-1)
        if self.model_type == "sequence_crf":
            return sequence_to_words(predictions, self.pos_tags)
        return project_grid(predictions, length_from_grid_cells(len(emissions)), self.pos_tags)

    @torch.no_grad()
    def decode(self, input_ids):
        return self.decode_emissions(self(input_ids))
