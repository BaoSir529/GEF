import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import BertConfig, BertModel

from gef.batching import (BATCHING_PROTOCOL_VERSION, encode_characters_batch,
                          forward_batch, loss_batch, source_sha256)
from gef.model import GEFModel


class MaskAwareEncoder(nn.Module):
    """Masked global context ensures ignoring the padding mask fails these tests."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, max_position_embeddings=10,
                                      pad_token_id=0, cls_token_id=1, sep_token_id=2)
        self.embedding = nn.Embedding(32, 4)
        self.calls = []
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(128).reshape(32, 4) / 100)
            self.embedding.weight[0].fill_(1000.)

    def forward(self, input_ids, attention_mask):
        self.calls.append((input_ids.detach().clone(), attention_mask.detach().clone()))
        values = self.embedding(input_ids)
        mask = attention_mask[:, :, None].to(values.dtype)
        context = (values * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :, None] / 10
        return SimpleNamespace(last_hidden_state=values + positions + context)


def masked_model(model_type="grid_softmax"):
    return GEFModel(None, ("n", "v"), model_type=model_type, d=4, t=2, dropout=0.,
                    encoder=MaskAwareEncoder(), encoder_window=6, encoder_overlap=2,
                    feature_chunk_size=8, crf_chunk_size=8, gradient_checkpointing=False)


def tiny_model(model_type="gef", checkpointing=False):
    encoder = BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=32,
                                  max_position_embeddings=10, pad_token_id=0,
                                  hidden_dropout_prob=0., attention_probs_dropout_prob=0.),
                        add_pooling_layer=False)
    return GEFModel(None, ("n", "v"), model_type=model_type, d=8, t=3, dropout=0.,
                    encoder_window=6, encoder_overlap=2, crf_chunk_size=4,
                    feature_chunk_size=8, gradient_checkpointing=checkpointing,
                    encoder=encoder, cls_token_id=1, sep_token_id=2)


def gold_words(n):
    if n == 1:
        return [{"start": 0, "end": 1, "pos": "n"}]
    return [{"start": 0, "end": min(2, n - 1), "pos": "n"},
            {"start": min(2, n - 1), "end": n, "pos": "v"}]


def compare_gradients(first, second, atol=3e-5, rtol=3e-4):
    for (name, expected), (other_name, actual) in zip(first.named_parameters(), second.named_parameters()):
        assert name == other_name
        assert expected.grad is not None and actual.grad is not None, name
        assert torch.isfinite(actual.grad).all(), name
        assert torch.allclose(expected.grad, actual.grad, atol=atol, rtol=rtol), name


def test_short_sentences_share_one_encoder_call_with_exact_padding_mask():
    model = masked_model().eval()
    model.char_projection = nn.Identity()
    inputs = [torch.tensor([3, 4]), torch.tensor([5, 6, 7, 8]), torch.tensor([9])]
    expected = [model.encode_characters(ids) for ids in inputs]
    model.encoder.calls.clear()
    actual = encode_characters_batch(model, inputs)
    assert len(model.encoder.calls) == 1
    ids, mask = model.encoder.calls[0]
    assert ids.tolist() == [[1, 3, 4, 2, 0, 0], [1, 5, 6, 7, 8, 2], [1, 9, 2, 0, 0, 0]]
    assert mask.tolist() == [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]
    for a, b in zip(actual, expected):
        assert a.shape == b.shape
        assert torch.allclose(a, b, atol=1e-6)


def test_long_inputs_keep_window_queue_order_overlap_mean_and_final_characters():
    model = masked_model().eval()
    model.char_projection = nn.Identity()
    inputs = [torch.arange(3, 16), torch.tensor([16, 17, 18])]
    expected = [model.encode_characters(ids) for ids in inputs]
    model.encoder.calls.clear()
    actual = encode_characters_batch(model, inputs, encoder_batch_size=2)
    assert len(model.encoder.calls) == 2
    queued = []
    for ids, mask in model.encoder.calls:
        queued.extend(row[valid.bool()].tolist() for row, valid in zip(ids, mask))
    assert queued == [[1, 3, 4, 5, 6, 7, 8, 2], [1, 7, 8, 9, 10, 11, 12, 2],
                      [1, 11, 12, 13, 14, 15, 2], [1, 16, 17, 18, 2]]
    assert [len(value) for value in actual] == [13, 3]
    for a, b in zip(actual, expected):
        assert torch.allclose(a, b, atol=1e-6)
    sum(value.sum() for value in actual).backward()
    assert model.encoder.embedding.weight.grad[15].abs().sum() > 0
    assert model.encoder.embedding.weight.grad[18].abs().sum() > 0
    assert model.encoder.embedding.weight.grad[0].abs().sum() == 0


@pytest.mark.parametrize("model_type", ["gef", "grid_softmax", "sequence_crf"])
def test_three_architectures_eval_outputs_and_decoding_match_sequential(model_type):
    torch.manual_seed(113)
    model = tiny_model(model_type).eval()
    inputs = [torch.tensor([3]), torch.arange(4, 9), torch.arange(10, 19)]
    expected = [model(ids) for ids in inputs]
    actual = forward_batch(model, inputs, encoder_batch_size=3)
    for ids, a, b in zip(inputs, actual, expected):
        rows = len(ids) if model_type == "sequence_crf" else len(ids) * (len(ids) + 1) // 2
        assert a.shape == (rows, model.num_tags)
        assert torch.allclose(a, b, atol=2e-6, rtol=2e-5)
        assert model.decode_emissions(a) == model.decode_emissions(b)


@pytest.mark.parametrize("model_type", ["gef", "grid_softmax", "sequence_crf"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_all_parameter_gradients_and_per_sample_mean_match_reference(model_type, checkpointing):
    torch.manual_seed(127)
    sequential = tiny_model(model_type, checkpointing=checkpointing).train()
    batched = copy.deepcopy(sequential)
    inputs = [torch.tensor([3, 4]), torch.arange(5, 14), torch.tensor([14, 15, 16, 17])]
    words = [gold_words(len(ids)) for ids in inputs]
    first = torch.stack([sequential.loss(ids, gold) for ids, gold in zip(inputs, words)]).mean()
    second = loss_batch(batched, inputs, words, encoder_batch_size=2)
    assert torch.allclose(first, second, atol=2e-6, rtol=2e-5)
    first.backward()
    second.backward()
    compare_gradients(sequential, batched)
    assert batched.encoder.embeddings.word_embeddings.weight.grad[13].abs().sum() > 0


@pytest.mark.parametrize("microbatch", [2, 4, 16])
def test_effective_16_sample_gradient_accumulation_is_not_padding_weighted(microbatch):
    torch.manual_seed(139)
    sequential = masked_model().train()
    batched = copy.deepcopy(sequential)
    inputs = [torch.arange(3, 4 + i % 5) for i in range(16)]
    words = [gold_words(len(ids)) for ids in inputs]
    for ids, gold in zip(inputs, words):
        (sequential.loss(ids, gold) / len(inputs)).backward()
    for start in range(0, len(inputs), microbatch):
        outputs = forward_batch(batched, inputs[start:start + microbatch])
        # Shared batched encoder graph: backward ONCE per microbatch.
        group_loss = sum(batched.loss_from_emissions(output, gold) / len(inputs)
                         for output, gold in zip(outputs, words[start:start + microbatch]))
        group_loss.backward()
    compare_gradients(sequential, batched, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("inputs", [[], [torch.tensor([], dtype=torch.long)],
                                   [torch.tensor([[3, 4]])], [torch.tensor([3.])],
                                   [torch.tensor([3]), torch.ones(2, dtype=torch.long, device="meta")]])
def test_invalid_inputs_are_rejected_before_encoder_call(inputs):
    model = masked_model()
    with pytest.raises(ValueError):
        forward_batch(model, inputs)
    assert model.encoder.calls == []


@pytest.mark.parametrize("batch_size", [0, -1, 1.5])
def test_invalid_encoder_batch_size_is_rejected(batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        forward_batch(masked_model(), [torch.tensor([3])], encoder_batch_size=batch_size)


def test_loss_requires_one_gold_partition_per_original_sample():
    with pytest.raises(ValueError, match="same length"):
        loss_batch(masked_model(), [torch.tensor([3])], [])


def test_independent_batching_source_has_stable_protocol_fingerprint():
    import gef.batching
    assert BATCHING_PROTOCOL_VERSION == 1
    assert source_sha256() == hashlib.sha256(Path(gef.batching.__file__).read_bytes()).hexdigest()
