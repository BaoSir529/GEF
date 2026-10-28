from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import BertConfig, BertModel

from gef.model import GEFModel


def tiny_model(model_type="gef", checkpointing=False):
    encoder = BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=32,
                                  max_position_embeddings=10, hidden_dropout_prob=0.,
                                  attention_probs_dropout_prob=0.), add_pooling_layer=False)
    return GEFModel(None, ("n", "v"), model_type=model_type, d=8, t=3, dropout=0.,
                    encoder_window=6, encoder_overlap=2, crf_chunk_size=3,
                    feature_chunk_size=4, gradient_checkpointing=checkpointing,
                    encoder=encoder, cls_token_id=1, sep_token_id=2)


@pytest.mark.parametrize("model_type", ["gef", "grid_softmax", "sequence_crf"])
def test_offline_tiny_model_loss_backward_and_full_coverage(model_type):
    torch.manual_seed(31)
    model = tiny_model(model_type)
    ids = torch.arange(3, 12, dtype=torch.long)  # Longer than one encoder window.
    words = [{"start": 0, "end": 2, "pos": "n"},
             {"start": 2, "end": 4, "pos": "v"},
             {"start": 4, "end": 9, "pos": "n"}]
    emissions = model(ids)
    assert emissions.shape == ((9, 8) if model_type == "sequence_crf" else (45, 3))
    loss = model.loss_from_emissions(emissions, words)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.encoder.embeddings.word_embeddings.weight.grad is not None
    assert torch.isfinite(model.encoder.embeddings.word_embeddings.weight.grad).all()
    result = model.decode_emissions(emissions)
    assert result["words"][0]["start"] == 0
    assert result["words"][-1]["end"] == len(ids)
    assert all(a["end"] == b["start"] for a, b in zip(result["words"], result["words"][1:]))


@pytest.mark.parametrize("dropout", [0., .2])
def test_feature_checkpointing_preserves_outputs_and_gradients(dropout):
    torch.manual_seed(37)
    first = tiny_model("grid_softmax", checkpointing=False)
    second = tiny_model("grid_softmax", checkpointing=True)
    first.grid_reduction[1].p = second.grid_reduction[1].p = dropout
    second.load_state_dict(first.state_dict())
    ids = torch.tensor([3, 4, 5, 6])
    words = [{"start": 0, "end": 4, "pos": "n"}]
    torch.manual_seed(41)
    a = first(ids)
    torch.manual_seed(41)
    b = second(ids)
    assert torch.allclose(a, b, atol=1e-6)
    first.loss_from_emissions(a, words).backward()
    second.loss_from_emissions(b, words).backward()
    for (na, pa), (nb, pb) in zip(first.named_parameters(), second.named_parameters()):
        assert na == nb
        if pa.grad is not None:
            assert torch.allclose(pa.grad, pb.grad, atol=2e-5), na


def test_grid_features_match_explicit_interval_and_bilinear_calculation():
    model = tiny_model("grid_softmax").eval()
    hidden = torch.randn(3, 8)
    rows, cols = torch.triu_indices(3, 3)
    context = torch.cat([torch.cummax(hidden[i:], dim=0).values for i in range(3)])
    left_projection = torch.einsum("nd,kde->nke", hidden, model.bilinear)
    actual = model._feature_chunk(hidden, left_projection, context, rows, cols)
    explicit = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        interval = hidden[i:j + 1].max(dim=0).values
        relation = torch.stack([hidden[i] @ matrix @ hidden[j] for matrix in model.bilinear])
        joined = torch.cat((hidden[i], hidden[j], interval, relation))
        grid = model.activation(model.grid_feature_projection(joined))
        explicit.append(model.emission(model.grid_reduction(grid)))
    assert torch.allclose(actual, torch.stack(explicit), atol=1e-6)


class PositionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, max_position_embeddings=10,
                                      cls_token_id=1, sep_token_id=2)
        self.embedding = nn.Embedding(32, 4)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(128).reshape(32, 4) / 100)

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        offsets = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :, None] / 10
        return SimpleNamespace(last_hidden_state=hidden + offsets)


def test_overlapping_encoder_windows_are_averaged_without_truncation():
    encoder = PositionEncoder()
    model = GEFModel(None, ("n",), d=4, t=2, encoder=encoder,
                    encoder_window=6, encoder_overlap=2, gradient_checkpointing=False)
    model.char_projection = nn.Identity()
    ids = torch.arange(3, 12)
    actual = model.encode_characters(ids)
    expected_position = torch.tensor([1, 2, 3, 4, 3, 4, 3, 4, 5])[:, None] / 10
    assert torch.allclose(actual, encoder.embedding(ids) + expected_position, atol=1e-7)


def test_automodel_loads_a_local_checkpoint_without_network(tmp_path):
    source = BertModel(tiny_model().encoder.config, add_pooling_layer=False)
    source.save_pretrained(tmp_path)
    model = GEFModel(tmp_path, ("n",), d=4, t=2, encoder_window=6, encoder_overlap=2,
                    gradient_checkpointing=False, cls_token_id=1, sep_token_id=2)
    assert model.encoder.pooler is None
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, model.encoder.state_dict()[name]), name
    assert model(torch.tensor([3, 4])).shape == (3, 2)


@pytest.mark.parametrize("model_type", ["gef", "grid_softmax", "sequence_crf"])
def test_state_dict_roundtrip_reproduces_predictions(tmp_path, model_type):
    torch.manual_seed(53)
    model = tiny_model(model_type).eval()
    assert model.encoder.pooler is None
    ids = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11])
    before = model(ids)
    result = model.decode_emissions(before)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    restored = tiny_model(model_type).eval()
    restored.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    after = restored(ids)
    assert torch.equal(before, after)
    assert result == restored.decode_emissions(after)


def test_all_trainable_nonpooler_parameters_receive_finite_gradients():
    torch.manual_seed(59)
    model = tiny_model("gef", checkpointing=True)
    ids = torch.arange(3, 12)
    words = [{"start": 0, "end": 2, "pos": "n"},
             {"start": 2, "end": 9, "pos": "v"}]
    model.loss(ids, words).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    # The final original character is present in the differentiable objective.
    assert model.encoder.embeddings.word_embeddings.weight.grad[11].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_full_gef_gradients_match_sequential_crf_reference():
    torch.manual_seed(71)
    reference = tiny_model("gef", checkpointing=True).cuda()
    optimized = tiny_model("gef", checkpointing=True).cuda()
    optimized.load_state_dict(reference.state_dict())
    reference.crf.log_partition = reference.crf.log_partition_reference
    ids = torch.arange(3, 12, device="cuda")
    words = [{"start": 0, "end": 2, "pos": "n"},
             {"start": 2, "end": 4, "pos": "v"},
             {"start": 4, "end": 9, "pos": "n"}]
    first, second = reference.loss(ids, words), optimized.loss(ids, words)
    assert torch.allclose(first, second, atol=3e-6, rtol=3e-6)
    first.backward()
    second.backward()
    for (name, expected), (other_name, actual) in zip(reference.named_parameters(),
                                                     optimized.named_parameters()):
        assert name == other_name
        assert expected.grad is not None and actual.grad is not None, name
        assert torch.isfinite(actual.grad).all(), name
        assert torch.allclose(expected.grad, actual.grad, atol=2e-5, rtol=2e-4), name
