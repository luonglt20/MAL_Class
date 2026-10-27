from pathlib import Path

import pytest
import torch

from malware_hybrid.dataset import ReportTensorizer, collate_samples
from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel, RelationalAttentionBlock, TemporalHierarchicalEncoder
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.training import load_checkpoint, save_checkpoint


def block_inputs():
    torch.manual_seed(44)
    return torch.randn(1, 4, 16), torch.tensor([[True, True, True, False]]), torch.zeros(1, 4, 4, 4)


def test_zero_initialized_value_messages_preserve_bias_only_function():
    x, mask, bias = block_inputs()
    old = RelationalAttentionBlock(16, 4, 0.).eval()
    new = RelationalAttentionBlock(16, 4, 0., relation_values=True).eval()
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert missing == ["relation_value_embeddings"] and not unexpected
    features = torch.ones(1, 4, 4, 5)
    with torch.inference_mode():
        expected, _ = old(x, mask, bias)
        actual, _ = new(x, mask, bias, relation_features=features)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_value_messages_can_learn_uniform_known_vs_unknown_relation():
    x, mask, bias = block_inputs()
    block = RelationalAttentionBlock(16, 4, 0., relation_values=True).eval()
    features = torch.zeros(1, 4, 4, 5)
    features[..., 3] = 1.  # Same resource for every valid event pair.
    optimizer = torch.optim.SGD(block.parameters(), lr=.1)
    output, _ = block(x, mask, bias, relation_features=features)
    output[0, 0, 0].backward()
    assert block.relation_value_embeddings.grad.abs().sum() > 0
    optimizer.step()
    with torch.inference_mode():
        known, weights = block(x, mask, bias, relation_features=features)
        unknown, unknown_weights = block(x, mask, bias, relation_features=torch.zeros_like(features))
    torch.testing.assert_close(weights, unknown_weights)
    assert not torch.allclose(known, unknown, atol=1e-6, rtol=1e-6)


def test_masked_keys_cannot_supply_relation_values():
    x, mask, bias = block_inputs()
    block = RelationalAttentionBlock(16, 4, 0., relation_values=True).eval()
    with torch.no_grad():
        block.relation_value_embeddings.normal_()
    empty = torch.zeros(1, 4, 4, 5)
    masked_only = empty.clone()
    masked_only[:, :, 3] = 100.
    with torch.inference_mode():
        before, _ = block(x, mask, bias, relation_features=empty)
        after, _ = block(x, mask, bias, relation_features=masked_only)
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("features", [None, torch.zeros(1, 4, 4, 4)])
def test_value_mode_requires_typed_relation_features(features):
    x, mask, bias = block_inputs()
    with pytest.raises(ValueError, match="relation_features"):
        RelationalAttentionBlock(16, 4, 0., relation_values=True)(x, mask, bias, relation_features=features)


def test_value_encoder_preserves_per_sample_padding_invariance():
    torch.manual_seed(45)
    encoder = TemporalHierarchicalEncoder(32, 16, 4, 1, 0., 4, 2, 16, 16, relation_values=True).eval()
    with torch.no_grad():
        encoder.event_blocks[0].relation_value_embeddings.normal_()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8]])
    ones, zeros = torch.ones_like(ids), torch.zeros_like(ids)
    args = [ids, zeros.float(), ones, zeros.float(), ones, zeros, ones, ones.bool(), ones]
    padded = [torch.nn.functional.pad(value, (0, 6)) for value in args]
    with torch.inference_mode():
        before = encoder(*args, return_attention=False)
        after = encoder(*padded, return_attention=False)
    torch.testing.assert_close(before[0], after[0][:, :7], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(before[2], after[2], atol=1e-6, rtol=1e-6)


def test_family_gradient_and_checkpoint_round_trip(tmp_path):
    report = CapeReportParser().parse_file(Path(__file__).parent / "fixtures/cape_process_injection.json")
    tensorizer = ReportTensorizer.fit([report])
    config = HybridAttentionConfig(
        token_vocab_size=len(tensorizer.token_vocab), edge_vocab_size=len(tensorizer.edge_vocab),
        node_type_vocab_size=len(tensorizer.node_type_vocab), n_families=len(tensorizer.family_vocab),
        behavior_ids=tensorizer.behavior_ids, d_model=16, n_heads=4, static_layers=1,
        temporal_layers=1, graph_layers=1, fusion_layers=1, window_size=4, window_overlap=2,
        dropout=0., modality_dropout=0., temporal_relation_values=True,
    )
    model = HybridAttentionModel(config).eval()
    batch = collate_samples([tensorizer.encode(report)])
    model(batch, return_attention=False)["family_logits"].sum().backward()
    assert model.temporal_encoder.event_blocks[0].relation_value_embeddings.grad.abs().sum() > 0
    checkpoint = tmp_path / "relvalues.pt"
    save_checkpoint(checkpoint, model, tensorizer)
    loaded, _ = load_checkpoint(checkpoint)
    loaded.eval()
    assert loaded.config.temporal_relation_values is True
    with torch.inference_mode():
        fast = model(batch, return_attention=False)
        explained = loaded(batch, return_attention=True)
    for key in ("family_logits", "behavior_logits", "malware_logits"):
        torch.testing.assert_close(fast[key], explained[key], atol=1e-6, rtol=1e-6)


def test_legacy_saved_configs_do_not_enable_untrained_value_parameters():
    config = HybridAttentionConfig.from_saved(dict(
        token_vocab_size=32, edge_vocab_size=8, node_type_vocab_size=8,
        n_families=2, behavior_ids=["T1055"], d_model=16,
    ))
    model = HybridAttentionModel(config)
    assert config.temporal_relation_values is False
    assert not any("relation_value_embeddings" in key for key in model.state_dict())
