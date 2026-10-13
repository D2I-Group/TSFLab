"""Check optimized execution against the original per-channel equations."""
import copy

import pytest
import torch

from tsflab.models.tide.model import Model

pytestmark = pytest.mark.model


def original_forward(model, x, historical, future):
    projected = model.feature_projection(model._covariates(x, historical, future))
    mean = x.mean(1, keepdim=True).detach()
    scale = x.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    normalized = (x - mean) / scale
    outputs = []
    for channel in range(x.shape[-1]):
        encoded = model.encoder_input(torch.cat((normalized[:, :, channel], projected.flatten(1)), -1))
        for block in model.encoder_blocks:
            encoded = block(encoded)
        for block in model.decoder_blocks:
            encoded = block(encoded)
        decoded = model.dense_decoder(encoded).reshape(x.shape[0], model.pred_len, model.decoder_output_dim)
        nonlinear = model.temporal_decoder(torch.cat((decoded, projected[:, model.seq_len :]), -1)).squeeze(-1)
        outputs.append(nonlinear + model.global_residual(normalized[:, :, channel]))
    return torch.stack(outputs, -1) * scale + mean


@pytest.mark.parametrize('training,chunk_size', [(False, 20), (False, 2), (True, 20), (True, 2)])
def test_channel_batching_preserves_original_outputs_and_all_gradients(training, chunk_size):
    torch.manual_seed(73)
    model = Model(seq_len=12, pred_len=7, d_model=8, e_layers=2, d_layers=2,
                  d_ff=16, decoder_output_dim=3, time_feat_dim=6, dropout=0,
                  bias=True, feature_encode_dim=2).double().train(training)
    reference = copy.deepcopy(model)
    model._channel_chunk_size = lambda batch, channels: min(chunk_size, channels)
    inputs = [torch.randn(2, n, c, dtype=torch.double, requires_grad=True)
              for n, c in ((12, 5), (12, 6), (7, 6))]
    originals = [x.detach().clone().requires_grad_() for x in inputs]
    actual = model(inputs[0], inputs[1], x_mark_dec=inputs[2])
    expected = original_forward(reference, *originals)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for actual_input, original_input in zip(inputs, originals):
        torch.testing.assert_close(actual_input.grad, original_input.grad, atol=1e-9, rtol=1e-9)
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, other.grad, atol=1e-9, rtol=1e-9)


def test_checkpoint_recomputation_preserves_dropout_rng_and_parameter_gradients():
    torch.manual_seed(12)
    model = Model(seq_len=12, pred_len=7, d_model=8, e_layers=2, d_layers=1,
                  d_ff=16, decoder_output_dim=3, time_feat_dim=6, dropout=.2,
                  bias=True, feature_encode_dim=2).train()
    model._channel_chunk_size = lambda batch, channels: 2
    x = torch.randn(2, 12, 5)
    state = torch.random.get_rng_state()
    output = model(x)
    after_forward = torch.random.get_rng_state()
    output.square().mean().backward()
    torch.testing.assert_close(torch.random.get_rng_state(), after_forward)
    gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    torch.random.set_rng_state(state)
    repeated = model(x)
    repeated.square().mean().backward()
    torch.testing.assert_close(output, repeated)
    for parameter, gradient in zip(model.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient)


def test_high_channel_long_horizon_chunks_bound_temporal_hidden_activation():
    model = Model(seq_len=12, pred_len=720, d_model=8, e_layers=1, d_layers=1,
                  d_ff=2048, decoder_output_dim=3, time_feat_dim=6, dropout=0,
                  bias=True, feature_encode_dim=2)
    chunk = model._channel_chunk_size(4, 862)
    assert 1 < chunk < 862
    assert 4 * chunk * 720 * 2048 <= 32 * 1024 * 1024
