"""Archived-default regression and full-network structural search contracts."""

import pytest
import torch

from tsflab.models.stateflow.model import Model, StateFlowCore
from tsflab.models.stateflow.spec import ModelParameterConfig

pytestmark = pytest.mark.model


def test_archived_default_output_regression():
    # Frozen v9 archive 0ef31d9b...; default architecture must retain its predictions.
    torch.manual_seed(123)
    model = StateFlowCore(8, 3)
    values = torch.tensor([[1.0, 2.0, 4.0, 3.0, 7.0, 6.0, 9.0, 8.0], [-2.0, 3.0, -1.0, 5.0, 0.0, 2.0, 1.0, 4.0]])
    expected = torch.tensor([[5.1627731323, 5.1769752502, 5.1221609116], [1.6967325211, 1.7736828327, 1.6622543335]])
    torch.testing.assert_close(model(values), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "factor,value", [("hidden_size", 64), ("memory_size", 32), ("chunk_width", 7), ("chunk_stride", 4)]
)
def test_structural_variant_retains_two_stage_training(factor, value):
    parameters = ModelParameterConfig(enc_in=3, encoder_epochs=1, **{factor: value}).model_dump()
    parameters.pop("instance_norm")
    model = Model(16, 5, **parameters)
    values = torch.randn(2, 16, 3)
    seen = []

    def stage_runner(stage, loader, device, optimizer, loss, epochs):
        assert epochs == 1
        optimizer.zero_grad()
        loss(stage(values), values).backward()
        optimizer.step()
        seen.append(True)

    model.pretrain([], torch.device("cpu"), stage_runner=stage_runner)
    encoder = [p for module in (model.core.hidden, model.core.base, model.core.memory) for p in module.parameters()]
    frozen = [p.detach().clone() for p in encoder]
    assert all(not p.requires_grad for p in encoder)
    output = model(values)
    assert output.shape == (2, 5, 3) and torch.isfinite(output).all()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.001)
    optimizer.zero_grad()
    output.square().mean().backward()
    optimizer.step()
    assert model.core.head.weight.grad is not None
    assert all(torch.equal(a, b) for a, b in zip(frozen, encoder))
    model.pretrain([], torch.device("cpu"), stage_runner=stage_runner)
    assert len(seen) == 1
