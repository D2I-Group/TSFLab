"""Paper recipes can use native losses/schedules without custom training loops."""

import copy

import pytest
import torch
from pydantic import ValidationError

from tsflab.catalog.registry.losses import get_loss
from tsflab.experiments.config.schema.training import TrainOptimizerConfig
from tsflab.experiments.infra.comparison import protocol_fingerprint
from tsflab.experiments.utils.training import adjust_learning_rate


def test_huber_piecewise_loss_and_gradient_with_nonunit_delta():
    prediction = torch.tensor([0.0, 1.0, 4.0], requires_grad=True)
    loss = get_loss("huber", delta=2)(prediction, torch.zeros(3))
    torch.testing.assert_close(loss, torch.tensor(6.5 / 3))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([0.0, 1 / 3, 2 / 3]))


def test_cosine_floor_matches_pytorch_schedule_over_complete_training_budget():
    actual = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.0007)
    reference = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.0007)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(reference, T_max=50, eta_min=1e-6)
    for epoch in range(1, 51):
        reference.step()
        scheduler.step()
        adjust_learning_rate(actual, epoch, "cosine_annealing", 0.0007, 50, min_lr=1e-6)
        assert actual.param_groups[0]["lr"] == pytest.approx(reference.param_groups[0]["lr"])
    assert actual.param_groups[0]["lr"] == pytest.approx(1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [dict(lr=0.001, min_lr=0.002, lradj="cosine_annealing"), dict(min_lr=-1), dict(min_lr=1e-6, lradj="constant")],
)
def test_invalid_cosine_floor_is_rejected_before_compute(kwargs):
    with pytest.raises(ValidationError):
        TrainOptimizerConfig(**kwargs)


def test_zero_floor_reuses_historical_protocol_without_mutating_saved_snapshot():
    historical = {"task": {}, "evaluation": {}, "training": {"optimizer": {"lr": 0.001, "lradj": "cosine_annealing"}}}
    current = copy.deepcopy(historical)
    current["training"]["optimizer"]["min_lr"] = 0.0
    original = copy.deepcopy(current)
    assert protocol_fingerprint(current, {}) == protocol_fingerprint(historical, {})
    assert current == original
    current["training"]["optimizer"]["min_lr"] = 1e-6
    assert protocol_fingerprint(current, {}) != protocol_fingerprint(historical, {})
