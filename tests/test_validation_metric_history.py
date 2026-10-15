"""Native histories expose forecast metrics without a second validation pass."""

from types import SimpleNamespace

import pytest
import torch

from tsflab.experiments.runner import trainer


def wire_forecaster(monkeypatch):
    monkeypatch.setattr(trainer, "make_decoder_input", lambda *args: None)
    monkeypatch.setattr(trainer, "call_forecaster", lambda model, x, *args: model(x))
    monkeypatch.setattr(trainer, "slice_prediction_target", lambda x, y, *args: (x, y))


class CounterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + self.offset


def batches():
    return [
        (torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]), torch.zeros(2, 1, 2), None, None),
        (torch.tensor([[[5.0, 6.0]]]), torch.zeros(1, 1, 2), None, None),
    ]


def test_huber_selection_and_element_metrics_keep_partial_batch(monkeypatch):
    wire_forecaster(monkeypatch)
    model = CounterModel()
    metrics = {}
    loss = trainer.validate(
        model, batches(), torch.device("cpu"), torch.nn.HuberLoss(delta=1), 0, 1, "M", metrics=metrics
    )
    assert loss == pytest.approx(3.0)
    assert metrics == pytest.approx({"mse": 91 / 6, "mae": 3.5})
    assert model.calls == 2


def test_training_tracker_records_metrics_without_changing_selection(monkeypatch, tmp_path):
    wire_forecaster(monkeypatch)
    model = CounterModel()
    logged = []
    tracker = SimpleNamespace(start=lambda epoch: None, log=lambda values, step: logged.append((values, step)))
    trainer.train(
        model,
        batches()[:1],
        batches(),
        torch.device("cpu"),
        epochs=1,
        patience=2,
        loss_name="huber",
        loss_params={"delta": 1},
        optimizer=torch.optim.SGD(model.parameters(), lr=0),
        lradj="constant",
        base_lr=0,
        total_epochs=1,
        label_len=0,
        pred_len=1,
        features="M",
        use_amp=False,
        checkpoint_dir=str(tmp_path),
        checkpoint_cfg=SimpleNamespace(strategy="best", save_k=0),
        tracker=tracker,
    )
    values, step = logged[0]
    assert step == 1
    assert values["validation/loss"] == pytest.approx(3.0)
    assert values["validation/mse"] == pytest.approx(91 / 6)
    assert values["validation/mae"] == pytest.approx(3.5)
    assert model.calls == 3
