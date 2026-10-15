"""Local scalar events with optional, failure-isolated TensorBoard/W&B mirrors."""

import json
import math
from pathlib import Path
import time
import warnings


class Tracker:
    def __init__(self, directory, run_id, config=None, options=None, attempt=1):
        from tsflab.experiments.infra.policy import Tracking

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.closed = False
        self.events = self.directory / "events.jsonl"
        self.writer = None
        self.wandb_run = None
        self.options = options if options is not None else Tracking()
        self.run_id = run_id
        self.config = config if config is not None else {}
        self.attempt = attempt
        self.started = False
        self.writer_failed = False
        self.wandb_failed = False

    def _warning(self, backend, error):
        warnings.warn(
            f"{backend} tracking unavailable: {error}; local events are preserved",
            stacklevel=2,
        )

    def start(self, step=0):
        if self.closed:
            raise RuntimeError("tracker is closed")
        if self.started:
            return
        self.started = True
        if self.options.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(
                    str(self.directory / "tensorboard"), purge_step=step
                )
                self.writer.add_text("config", json.dumps(self.config, indent=2), step)
            except Exception as exc:
                self._warning("TensorBoard", exc)
        if self.options.wandb != "disabled":
            try:
                import wandb

                # Offline attempts are separate W&B runs grouped by the local run_id.
                # Offline files cannot provide the service's online resume guarantee.
                self.wandb_run = wandb.init(
                    project=self.options.project,
                    entity=self.options.entity,
                    mode=self.options.wandb,
                    dir=str(self.directory),
                    id=f"{self.run_id}-a{self.attempt}",
                    group=self.run_id,
                    config=self.config,
                    tags=self.options.tags,
                    save_code=False,
                )
            except Exception as exc:
                self._warning("W&B", exc)

    def log(self, values, step):
        self.start(step)
        scalars = {
            key: float(value)
            for key, value in values.items()
            if value is not None and math.isfinite(float(value))
        }
        with self.events.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "attempt": self.attempt,
                        "step": step,
                        "metrics": scalars,
                    }
                )
                + "\n"
            )
        self._log_tensorboard(scalars, step)
        self._log_wandb(scalars, step)

    def _log_tensorboard(self, scalars, step):
        if self.writer and not self.writer_failed:
            try:
                for key, value in scalars.items():
                    self.writer.add_scalar(key, value, step)
                self.writer.flush()
            except Exception as exc:
                self._warning("TensorBoard", exc)
                self.writer_failed = True

    def _log_wandb(self, scalars, step):
        if self.wandb_run and not self.wandb_failed:
            try:
                self.wandb_run.log({**scalars, "epoch": step})
            except Exception as exc:
                self._warning("W&B", exc)
                self.wandb_failed = True

    def figure(self, figure, name, step):
        """Mirror explicitly requested prediction figures, never raw datasets."""
        self.start(step)
        figure.savefig(self.directory / f"{name}.png", dpi=120, bbox_inches="tight")
        if self.writer and not self.writer_failed:
            try:
                self.writer.add_figure(name, figure, step, close=False)
                self.writer.flush()
            except Exception as exc:
                self._warning("TensorBoard", exc)
        if self.wandb_run and not self.wandb_failed:
            try:
                import wandb

                self.wandb_run.log({name: wandb.Image(figure), "epoch": step})
            except Exception as exc:
                self._warning("W&B", exc)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(failed=exc_type is not None)

    def close(self, failed=False):
        if self.closed:
            return
        self.closed = True
        for name, backend in (("TensorBoard", self.writer), ("W&B", self.wandb_run)):
            if backend:
                try:
                    backend.close() if name == "TensorBoard" else backend.finish(
                        exit_code=int(failed)
                    )
                except Exception as exc:
                    self._warning(name, exc)
