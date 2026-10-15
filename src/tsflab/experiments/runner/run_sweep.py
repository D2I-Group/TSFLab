"""Sweep runner that executes multiple expanded configs."""

from __future__ import annotations

import os
from collections.abc import Iterable

from tsflab.experiments.runner.run_one import run_one
from tsflab.research.rounds import ROUND_ENV, claim_run, finish_run


def run_sweep(configs: Iterable) -> list:
    """Run a list of expanded configs and collect results.

    Parameters
    ----------
    configs : Iterable
        Iterable of LoadedConfig instances.

    Returns
    -------
    list
        List of RunResult objects from each run.
    """
    results = []
    for loaded in configs:
        round_id = os.environ.get(ROUND_ENV)
        number = None
        if round_id:
            number = claim_run(
                round_id,
                {
                    "config": loaded.config_name,
                    "model": loaded.config.model.name,
                    "dataset": loaded.config.dataset.name,
                    "seed": loaded.config.experiment.random_seed,
                    "prediction_length": loaded.config.task.pred_len,
                },
            )
        try:
            result = run_one(loaded.config, loaded.raw, loaded.sweep_keys)
        except Exception as exc:
            if round_id and number is not None:
                finish_run(round_id, number, status="failed", error=str(exc))
            raise
        if round_id and number is not None:
            finish_run(
                round_id,
                number,
                status="passed",
                run_id=result.run_id,
                metrics=result.metrics,
            )
        results.append(result)
    return results
