"""Single-run and sweep execution.

Import the focused submodules directly (``runner.run_one``, ``runner.run_sweep``);
the package initializer stays import-free so the evaluation and runner layers do
not pull each other in while loading.
"""
