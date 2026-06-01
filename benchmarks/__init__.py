"""PyRenew benchmark suites.

Run a suite as a module, for example:

    python -m benchmarks.suites.rt_params --quick

Suites read datasets through :mod:`benchmarks.core.datasets` and build models
through :mod:`benchmarks.core.models`. The signal data interface lives in
:mod:`benchmarks.core.signals` and is the seam where real reporting inputs
can be substituted for the synthetic providers in the future.
"""
