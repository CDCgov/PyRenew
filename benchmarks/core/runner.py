"""Run one MCMC fit and collect metrics.

The runner is a thin wrapper around ``model.run`` that:

- requests the extra fields needed by :mod:`benchmarks.core.metrics`,
- forces a ``jax.block_until_ready`` so wall time covers the full kernel
  execution (otherwise ``mcmc.run`` returns when work is dispatched),
- packages the result as a :class:`FitResult` row suitable for reporting.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import jax
import jax.random as random

from benchmarks.core.metrics import FitMetrics, compute_fit_metrics
from benchmarks.core.models import BuildConfig, BuiltFit


@dataclass(frozen=True)
class McmcSettings:
    """NUTS sampler configuration shared across candidates in a suite."""

    num_warmup: int
    num_samples: int
    num_chains: int
    seed: int
    progress_bar: bool = False


@dataclass
class FitResult:
    """One row of benchmark output."""

    candidate: str
    repeat: int
    dataset: str
    config: BuildConfig
    settings: McmcSettings
    metrics: FitMetrics
    n_initialization_points: int


def fit_and_measure(
    candidate: str,
    built: BuiltFit,
    config: BuildConfig,
    settings: McmcSettings,
    repeat: int,
) -> FitResult:
    """Fit ``built.model`` and return a :class:`FitResult`.

    Parameters
    ----------
    candidate
        Display name of the benchmark candidate.
    built
        Assembled model and ``run_kwargs`` from a builder in
        :mod:`benchmarks.core.models`.
    config
        Configuration used to build the model. Stored on the result.
    settings
        MCMC controls shared across the suite.
    repeat
        Repeat index. Used to perturb the seed so repeats explore different
        chain trajectories.

    Returns
    -------
    FitResult
        Per-fit metrics and metadata.
    """
    jax.clear_caches()
    rng_key = random.PRNGKey(settings.seed + repeat)
    start = time.perf_counter()
    built.model.run(
        num_warmup=settings.num_warmup,
        num_samples=settings.num_samples,
        rng_key=rng_key,
        mcmc_args={
            "num_chains": settings.num_chains,
            "progress_bar": settings.progress_bar,
        },
        extra_fields=("diverging", "num_steps", "energy"),
        **built.run_kwargs,
    )
    samples = built.model.mcmc.get_samples()
    jax.block_until_ready(samples)
    wall_time_s = time.perf_counter() - start

    metrics = compute_fit_metrics(built.model, wall_time_s)
    result = FitResult(
        candidate=candidate,
        repeat=repeat,
        dataset=built.dataset_name,
        config=config,
        settings=settings,
        metrics=metrics,
        n_initialization_points=built.n_initialization_points,
    )
    gc.collect()
    return result
