"""Run one MCMC fit and collect metrics.

The runner is a thin wrapper around ``model.run`` that:

- requests the extra fields needed for diagnostics,
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
import numpy as np
import numpyro

from benchmarks.core.models import BuildConfig, BuiltFit
from pyrenew.model import MultiSignalModel

RT_SITE_NAMES: tuple[str, ...] = (
    "PopulationInfections::rt_single",
    "SubpopulationInfections::rt_baseline",
)


@dataclass(frozen=True)
class McmcSettings:
    """NUTS sampler configuration shared across candidates in a suite."""

    num_warmup: int
    num_samples: int
    num_chains: int
    seed: int
    progress_bar: bool = False


@dataclass
class FitMetrics:
    """Performance and convergence summary for one MCMC fit."""

    wall_time_s: float
    ess_per_sec_rt_median: float
    ess_per_sec_rt_min: float
    divergences: int
    tree_depth_mean: float
    tree_depth_max: int
    ebfmi_min: float
    rhat_rt_max: float


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


def _extract_rt_array(model: MultiSignalModel) -> np.ndarray | None:
    """Locate and squeeze the Rt posterior trajectory.

    Returns
    -------
    numpy.ndarray | None
        Rt samples grouped by chain, or ``None`` if no Rt site was sampled.
    """
    samples = model.mcmc.get_samples(group_by_chain=True)
    for name in RT_SITE_NAMES:
        if name not in samples:
            continue
        rt = np.asarray(samples[name])
        while rt.ndim > 3:
            rt = rt.squeeze(-1)
        return rt
    return None


def _ebfmi_per_chain(energy: np.ndarray) -> np.ndarray:
    """Compute the energy Bayesian fraction of missing information per chain.

    Returns
    -------
    numpy.ndarray
        E-BFMI value for each chain.
    """
    n_per_chain = energy.shape[1]
    return np.sum(np.diff(energy, axis=1) ** 2, axis=1) / (
        np.var(energy, axis=1) * n_per_chain
    )


def _rhat_max(rt: np.ndarray) -> float:
    """Compute the maximum split R-hat across timepoints of the Rt trajectory.

    Returns
    -------
    float
        Maximum finite split R-hat, or NaN when it cannot be computed.
    """
    if rt.shape[0] < 2:
        return float("nan")
    values = np.asarray(numpyro.diagnostics.split_gelman_rubin(rt)).flatten()
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def compute_fit_metrics(model: MultiSignalModel, wall_time_s: float) -> FitMetrics:
    """Compute performance and convergence metrics from a completed MCMC fit.

    Returns
    -------
    FitMetrics
        Performance and convergence metrics for the completed fit.
    """
    rt = _extract_rt_array(model)
    if rt is None:
        ess_median = float("nan")
        ess_min = float("nan")
        rhat_max = float("nan")
    else:
        ess_values = np.asarray(numpyro.diagnostics.effective_sample_size(rt)).flatten()
        finite_ess = ess_values[np.isfinite(ess_values)]
        ess_median = float(np.median(finite_ess)) if finite_ess.size else float("nan")
        ess_min = float(np.min(finite_ess)) if finite_ess.size else float("nan")
        rhat_max = _rhat_max(rt)

    extras = model.mcmc.get_extra_fields(group_by_chain=True)
    jax.block_until_ready(extras)
    divergences = int(np.sum(np.asarray(extras["diverging"])))
    num_steps = np.asarray(extras["num_steps"]).flatten()
    tree_depth = np.log2(num_steps + 1)
    energy = np.asarray(extras["energy"])
    bfmi = _ebfmi_per_chain(energy)

    elapsed = wall_time_s if wall_time_s > 0 else float("nan")
    return FitMetrics(
        wall_time_s=wall_time_s,
        ess_per_sec_rt_median=ess_median / elapsed,
        ess_per_sec_rt_min=ess_min / elapsed,
        divergences=divergences,
        tree_depth_mean=float(np.mean(tree_depth)),
        tree_depth_max=int(np.max(tree_depth)),
        ebfmi_min=float(np.min(bfmi)),
        rhat_rt_max=rhat_max,
    )


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
