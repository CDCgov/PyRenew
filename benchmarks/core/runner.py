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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.random as random
import numpy as np
import numpyro

from benchmarks.core.models import BuiltFit

RT_SITE_NAMES: tuple[str, ...] = ("PopulationInfections::rt_single",)


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


@dataclass(frozen=True)
class ParameterSummary:
    """Posterior summary for one scalar parameter element.

    Carries the posterior point estimate (``mean``), its spread (``std`` and
    the ``q*`` credible-interval quantiles: 2.5 / 25 / 50 / 75 / 97.5 percent),
    and the per-element convergence diagnostics (``ess``, ``rhat``).
    """

    site: str
    index: str
    mean: float
    ess: float
    rhat: float
    std: float = float("nan")
    q025: float = float("nan")
    q25: float = float("nan")
    q50: float = float("nan")
    q75: float = float("nan")
    q975: float = float("nan")


@dataclass
class FitResult:
    """One row of benchmark output.

    Parameters
    ----------
    candidate
        Display name of the benchmark candidate.
    arm
        Comparison arm this fit belongs to, as declared by the suite's
        :class:`benchmarks.core.comparison.ComparisonSpec`.
    config_fields
        Flat, serializable mapping of the candidate's configuration axes.
        Reporting reads only this view, so candidates with different model
        families can coexist in one result set.
    config
        Optional builder-specific configuration object retained for callers
        that need the original; reporting does not read it.
    """

    candidate: str
    arm: str
    repeat: int
    dataset: str
    settings: McmcSettings
    metrics: FitMetrics
    n_initialization_points: int
    config_fields: dict[str, Any] = field(default_factory=dict)
    config: Any = None
    parameter_summaries: list[ParameterSummary] = field(default_factory=list)


@dataclass(frozen=True)
class Candidate:
    """One benchmark candidate: how to build a model and how to report it.

    A suite assembles a list of candidates and hands them to the runner. The
    ``build`` callable returns a :class:`benchmarks.core.models.BuiltFit`, so a
    candidate may wrap any model the suite can construct: a PyRenew
    ``MultiSignalModel`` or the production HEW model. Model-construction code
    lives in the suite; the runner stays model-agnostic.

    Parameters
    ----------
    name
        Display name, unique within the suite.
    arm
        Comparison arm this candidate belongs to, matching the suite's
        :class:`benchmarks.core.comparison.ComparisonSpec`.
    config_fields
        Flat mapping of the candidate's configuration axes, used by reporting
        to group and label candidates.
    build
        Zero-argument callable returning the assembled ``BuiltFit``. Called
        once per repeat so each fit starts from a fresh model.
    rt_site_names
        Posterior site names to search for the Rt trajectory, in priority
        order.
    """

    name: str
    arm: str
    config_fields: dict[str, Any]
    build: Callable[[], BuiltFit]
    rt_site_names: tuple[str, ...] = RT_SITE_NAMES


def _extract_rt_array(mcmc: Any, rt_site_names: tuple[str, ...]) -> np.ndarray | None:
    """Locate and squeeze the Rt posterior trajectory.

    Returns
    -------
    numpy.ndarray | None
        Rt samples grouped by chain, or ``None`` if no Rt site was sampled.
    """
    samples = mcmc.get_samples(group_by_chain=True)
    for name in rt_site_names:
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


def compute_fit_metrics(
    mcmc: Any, wall_time_s: float, rt_site_names: tuple[str, ...]
) -> FitMetrics:
    """Compute performance and convergence metrics from a completed MCMC fit.

    Returns
    -------
    FitMetrics
        Performance and convergence metrics for the completed fit.
    """
    rt = _extract_rt_array(mcmc, rt_site_names)
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

    extras = mcmc.get_extra_fields(group_by_chain=True)
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


_QUANTILE_LEVELS: tuple[float, ...] = (0.025, 0.25, 0.5, 0.75, 0.975)
"""Credible-interval quantile levels summarized per posterior element."""


def summarize_posterior_parameters(mcmc: Any) -> list[ParameterSummary]:
    """Summarize the posterior of every sampled site.

    For each scalar element of each site, records the posterior mean, standard
    deviation, the credible-interval quantiles in :data:`_QUANTILE_LEVELS`, and
    the ESS and R-hat convergence diagnostics. Statistics reduce over the chain
    and draw axes.

    Returns
    -------
    list[ParameterSummary]
        One row per scalar element of each posterior sample site.
    """
    samples = mcmc.get_samples(group_by_chain=True)
    summaries: list[ParameterSummary] = []
    for site, values in sorted(samples.items()):
        array = np.asarray(values)
        if array.ndim < 2:
            continue
        mean = np.asarray(np.mean(array, axis=(0, 1)))
        std = np.asarray(np.std(array, axis=(0, 1)))
        quantiles = np.asarray(np.quantile(array, _QUANTILE_LEVELS, axis=(0, 1)))
        ess = np.asarray(numpyro.diagnostics.effective_sample_size(array))
        if array.shape[0] < 2:
            rhat = np.full(mean.shape, np.nan)
        else:
            rhat = np.asarray(numpyro.diagnostics.split_gelman_rubin(array))

        flat_quantiles = quantiles.reshape(len(_QUANTILE_LEVELS), -1)
        flat_std = std.reshape(-1)
        flat_ess = ess.reshape(-1)
        flat_rhat = rhat.reshape(-1)
        for flat_index, mean_value in enumerate(mean.reshape(-1)):
            index = _format_sample_index(mean.shape, flat_index)
            q025, q25, q50, q75, q975 = (
                float(flat_quantiles[level, flat_index])
                for level in range(len(_QUANTILE_LEVELS))
            )
            summaries.append(
                ParameterSummary(
                    site=site,
                    index=index,
                    mean=float(mean_value),
                    std=float(flat_std[flat_index]),
                    q025=q025,
                    q25=q25,
                    q50=q50,
                    q75=q75,
                    q975=q975,
                    ess=float(flat_ess[flat_index]),
                    rhat=float(flat_rhat[flat_index]),
                )
            )
    return summaries


def _format_sample_index(shape: tuple[int, ...], flat_index: int) -> str:
    """Format one posterior sample element index.

    Returns
    -------
    str
        Empty string for scalar sites, otherwise a bracketed array index.
    """
    if shape == ():
        return ""
    return "[" + ",".join(str(i) for i in np.unravel_index(flat_index, shape)) + "]"


def fit_and_measure(
    candidate: str,
    built: BuiltFit,
    settings: McmcSettings,
    repeat: int,
    *,
    arm: str,
    config_fields: dict[str, Any],
    rt_site_names: tuple[str, ...] = RT_SITE_NAMES,
    config: Any = None,
) -> FitResult:
    """Fit ``built.model`` and return a :class:`FitResult`.

    The model is any :class:`pyrenew.metaclass.Model` exposing ``run`` and
    ``mcmc``, so the same runner serves PyRenew ``MultiSignalModel`` builds
    and the production HEW model.

    Parameters
    ----------
    candidate
        Display name of the benchmark candidate.
    built
        Assembled model and ``run_kwargs`` from a builder in
        :mod:`benchmarks.core.models`.
    settings
        MCMC controls shared across the suite.
    repeat
        Repeat index. Used to perturb the seed so repeats explore different
        chain trajectories.
    arm
        Comparison arm this candidate belongs to.
    config_fields
        Flat mapping of the candidate's configuration axes, stored for
        reporting.
    rt_site_names
        Posterior site names to search for the Rt trajectory, in priority
        order.
    config
        Optional builder-specific configuration object, stored verbatim.

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

    metrics = compute_fit_metrics(built.model.mcmc, wall_time_s, rt_site_names)
    parameter_summaries = summarize_posterior_parameters(built.model.mcmc)
    result = FitResult(
        candidate=candidate,
        arm=arm,
        repeat=repeat,
        dataset=built.dataset_name,
        settings=settings,
        metrics=metrics,
        n_initialization_points=built.n_initialization_points,
        config_fields=dict(config_fields),
        config=config,
        parameter_summaries=parameter_summaries,
    )
    gc.collect()
    return result


def fit_candidate(
    candidate: Candidate, settings: McmcSettings, repeat: int
) -> FitResult:
    """Build and fit one :class:`Candidate`.

    Returns
    -------
    FitResult
        Per-fit metrics and metadata for this candidate and repeat.
    """
    return fit_and_measure(
        candidate=candidate.name,
        built=candidate.build(),
        settings=settings,
        repeat=repeat,
        arm=candidate.arm,
        config_fields=candidate.config_fields,
        rt_site_names=candidate.rt_site_names,
    )
