"""Per-fit MCMC performance and convergence metrics.

All quantities are computed from ``numpyro.diagnostics`` and the raw
``extra_fields`` returned by ``mcmc.run``, so the module does not import
ArviZ.

The headline metric is ESS per second on the Rt trajectory: median across
timepoints summarizes typical mixing, and minimum captures the worst
timepoint that limits downstream inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
import numpyro

from pyrenew.model import MultiSignalModel

RT_SITE_NAMES: tuple[str, ...] = (
    "PopulationInfections::rt_single",
    "SubpopulationInfections::rt_baseline",
)


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


def _extract_rt_array(model: MultiSignalModel) -> np.ndarray | None:
    """Locate and squeeze the Rt posterior trajectory.

    Returns
    -------
    np.ndarray | None
        Shape ``(chains, draws, time)`` or ``None`` if no Rt site is present.
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

    Parameters
    ----------
    energy
        Energy values of shape ``(chains, draws)``.

    Returns
    -------
    np.ndarray
        E-BFMI for each chain.
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
        Maximum split R-hat, or ``nan`` when the diagnostic cannot be computed.
    """
    if rt.shape[0] < 2:
        return float("nan")
    values = np.asarray(numpyro.diagnostics.split_gelman_rubin(rt)).flatten()
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def compute_fit_metrics(model: MultiSignalModel, wall_time_s: float) -> FitMetrics:
    """Compute :class:`FitMetrics` from a completed MCMC fit.

    Parameters
    ----------
    model
        Model whose ``mcmc`` attribute has just run with
        ``extra_fields=("diverging", "num_steps", "energy")``.
    wall_time_s
        Elapsed wall time, ideally measured around a
        ``jax.block_until_ready`` on the samples.

    Returns
    -------
    FitMetrics
        Performance and convergence summary.
    """
    rt = _extract_rt_array(model)
    if rt is None:
        ess_median = float("nan")
        ess_min = float("nan")
        rhat_max = float("nan")
    else:
        ess_values = np.asarray(
            numpyro.diagnostics.effective_sample_size(rt)
        ).flatten()
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
