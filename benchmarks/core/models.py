"""Shared model-fit machinery for benchmark suites.

:class:`BuiltFit` packages an assembled model with the keyword arguments
``model.run`` needs; :func:`align_weekly_observations` pads a weekly
observation series onto a model's period grid. Both are model-agnostic and
reused across suites. Suite-specific model construction lives in the suites
themselves, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import jax.numpy as jnp

from pyrenew.metaclass import Model
from pyrenew.model import MultiSignalModel


@dataclass
class BuiltFit:
    """Assembled model plus the kwargs that ``model.run`` needs.

    Parameters
    ----------
    model
        Any :class:`pyrenew.metaclass.Model` exposing ``run`` and ``mcmc``,
        such as a PyRenew :class:`MultiSignalModel` or the production HEW
        model.
    run_kwargs
        Mapping passed as ``**kwargs`` to ``model.run`` after the MCMC
        controls. For the H+E builder this includes ``n_days_post_init``,
        ``population_size``, ``obs_start_date`` and the per-signal
        observation dicts.
    dataset_name
        Identifier of the dataset bundle used.
    n_initialization_points
        Latent initialization points the model requires. When omitted, it is
        read from ``model.latent.n_initialization_points``; builders whose
        model does not expose that attribute must pass it explicitly.
    """

    model: Model
    run_kwargs: dict[str, Any]
    dataset_name: str
    n_initialization_points: int | None = None

    def __post_init__(self) -> None:
        """Default ``n_initialization_points`` from the latent process."""
        if self.n_initialization_points is None:
            self.n_initialization_points = self.model.latent.n_initialization_points


def align_weekly_observations(
    model: MultiSignalModel,
    signal_name: str,
    weekly_values: jnp.ndarray,
    obs_start_date: date,
    n_days_post_init: int,
) -> jnp.ndarray:
    """Pad a weekly observation series with leading NaNs to match the period grid.

    Returns
    -------
    jnp.ndarray
        Dense weekly observations aligned to the model's period grid.
    """
    obs = model.observations[signal_name]
    first_day_dow = model._resolve_first_day_dow(obs_start_date)
    n_total = model.latent.n_initialization_points + n_days_post_init
    offset = obs._compute_period_offset(first_day_dow, obs.start_dow)
    n_periods = (n_total - offset) // obs.aggregation_period
    n_pre = n_periods - len(weekly_values)
    if n_pre < 0:
        raise ValueError(
            f"Weekly observations for {signal_name!r} are longer than the "
            f"model period grid: {len(weekly_values)} > {n_periods}."
        )
    return jnp.concatenate([jnp.full(n_pre, jnp.nan, dtype=jnp.float32), weekly_values])
