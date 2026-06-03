"""Signal and dataset interface for benchmark suites.

The interface decouples benchmark suites from where the data comes from.
A suite asks a :class:`DatasetProvider` for a named bundle. The synthetic
provider in :mod:`benchmarks.core.datasets` wraps the fixtures in
``pyrenew/datasets/``. A future provider can wrap CDC reporting inputs
without any change to the suites or the model builders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal, Protocol

import jax.numpy as jnp

Cadence = Literal["daily", "weekly"]


@dataclass(frozen=True)
class SignalSeries:
    """One observed time series for one signal.

    Parameters
    ----------
    name
        Identifier used as the observation key in a PyRenew model.
    values
        Observation values aligned to ``start_date`` at the given ``cadence``.
        Use ``jnp.nan`` for missing periods.
    cadence
        ``"daily"`` or ``"weekly"``.
    start_date
        Calendar date of ``values[0]``. Must lie in the model's post-init window
        unless ``times`` is provided.
    times
        Integer time indices into the model grid. Provide for irregular signals
        such as wastewater. Leave ``None`` for regular signals.
    subpop_indices
        Subpopulation index per observation. Required by signals that read
        per-subpopulation infections, such as wastewater.
    sensor_indices
        Sensor identifier per observation. Required by signals that have a
        sensor-level random effect, such as wastewater.
    extras
        Free-form per-signal metadata that downstream model builders may
        consume (delay PMFs, day-of-week effects, shedding kinetics, ...).
    """

    name: str
    values: jnp.ndarray
    cadence: Cadence
    start_date: date
    times: jnp.ndarray | None = None
    subpop_indices: jnp.ndarray | None = None
    sensor_indices: jnp.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def end_date(self) -> date:  # numpydoc ignore=RT01
        """Calendar date of the last observation.

        For regular signals the date is stepped from ``start_date`` by the
        cadence; for irregular signals it is offset by the maximum ``times``
        index.
        """
        if self.times is None:
            step_days = 7 if self.cadence == "weekly" else 1
            return self.start_date + timedelta(days=(len(self.values) - 1) * step_days)
        return self.start_date + timedelta(days=int(jnp.max(self.times)))


@dataclass(frozen=True)
class DatasetBundle:
    """All inputs needed to fit one model on one dataset.

    Parameters
    ----------
    name
        Unique identifier reported in benchmark output.
    population_size
        Total population used by the renewal process.
    obs_start_date
        Calendar date corresponding to the first day of the post-init window.
    n_days_post_init
        Number of days fit beyond the latent initialization window.
    signals
        Mapping from signal name to :class:`SignalSeries`.
    gen_int_pmf
        Generation interval PMF used by the latent process.
    fixed_params
        Free-form mapping of additional fixed parameters that model builders
        may need (e.g. true initial prevalence, subpopulation fractions).
    """

    name: str
    population_size: float
    obs_start_date: date
    n_days_post_init: int
    signals: dict[str, SignalSeries]
    gen_int_pmf: jnp.ndarray
    fixed_params: dict[str, Any] = field(default_factory=dict)


class DatasetProvider(Protocol):
    """Source of :class:`DatasetBundle` objects.

    Implementations may wrap built-in fixtures, CSV files, parquet files,
    or remote reporting systems. The benchmark suites only see this protocol.
    """

    def list_datasets(self) -> list[str]:
        """Return the names of datasets this provider exposes."""

    def get(self, name: str) -> DatasetBundle:
        """Return the named dataset bundle."""
