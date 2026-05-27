"""Synthetic dataset provider wrapping ``pyrenew/datasets/``.

Each :class:`DatasetBundle` exposed here is paired with one model builder in
:mod:`benchmarks.core.models`. The pairing is implicit: a suite chooses a
model, and the model's builder calls a specific dataset by name.

A real-data provider would implement the same :class:`DatasetProvider`
protocol; suites would not change.
"""

from __future__ import annotations

from datetime import date

import jax.numpy as jnp

from benchmarks.core.signals import (
    DatasetBundle,
    DatasetProvider,
    SignalSeries,
)
from pyrenew.datasets import (
    load_example_infection_admission_interval,
    load_synthetic_daily_ed_visits,
    load_synthetic_true_parameters,
    load_synthetic_weekly_hospital_admissions,
)

GEN_INT_PMF: jnp.ndarray = jnp.array(
    [0.6326975, 0.2327564, 0.0856263, 0.03150015, 0.01158826, 0.00426308, 0.0015683]
)

SYNTHETIC_HE_WEEKLY_HOSPITAL = "synthetic_he_weekly_hospital"


def _build_synthetic_he_weekly_hospital() -> DatasetBundle:  # numpydoc ignore=RT01
    """Build the synthetic H+E bundle with weekly-aggregated hospital admissions."""
    weekly_hosp = load_synthetic_weekly_hospital_admissions()
    daily_ed = load_synthetic_daily_ed_visits()
    true_params = load_synthetic_true_parameters()
    hosp_delay_pmf = jnp.array(
        load_example_infection_admission_interval()["probability_mass"].to_numpy()
    )
    ed_delay_pmf = jnp.array(true_params["ed_visits"]["delay_pmf"])
    ed_dow = jnp.array(true_params["ed_visits"]["day_of_week_effects"])

    obs_start = date(2023, 11, 5)
    hospital = SignalSeries(
        name="hospital",
        values=jnp.array(
            weekly_hosp["weekly_hosp_admits"].to_numpy(), dtype=jnp.float32
        ),
        cadence="weekly",
        start_date=obs_start,
        extras={"delay_pmf": hosp_delay_pmf, "aggregation": "weekly"},
    )
    ed_visits = SignalSeries(
        name="ed_visits",
        values=jnp.array(daily_ed["ed_visits"].to_numpy(), dtype=jnp.float32),
        cadence="daily",
        start_date=obs_start,
        extras={"delay_pmf": ed_delay_pmf, "day_of_week_effects": ed_dow},
    )
    return DatasetBundle(
        name=SYNTHETIC_HE_WEEKLY_HOSPITAL,
        population_size=float(weekly_hosp["pop"][0]),
        obs_start_date=obs_start,
        n_days_post_init=126,
        signals={"hospital": hospital, "ed_visits": ed_visits},
        gen_int_pmf=GEN_INT_PMF,
        fixed_params={"i0_per_capita": true_params["i0_per_capita"]},
    )


_BUILDERS = {
    SYNTHETIC_HE_WEEKLY_HOSPITAL: _build_synthetic_he_weekly_hospital,
}


class SyntheticProvider(DatasetProvider):
    """Provider that wraps the built-in synthetic fixtures in ``pyrenew/datasets/``.

    Bundles are cached on first request so repeated suite candidates do not
    re-read the CSV files.
    """

    def __init__(self) -> None:
        """Create an empty cache."""
        self._cache: dict[str, DatasetBundle] = {}

    def list_datasets(self) -> list[str]:  # numpydoc ignore=RT01
        """Return the dataset names this provider exposes."""
        return list(_BUILDERS)

    def get(self, name: str) -> DatasetBundle:  # numpydoc ignore=RT01
        """Return the named dataset bundle, building and caching on first request."""
        if name not in _BUILDERS:
            raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(_BUILDERS)}")
        if name not in self._cache:
            self._cache[name] = _BUILDERS[name]()
        return self._cache[name]
