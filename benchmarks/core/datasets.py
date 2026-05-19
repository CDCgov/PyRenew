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
    load_hospital_data_for_state,
    load_synthetic_daily_ed_visits,
    load_synthetic_daily_hospital_admissions,
    load_synthetic_true_parameters,
    load_synthetic_weekly_hospital_admissions,
    load_wastewater_data_for_state,
)

GEN_INT_PMF: jnp.ndarray = jnp.array(
    [0.6326975, 0.2327564, 0.0856263, 0.03150015, 0.01158826, 0.00426308, 0.0015683]
)

SUBPOP_GEN_INT_PMF: jnp.ndarray = jnp.array([0.16, 0.32, 0.25, 0.14, 0.07, 0.04, 0.02])

SHEDDING_PMF: jnp.ndarray = (lambda raw: jnp.asarray(raw) / jnp.asarray(raw).sum())(
    [0.0, 0.02, 0.08, 0.15, 0.20, 0.18, 0.14, 0.10, 0.06, 0.04, 0.02, 0.01]
)

SYNTHETIC_HE_DAILY_HOSPITAL = "synthetic_he_daily_hospital"
SYNTHETIC_HE_WEEKLY_HOSPITAL = "synthetic_he_weekly_hospital"
SUBPOP_HOSPITAL_WASTEWATER_CA = "subpop_hospital_wastewater_ca"


def _build_synthetic_he_daily_hospital() -> DatasetBundle:  # numpydoc ignore=RT01
    """Build the synthetic H+E bundle with daily hospital admissions."""
    daily_hosp = load_synthetic_daily_hospital_admissions()
    daily_ed = load_synthetic_daily_ed_visits()
    true_params = load_synthetic_true_parameters()
    hosp_delay_pmf = jnp.array(
        load_example_infection_admission_interval()["probability_mass"].to_numpy()
    )
    ed_delay_pmf = jnp.array(true_params["ed_visits"]["delay_pmf"])
    ed_dow = jnp.array(true_params["ed_visits"]["day_of_week_effects"])

    obs_start = date(2023, 11, 6)
    hospital = SignalSeries(
        name="hospital",
        values=jnp.array(daily_hosp["daily_hosp_admits"].to_numpy(), dtype=jnp.float32),
        cadence="daily",
        start_date=obs_start,
        extras={"delay_pmf": hosp_delay_pmf},
    )
    ed_visits = SignalSeries(
        name="ed_visits",
        values=jnp.array(daily_ed["ed_visits"].to_numpy(), dtype=jnp.float32),
        cadence="daily",
        start_date=obs_start,
        extras={"delay_pmf": ed_delay_pmf, "day_of_week_effects": ed_dow},
    )
    return DatasetBundle(
        name=SYNTHETIC_HE_DAILY_HOSPITAL,
        population_size=float(daily_hosp["pop"][0]),
        obs_start_date=obs_start,
        n_days_post_init=126,
        signals={"hospital": hospital, "ed_visits": ed_visits},
        gen_int_pmf=GEN_INT_PMF,
        fixed_params={"i0_per_capita": true_params["i0_per_capita"]},
    )


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


def _build_subpop_hospital_wastewater_ca() -> DatasetBundle:  # numpydoc ignore=RT01
    """Build the hospital+wastewater subpopulation bundle for California."""
    hospital_data = load_hospital_data_for_state("CA", "2023-11-06.csv")
    wastewater_data = load_wastewater_data_for_state("CA", "fake_nwss.csv")
    hosp_delay_pmf = jnp.array(
        load_example_infection_admission_interval()["probability_mass"].to_numpy()
    )

    n_days_post_init = 90
    subpop_fractions = jnp.array([0.10, 0.14, 0.21, 0.22, 0.07, 0.26])
    ww_monitored_subpops = jnp.array([0, 1, 2, 3, 4])

    ww_mask = wastewater_data["time_indices"] < n_days_post_init
    ww_values = wastewater_data["observed_conc"][ww_mask]
    ww_sites = wastewater_data["site_ids"][ww_mask]
    ww_times = wastewater_data["time_indices"][ww_mask]
    n_ww_sites = int(wastewater_data["n_sites"])
    n_monitored = int(ww_monitored_subpops.shape[0])
    sensor_to_subpop = {
        i: int(ww_monitored_subpops[i % n_monitored]) for i in range(n_ww_sites)
    }
    ww_subpop_indices = jnp.array([sensor_to_subpop[int(s)] for s in ww_sites])

    hospital = SignalSeries(
        name="hospital",
        values=jnp.asarray(
            hospital_data["daily_admits"][:n_days_post_init], dtype=jnp.float32
        ),
        cadence="daily",
        start_date=hospital_data["dates"][0],
        extras={"delay_pmf": hosp_delay_pmf},
    )
    wastewater = SignalSeries(
        name="wastewater",
        values=ww_values,
        cadence="daily",
        start_date=hospital_data["dates"][0],
        times=ww_times,
        subpop_indices=ww_subpop_indices,
        sensor_indices=ww_sites,
        extras={
            "shedding_pmf": SHEDDING_PMF,
            "n_sensors": n_ww_sites,
        },
    )
    return DatasetBundle(
        name=SUBPOP_HOSPITAL_WASTEWATER_CA,
        population_size=float(hospital_data["population"]),
        obs_start_date=hospital_data["dates"][0],
        n_days_post_init=n_days_post_init,
        signals={"hospital": hospital, "wastewater": wastewater},
        gen_int_pmf=SUBPOP_GEN_INT_PMF,
        fixed_params={"subpop_fractions": subpop_fractions},
    )


_BUILDERS = {
    SYNTHETIC_HE_DAILY_HOSPITAL: _build_synthetic_he_daily_hospital,
    SYNTHETIC_HE_WEEKLY_HOSPITAL: _build_synthetic_he_weekly_hospital,
    SUBPOP_HOSPITAL_WASTEWATER_CA: _build_subpop_hospital_wastewater_ca,
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
