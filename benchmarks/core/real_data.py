"""Real-data provider for CDC NHSN + NSSP feeds.

Implements the :class:`DatasetProvider` protocol from
:mod:`benchmarks.core.signals` so suites can swap a synthetic provider
for live CDC data without changing the suite or the model builders.

Live observations and disease-specific PMFs require
``cfa-stf-routine-forecasting`` and valid Azure credentials at call time.
Location populations come from :mod:`benchmarks.core.reference_data` so the
benchmark does not call the R ``forecasttools`` package.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import polars as pl

from benchmarks.core.reference_data import population_for_location
from benchmarks.core.signals import (
    DatasetBundle,
    DatasetProvider,
    SignalSeries,
)

Disease = Literal["COVID-19", "Influenza", "RSV"]

NHSN_AVAILABILITY_START: dt.date = dt.date(2024, 11, 9)


@dataclass(frozen=True)
class RealDataSpec:
    """Parameters identifying one real-data extract.

    Parameters
    ----------
    disease
        Disease name accepted by ``cfa.stf.data``.
    loc_abbr
        US location abbreviation, e.g. ``"US"`` or ``"CA"``.
    as_of
        Vintage date applied to every reporting feed.
    n_training_days
        Length of the training window in days.
    n_days_to_omit
        Number of trailing days dropped to buffer against right truncation.
    signals
        Subset of ``{"hospital", "ed_visits"}`` to include in the bundle.
    """

    disease: Disease
    loc_abbr: str
    as_of: dt.date
    n_training_days: int = 150
    n_days_to_omit: int = 2
    signals: tuple[str, ...] = ("hospital", "ed_visits")


class RealDataProvider(DatasetProvider):
    """:class:`DatasetProvider` backed by ``cfa.stf.data`` feeds.

    Bundles are cached on first request so repeated suite candidates do
    not re-hit the reporting backend.

    Parameters
    ----------
    specs
        Mapping from dataset name to :class:`RealDataSpec`. Keys appear
        in ``--candidate`` arguments and in benchmark output.
    """

    def __init__(self, specs: dict[str, RealDataSpec]) -> None:
        """Store specs and initialise the in-memory cache."""
        self._specs: dict[str, RealDataSpec] = dict(specs)
        self._cache: dict[str, DatasetBundle] = {}

    def list_datasets(self) -> list[str]:  # numpydoc ignore=RT01
        """Return the dataset names this provider exposes."""
        return list(self._specs)

    def get(self, name: str) -> DatasetBundle:  # numpydoc ignore=RT01
        """Return the named bundle, building on first request."""
        if name not in self._specs:
            raise KeyError(
                f"Unknown dataset {name!r}. Available: {sorted(self._specs)}"
            )
        if name not in self._cache:
            self._cache[name] = _build_bundle(name, self._specs[name])
        return self._cache[name]


def _build_bundle(
    name: str, spec: RealDataSpec
) -> DatasetBundle:  # numpydoc ignore=RT01
    """Pull raw feeds and assemble a :class:`DatasetBundle` for one spec."""
    from cfa.stf.data import get_nnh_delay_pmf, get_nnh_generation_interval_pmf

    training_end = spec.as_of - dt.timedelta(days=1 + spec.n_days_to_omit)
    training_start = training_end - dt.timedelta(days=spec.n_training_days - 1)

    population = population_for_location(spec.loc_abbr)
    gen_int_pmf = jnp.asarray(
        get_nnh_generation_interval_pmf(disease=spec.disease, as_of=spec.as_of)
    )
    delay_pmf = jnp.asarray(get_nnh_delay_pmf(disease=spec.disease, as_of=spec.as_of))

    signals: dict[str, SignalSeries] = {}
    if "ed_visits" in spec.signals:
        signals["ed_visits"] = _build_ed_visits_signal(
            disease=spec.disease,
            loc_abbr=spec.loc_abbr,
            as_of=spec.as_of,
            start_date=training_start,
            end_date=training_end,
            delay_pmf=delay_pmf,
        )
    if "hospital" in spec.signals:
        signals["hospital"] = _build_hospital_signal(
            disease=spec.disease,
            loc_abbr=spec.loc_abbr,
            as_of=spec.as_of,
            start_date=max(training_start, NHSN_AVAILABILITY_START),
            end_date=training_end,
            delay_pmf=delay_pmf,
        )

    return DatasetBundle(
        name=name,
        population_size=float(population),
        obs_start_date=training_start,
        n_days_post_init=spec.n_training_days,
        signals=signals,
        gen_int_pmf=gen_int_pmf,
        fixed_params={},
    )


def _build_ed_visits_signal(
    disease: Disease,
    loc_abbr: str,
    as_of: dt.date,
    start_date: dt.date,
    end_date: dt.date,
    delay_pmf: jnp.ndarray,
) -> SignalSeries:  # numpydoc ignore=RT01
    """Build the daily ED-visits signal from ``get_nssp``."""
    from cfa.stf.data import get_nssp

    wide = (
        get_nssp(
            disease=[disease, "Total"],
            loc_abb=loc_abbr,
            as_of=as_of,
            start_date=start_date,
            end_date=end_date,
            lazy=False,
        )
        .select(["reference_date", "disease", "value"])
        .pivot(
            on="disease",
            index="reference_date",
            values="value",
            aggregate_function="first",
        )
        .rename({"reference_date": "date", disease: "observed_ed_visits"})
        .with_columns(
            (pl.col("Total") - pl.col("observed_ed_visits")).alias("other_ed_visits")
        )
        .sort("date")
    )
    return SignalSeries(
        name="ed_visits",
        values=jnp.asarray(wide["observed_ed_visits"].to_numpy(), dtype=jnp.float32),
        cadence="daily",
        start_date=wide["date"].min(),
        extras={
            "delay_pmf": delay_pmf,
            "other_ed_visits": jnp.asarray(
                wide["other_ed_visits"].to_numpy(), dtype=jnp.float32
            ),
        },
    )


def _build_hospital_signal(
    disease: Disease,
    loc_abbr: str,
    as_of: dt.date,
    start_date: dt.date,
    end_date: dt.date,
    delay_pmf: jnp.ndarray,
) -> SignalSeries:  # numpydoc ignore=RT01
    """Build the weekly hospital admissions signal from ``get_nhsn_hrd``."""
    from cfa.stf.data import get_nhsn_hrd

    raw = get_nhsn_hrd(
        disease=disease,
        loc_abb=loc_abbr,
        as_of=as_of,
        start_date=start_date,
        end_date=end_date,
        lazy=False,
    ).sort("weekendingdate")
    return SignalSeries(
        name="hospital",
        values=jnp.asarray(raw["hospital_admissions"].to_numpy(), dtype=jnp.float32),
        cadence="weekly",
        start_date=raw["weekendingdate"].min(),
        extras={"delay_pmf": delay_pmf, "aggregation": "weekly"},
    )
