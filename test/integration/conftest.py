"""
Shared fixtures for integration tests.

Provides synthetic data loading, model construction via PyrenewBuilder,
and ArviZ 1.0 posterior summary helpers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import polars as pl
import pytest

from pyrenew.datasets import (
    load_example_infection_admission_interval,
    load_synthetic_daily_ed_visits,
    load_synthetic_daily_hospital_admissions,
    load_synthetic_daily_infections,
    load_synthetic_true_parameters,
)
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import AR1
from pyrenew.latent.population_infections import PopulationInfections
from pyrenew.model import PyrenewBuilder
from pyrenew.observation import PopulationCounts, NegativeBinomialNoise
from pyrenew.randomvariable import DistributionalVariable


@pytest.fixture(scope="module")
def true_params() -> dict:
    """
    Load ground-truth parameters from the synthetic data generator.

    Returns
    -------
    dict
        True parameter values including R(t) trajectory,
        ascertainment rates, and delay PMFs.
    """
    return load_synthetic_true_parameters()


@pytest.fixture(scope="module")
def daily_infections() -> pl.DataFrame:
    """
    Load true daily infections and R(t) from synthetic data.

    Returns
    -------
    pl.DataFrame
        Columns: date, true_infections, true_rt.
    """
    return load_synthetic_daily_infections()


@pytest.fixture(scope="module")
def daily_hosp() -> pl.DataFrame:
    """
    Load synthetic daily hospital admissions.

    Returns
    -------
    pl.DataFrame
        Columns: date, geo_value, daily_hosp_admits, pop.
    """
    return load_synthetic_daily_hospital_admissions()


@pytest.fixture(scope="module")
def daily_ed() -> pl.DataFrame:
    """
    Load synthetic daily ED visits.

    Returns
    -------
    pl.DataFrame
        Columns: date, geo_value, disease, ed_visits.
    """
    return load_synthetic_daily_ed_visits()


@pytest.fixture(scope="module")
def hosp_delay_pmf() -> jnp.ndarray:
    """
    Load infection-to-hospitalization delay PMF.

    Returns
    -------
    jnp.ndarray
        Delay PMF from infection_admission_interval.tsv.
    """
    df = load_example_infection_admission_interval()
    return jnp.array(df["probability_mass"].to_numpy())


@pytest.fixture(scope="module")
def ed_delay_pmf(true_params: dict) -> jnp.ndarray:
    """
    Load ED visit delay PMF from true parameters.

    Parameters
    ----------
    true_params : dict
        Ground-truth parameter dictionary.

    Returns
    -------
    jnp.ndarray
        ED delay PMF.
    """
    return jnp.array(true_params["ed_visits"]["delay_pmf"])


@pytest.fixture(scope="module")
def ed_day_of_week_effects(true_params: dict) -> jnp.ndarray:
    """
    Load ED visit day-of-week effects from true parameters.

    Parameters
    ----------
    true_params : dict
        Ground-truth parameter dictionary.

    Returns
    -------
    jnp.ndarray
        Seven-element day-of-week multiplier vector.
    """
    return jnp.array(true_params["ed_visits"]["day_of_week_effects"])


@pytest.fixture(scope="module")
def he_model(
    hosp_delay_pmf: jnp.ndarray,
    ed_delay_pmf: jnp.ndarray,
    ed_day_of_week_effects: jnp.ndarray,
) -> PyrenewBuilder:
    """
    Build a PopulationInfections model with hospital + ED observation processes.

    Parameters
    ----------
    hosp_delay_pmf : jnp.ndarray
        Infection-to-hospitalization delay PMF.
    ed_delay_pmf : jnp.ndarray
        Infection-to-ED-visit delay PMF.
    ed_day_of_week_effects : jnp.ndarray
        Day-of-week multipliers used in synthetic ED generation.

    Returns
    -------
    MultiSignalModel
        Built model ready for fitting.
    """
    gen_int_pmf = jnp.array(
        [0.6326975, 0.2327564, 0.0856263, 0.03150015, 0.01158826, 0.00426308, 0.0015683]
    )

    builder = PyrenewBuilder()
    builder.configure_latent(
        PopulationInfections,
        gen_int_rv=DeterministicPMF("gen_int", gen_int_pmf),
        I0_rv=DistributionalVariable("I0", dist.Beta(1, 10)),
        log_rt_time_0_rv=DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5)),
        single_rt_process=AR1(autoreg=0.9, innovation_sd=0.05),
    )

    hospital_obs = PopulationCounts(
        name="hospital",
        ascertainment_rate_rv=DistributionalVariable("ihr", dist.Beta(1, 100)),
        delay_distribution_rv=DeterministicPMF("hosp_delay", hosp_delay_pmf),
        noise=NegativeBinomialNoise(
            DistributionalVariable("hosp_conc", dist.LogNormal(5.0, 1.0))
        ),
    )
    builder.add_observation(hospital_obs)

    ed_obs = PopulationCounts(
        name="ed",
        ascertainment_rate_rv=DistributionalVariable("iedr", dist.Beta(1, 100)),
        delay_distribution_rv=DeterministicPMF("ed_delay", ed_delay_pmf),
        noise=NegativeBinomialNoise(
            DistributionalVariable("ed_conc", dist.LogNormal(4.0, 1.0))
        ),
        day_of_week_rv=DeterministicVariable("ed_dow", ed_day_of_week_effects),
    )
    builder.add_observation(ed_obs)

    return builder.build()
