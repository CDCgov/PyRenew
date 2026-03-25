"""
Shared fixtures for integration tests.

Provides synthetic data loading, model construction via PyrenewBuilder,
and ArviZ 1.0 posterior summary helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpyro.distributions as dist
import polars as pl
import pytest

from pyrenew.deterministic import DeterministicPMF
from pyrenew.latent import AR1
from pyrenew.latent.shared_infections import SharedInfections
from pyrenew.model import PyrenewBuilder
from pyrenew.observation import Counts, NegativeBinomialNoise
from pyrenew.randomvariable import DistributionalVariable

DATA_DIR = (
    Path(__file__).resolve().parents[2] / "pyrenew" / "datasets" / "synthetic_CA_120"
)
HOSP_DELAY_TSV = (
    Path(__file__).resolve().parents[2]
    / "pyrenew"
    / "datasets"
    / "infection_admission_interval.tsv"
)


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
    with open(DATA_DIR / "true_parameters.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def daily_infections() -> pl.DataFrame:
    """
    Load true daily infections and R(t) from synthetic data.

    Returns
    -------
    pl.DataFrame
        Columns: date, true_infections, true_rt.
    """
    return pl.read_csv(DATA_DIR / "daily_infections.csv")


@pytest.fixture(scope="module")
def daily_hosp() -> pl.DataFrame:
    """
    Load synthetic daily hospital admissions.

    Returns
    -------
    pl.DataFrame
        Columns: date, geo_value, daily_hosp_admits, pop.
    """
    return pl.read_csv(DATA_DIR / "daily_hospital_admissions.csv")


@pytest.fixture(scope="module")
def daily_ed() -> pl.DataFrame:
    """
    Load synthetic daily ED visits.

    Returns
    -------
    pl.DataFrame
        Columns: date, geo_value, disease, ed_visits.
    """
    return pl.read_csv(DATA_DIR / "daily_ed_visits.csv")


@pytest.fixture(scope="module")
def hosp_delay_pmf() -> jnp.ndarray:
    """
    Load infection-to-hospitalization delay PMF.

    Returns
    -------
    jnp.ndarray
        Delay PMF from infection_admission_interval.tsv.
    """
    pmf = pl.read_csv(HOSP_DELAY_TSV, separator="\t")["probability_mass"].to_numpy()
    return jnp.array(pmf)


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
def he_model(
    hosp_delay_pmf: jnp.ndarray,
    ed_delay_pmf: jnp.ndarray,
) -> PyrenewBuilder:
    """
    Build a SharedInfections model with hospital + ED observation processes.

    Parameters
    ----------
    hosp_delay_pmf : jnp.ndarray
        Infection-to-hospitalization delay PMF.
    ed_delay_pmf : jnp.ndarray
        Infection-to-ED-visit delay PMF.

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
        SharedInfections,
        gen_int_rv=DeterministicPMF("gen_int", gen_int_pmf),
        I0_rv=DistributionalVariable("I0", dist.Beta(1, 10)),
        initial_log_rt_rv=DistributionalVariable(
            "initial_log_rt", dist.Normal(0.0, 0.5)
        ),
        shared_rt_process=AR1(autoreg=0.9, innovation_sd=0.05),
    )

    hospital_obs = Counts(
        name="hospital",
        ascertainment_rate_rv=DistributionalVariable("ihr", dist.Beta(1, 100)),
        delay_distribution_rv=DeterministicPMF("hosp_delay", hosp_delay_pmf),
        noise=NegativeBinomialNoise(
            DistributionalVariable("hosp_conc", dist.LogNormal(5.0, 1.0))
        ),
    )
    builder.add_observation(hospital_obs)

    ed_obs = Counts(
        name="ed",
        ascertainment_rate_rv=DistributionalVariable("iedr", dist.Beta(1, 100)),
        delay_distribution_rv=DeterministicPMF("ed_delay", ed_delay_pmf),
        noise=NegativeBinomialNoise(
            DistributionalVariable("ed_conc", dist.LogNormal(4.0, 1.0))
        ),
    )
    builder.add_observation(ed_obs)

    return builder.build()
