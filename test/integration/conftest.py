"""
Shared fixtures for integration tests.

Provides synthetic data loading, model construction via PyrenewBuilder,
and ArviZ 1.0 posterior summary helpers.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import polars as pl
import pytest
from jax.typing import ArrayLike

import pyrenew.transformation as transformation
from pyrenew.ascertainment import (
    IndependentAscertainment,
    JointAscertainment,
    RatioLinkedAscertainment,
)
from pyrenew.datasets import (
    load_example_infection_admission_interval,
    load_hospital_data_for_state,
    load_synthetic_daily_ed_visits,
    load_synthetic_daily_hospital_admissions,
    load_synthetic_daily_infections,
    load_synthetic_true_parameters,
    load_synthetic_weekly_hospital_admissions,
    load_wastewater_data_for_state,
)
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    AR1,
    GammaGroupSdPrior,
    HierarchicalNormalPrior,
    RandomWalk,
    SubpopulationInfections,
    TemporalProcess,
    WeeklyTemporalProcess,
)
from pyrenew.latent.population_infections import PopulationInfections
from pyrenew.latent.temporal_processes import Parameterization
from pyrenew.metaclass import RandomVariable
from pyrenew.model import MultiSignalModel, PyrenewBuilder
from pyrenew.observation import (
    HierarchicalNormalNoise,
    MeasurementNoise,
    MeasurementObservation,
    NegativeBinomialNoise,
    PopulationCounts,
)
from pyrenew.randomvariable import (
    DistributionalVariable,
    LogitNormalVariable,
    TransformedVariable,
)
from pyrenew.time import MMWR_WEEK
from test.test_helpers import (
    fixed_ar1,
    fixed_ar1_state,
    fixed_differenced_ar1_state,
    fixed_random_walk,
)

_GEN_INT_PMF = jnp.array(
    [0.6326975, 0.2327564, 0.0856263, 0.03150015, 0.01158826, 0.00426308, 0.0015683]
)
_WEEKLY_RT_DATA = files("pyrenew.datasets") / "synthetic_CA_126_weekly_rt"


@pytest.fixture(scope="module")
def weekly_rt_true_params() -> dict[str, Any]:
    """
    Load the generating parameters for the weekly-Rt dataset.

    Returns
    -------
    dict[str, Any]
        Known weekly Rt values, ascertainment rates, delay PMFs, and dates.
    """
    return json.loads((_WEEKLY_RT_DATA / "true_parameters.json").read_text())


@pytest.fixture(scope="module")
def weekly_rt_daily_infections() -> pl.DataFrame:
    """
    Load daily infections and truth that is constant within each Rt week.

    Returns
    -------
    pl.DataFrame
        Daily dates, true infections, and true Rt.
    """
    return pl.read_csv(str(_WEEKLY_RT_DATA / "daily_infections.csv"))


@pytest.fixture(scope="module")
def weekly_rt_daily_ed() -> pl.DataFrame:
    """
    Load daily ED observations generated from the weekly Rt trajectory.

    Returns
    -------
    pl.DataFrame
        Daily ED visits and population size.
    """
    return pl.read_csv(str(_WEEKLY_RT_DATA / "daily_ed_visits.csv"))


@pytest.fixture(scope="module")
def weekly_rt_weekly_hosp() -> pl.DataFrame:
    """
    Load weekly hospital observations generated from the weekly Rt trajectory.

    Returns
    -------
    pl.DataFrame
        Week-ending dates, hospital admissions, and population size.
    """
    return pl.read_csv(str(_WEEKLY_RT_DATA / "weekly_hospital_admissions.csv"))


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
def weekly_hosp() -> pl.DataFrame:
    """
    Load synthetic weekly (MMWR epiweek) hospital admissions.

    Returns
    -------
    pl.DataFrame
        Columns: week_end, weekly_hosp_admits, location, pop.
    """
    return load_synthetic_weekly_hospital_admissions()


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
def e_rtdaily_rw_timevary_model(
    ed_delay_pmf: jnp.ndarray,
) -> MultiSignalModel:
    """Build an ED model with daily Rt and time-varying ascertainment.

    The ascertainment baseline is sampled independently and a
    calendar-aligned weekly AR(1) process supplies logit-scale deviations.

    Parameters
    ----------
    ed_delay_pmf
        Infection-to-ED-visit delay PMF from the synthetic dataset.

    Returns
    -------
    MultiSignalModel
        ED-only model ready for integration testing.
    """
    rt_process = RandomWalk(
        innovation_sd_rv=DistributionalVariable(
            "eta_sd",
            dist.TruncatedNormal(
                0.15 / jnp.sqrt(7.0),
                0.05 / jnp.sqrt(7.0),
                low=0,
            ),
        ),
        parameterization="innovation",
    )
    infection_feedback_strength_rv = TransformedVariable(
        "inf_feedback",
        DistributionalVariable(
            "inf_feedback_raw",
            dist.LogNormal(jnp.log(50.0), jnp.log(1.5)),
        ),
        transforms=transformation.AffineTransform(loc=0, scale=-1),
    )
    infection_feedback_pmf_rv = DeterministicPMF(
        "infection_feedback_pmf",
        _GEN_INT_PMF,
    )
    ascertainment = IndependentAscertainment(
        name="ed_ascertainment",
        rate_rvs={
            "ed": LogitNormalVariable(
                name="p_ed_visit",
                median=0.005,
                scale=0.3,
            )
        },
        temporal_processes={
            "ed": WeeklyTemporalProcess(
                AR1(
                    autoreg_rv=DistributionalVariable(
                        "autoreg_p_ed_visit",
                        dist.Beta(1, 100),
                    ),
                    innovation_sd_rv=DistributionalVariable(
                        "p_ed_visit_w_sd",
                        dist.TruncatedNormal(0.0, 0.01, low=0),
                    ),
                    parameterization="innovation",
                ),
                start_dow=MMWR_WEEK,
            )
        },
    )

    builder = PyrenewBuilder()
    builder.configure_latent(
        PopulationInfections,
        gen_int_rv=DeterministicPMF("generation_interval_pmf", _GEN_INT_PMF),
        I0_rv=DistributionalVariable("i0_first_obs_n_rv", dist.Beta(1, 10)),
        log_rt_time_0_rv=DistributionalVariable(
            "log_r_mu_intercept_rv",
            dist.Normal(jnp.log(1.2), jnp.log(jnp.sqrt(2.0))),
        ),
        single_rt_process=rt_process,
        infection_feedback_strength_rv=infection_feedback_strength_rv,
        infection_feedback_pmf_rv=infection_feedback_pmf_rv,
    )
    builder.add_ascertainment(ascertainment)
    builder.add_observation(
        PopulationCounts(
            name="ed",
            ascertainment_rate_rv=ascertainment.for_signal("ed"),
            delay_distribution_rv=DeterministicPMF("inf_to_ed", ed_delay_pmf),
            noise=NegativeBinomialNoise(
                DistributionalVariable(
                    "ed_visit_neg_bin_concentration",
                    dist.LogNormal(4.0, 1.0),
                )
            ),
            right_truncation_rv=DeterministicPMF(
                "right_truncation_pmf",
                jnp.array([1.0]),
            ),
            day_of_week_rv=TransformedVariable(
                "ed_visit_wday_effect",
                DistributionalVariable(
                    "ed_visit_wday_effect_raw",
                    dist.Dirichlet(jnp.full(7, 5.0)),
                ),
                transforms=transformation.AffineTransform(loc=0, scale=7),
            ),
        )
    )
    return builder.build()


def _build_he_population_model(  # numpydoc ignore=RT01
    *,
    single_rt_process: TemporalProcess,
    hosp_delay_pmf: jnp.ndarray,
    ed_delay_pmf: jnp.ndarray,
    ed_day_of_week_effects: jnp.ndarray,
    gen_int_pmf: jnp.ndarray | None = None,
    hospital_weekly: bool = False,
    infection_feedback_strength_rv: RandomVariable | None = None,
    infection_feedback_pmf_rv: RandomVariable | None = None,
) -> MultiSignalModel:
    """Build the shared hospital + ED PopulationInfections test model."""
    builder = PyrenewBuilder()
    builder.configure_latent(
        PopulationInfections,
        gen_int_rv=DeterministicPMF(
            "gen_int", _GEN_INT_PMF if gen_int_pmf is None else gen_int_pmf
        ),
        I0_rv=DistributionalVariable("I0", dist.Beta(1, 10)),
        log_rt_time_0_rv=DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5)),
        single_rt_process=single_rt_process,
        infection_feedback_strength_rv=infection_feedback_strength_rv,
        infection_feedback_pmf_rv=infection_feedback_pmf_rv,
    )

    ascertainment = IndependentAscertainment(
        name="he_ascertainment",
        rate_rvs={
            "hospital": DistributionalVariable("ihr", dist.Beta(1, 100)),
            "ed": DistributionalVariable("iedr", dist.Beta(1, 100)),
        },
    )
    builder.add_ascertainment(ascertainment)

    hospital_kwargs = {}
    if hospital_weekly:
        hospital_kwargs = {
            "aggregation": "weekly",
            "reporting_schedule": "regular",
            "start_dow": MMWR_WEEK,
        }

    builder.add_observation(
        PopulationCounts(
            name="hospital",
            ascertainment_rate_rv=ascertainment.for_signal("hospital"),
            delay_distribution_rv=DeterministicPMF("hosp_delay", hosp_delay_pmf),
            noise=NegativeBinomialNoise(
                DistributionalVariable("hosp_conc", dist.LogNormal(5.0, 1.0))
            ),
            **hospital_kwargs,
        )
    )
    builder.add_observation(
        PopulationCounts(
            name="ed",
            ascertainment_rate_rv=ascertainment.for_signal("ed"),
            delay_distribution_rv=DeterministicPMF("ed_delay", ed_delay_pmf),
            noise=NegativeBinomialNoise(
                DistributionalVariable("ed_conc", dist.LogNormal(4.0, 1.0))
            ),
            day_of_week_rv=DeterministicVariable("ed_dow", ed_day_of_week_effects),
        )
    )

    return builder.build()


@pytest.fixture(scope="module")
def he_model_with_infection_feedback(
    hosp_delay_pmf: jnp.ndarray,
    ed_delay_pmf: jnp.ndarray,
    ed_day_of_week_effects: jnp.ndarray,
) -> MultiSignalModel:
    """
    Build a PopulationInfections H+E model with infection feedback enabled.

    This fixture exercises the infection feedback options passed through
    ``PyrenewBuilder.configure_latent``.

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
        Built model ready for prior predictive checks.
    """
    return _build_he_population_model(
        single_rt_process=fixed_ar1(autoreg=0.9, innovation_sd=0.05),
        hosp_delay_pmf=hosp_delay_pmf,
        ed_delay_pmf=ed_delay_pmf,
        ed_day_of_week_effects=ed_day_of_week_effects,
        infection_feedback_strength_rv=DeterministicVariable(
            "infection_feedback_strength",
            -1000.0,
        ),
        infection_feedback_pmf_rv=DeterministicPMF(
            "infection_feedback_pmf",
            _GEN_INT_PMF,
        ),
    )


@pytest.fixture(scope="module")
def he_model(
    hosp_delay_pmf: jnp.ndarray,
    ed_delay_pmf: jnp.ndarray,
    ed_day_of_week_effects: jnp.ndarray,
) -> MultiSignalModel:
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
    return _build_he_population_model(
        single_rt_process=fixed_ar1(autoreg=0.9, innovation_sd=0.05),
        hosp_delay_pmf=hosp_delay_pmf,
        ed_delay_pmf=ed_delay_pmf,
        ed_day_of_week_effects=ed_day_of_week_effects,
    )


def _build_weekly_rt_population_model(  # numpydoc ignore=RT01
    inner: TemporalProcess, true_params: dict[str, Any]
) -> MultiSignalModel:
    """Build a weekly-Rt model using the dataset's known delay and calendar settings."""
    return _build_he_population_model(
        single_rt_process=WeeklyTemporalProcess(
            inner, start_dow=true_params["week_start_dow"]
        ),
        gen_int_pmf=jnp.array(true_params["generation_interval_pmf"]),
        hosp_delay_pmf=jnp.array(true_params["hospitalizations"]["delay_pmf"]),
        ed_delay_pmf=jnp.array(true_params["ed_visits"]["delay_pmf"]),
        ed_day_of_week_effects=jnp.array(
            true_params["ed_visits"]["day_of_week_effects"]
        ),
        hospital_weekly=True,
    )


@pytest.fixture(scope="module")
def he_weekly_rt_model(weekly_rt_true_params: dict[str, Any]) -> MultiSignalModel:
    """
    Build a weekly AR1 Rt model for the dedicated weekly-Rt dataset.

    Parameters
    ----------
    weekly_rt_true_params
        Generating parameters for the weekly-Rt dataset.

    Returns
    -------
    MultiSignalModel
        Weekly hospital and daily ED model ready for fitting.
    """
    return _build_weekly_rt_population_model(
        fixed_ar1(autoreg=0.9, innovation_sd=0.05), weekly_rt_true_params
    )


@pytest.fixture(scope="module")
def he_weekly_model(
    hosp_delay_pmf: jnp.ndarray,
    ed_delay_pmf: jnp.ndarray,
    ed_day_of_week_effects: jnp.ndarray,
) -> MultiSignalModel:
    """
    Build a PopulationInfections model with WEEKLY hospital + DAILY ED observations.

    The hospital observation is aggregated to MMWR epiweeks
    (Sunday-Saturday, via ``MMWR_WEEK``); the ED observation stays
    daily with a day-of-week effect. R(t) is parametrized at the
    finest observation cadence (daily) per the coherence rules for
    mixed-cadence models.

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
    return _build_he_population_model(
        single_rt_process=fixed_ar1(autoreg=0.9, innovation_sd=0.05),
        hosp_delay_pmf=hosp_delay_pmf,
        ed_delay_pmf=ed_delay_pmf,
        ed_day_of_week_effects=ed_day_of_week_effects,
        hospital_weekly=True,
    )


@pytest.fixture(scope="module")
def he_weekly_joint_ascertainment_model(
    true_params: dict,
    hosp_delay_pmf: jnp.ndarray,
    ed_delay_pmf: jnp.ndarray,
    ed_day_of_week_effects: jnp.ndarray,
) -> MultiSignalModel:
    """
    Build a weekly-hospital + daily-ED model with joint ascertainment.

    The hospital observation is aggregated to MMWR epiweeks, the ED visit
    observation stays daily, and both signal-specific ascertainment rates are
    sampled once from a shared ``JointAscertainment`` model. This is
    structurally comparable to the pyrenew-multisignal H+E model while keeping
    PyRenew's scalar ascertainment-rate interface.

    Parameters
    ----------
    true_params : dict
        Ground-truth parameter dictionary used to center the prior.
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
    true_ihr = true_params["hospitalizations"]["ihr"]
    true_iedr = true_params["ed_visits"]["iedr"]
    ascertainment = JointAscertainment(
        name="he_ascertainment",
        signals=("hospital", "ed_visits"),
        baseline_rates=jnp.array([true_ihr, true_iedr]),
        scale_tril=jnp.array(
            [
                [0.7, 0.0],
                [0.35, 0.606],
            ]
        ),
    )

    builder = PyrenewBuilder()
    builder.configure_latent(
        PopulationInfections,
        gen_int_rv=DeterministicPMF("gen_int", _GEN_INT_PMF),
        I0_rv=DistributionalVariable("I0", dist.Beta(1, 10)),
        log_rt_time_0_rv=DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5)),
        single_rt_process=fixed_ar1(autoreg=0.9, innovation_sd=0.05),
    )
    builder.add_ascertainment(ascertainment)

    hospital_obs = PopulationCounts(
        name="hospital",
        ascertainment_rate_rv=ascertainment.for_signal("hospital"),
        delay_distribution_rv=DeterministicPMF("hosp_delay", hosp_delay_pmf),
        noise=NegativeBinomialNoise(
            DistributionalVariable("hosp_conc", dist.LogNormal(5.0, 1.0))
        ),
        aggregation="weekly",
        reporting_schedule="regular",
        start_dow=MMWR_WEEK,
    )
    builder.add_observation(hospital_obs)

    ed_obs = PopulationCounts(
        name="ed_visits",
        ascertainment_rate_rv=ascertainment.for_signal("ed_visits"),
        delay_distribution_rv=DeterministicPMF("ed_delay", ed_delay_pmf),
        noise=NegativeBinomialNoise(
            DistributionalVariable("ed_conc", dist.LogNormal(4.0, 1.0))
        ),
        day_of_week_rv=DeterministicVariable("ed_dow", ed_day_of_week_effects),
    )
    builder.add_observation(ed_obs)

    return builder.build()


@pytest.fixture(scope="module")
def he_weekly_ratio_linked_ascertainment_model(
    true_params: dict,
    hosp_delay_pmf: jnp.ndarray,
    ed_delay_pmf: jnp.ndarray,
    ed_day_of_week_effects: jnp.ndarray,
) -> MultiSignalModel:
    """
    Build a weekly-hospital + daily-ED model with ratio-linked ascertainment.

    Parameters
    ----------
    true_params : dict
        Ground-truth parameter dictionary used to center the priors.
    hosp_delay_pmf : jnp.ndarray
        Infection-to-hospitalization delay PMF.
    ed_delay_pmf : jnp.ndarray
        Infection-to-ED-visit delay PMF.
    ed_day_of_week_effects : jnp.ndarray
        Day-of-week multipliers used in synthetic ED generation.

    Returns
    -------
    MultiSignalModel
        Built model ready for integration testing.
    """
    true_ihr = true_params["hospitalizations"]["ihr"]
    true_iedr = true_params["ed_visits"]["iedr"]
    beta_concentration = 200.0
    ascertainment = RatioLinkedAscertainment(
        name="he_ascertainment",
        base_signal="ed_visits",
        linked_signal="hospital",
        base_rate_rv=DistributionalVariable(
            "iedr",
            dist.Beta(
                true_iedr * beta_concentration,
                (1.0 - true_iedr) * beta_concentration,
            ),
        ),
        ratio_rv=DistributionalVariable(
            "ihr_rel_iedr",
            dist.LogNormal(jnp.log(true_ihr / true_iedr), 0.35),
        ),
    )

    builder = PyrenewBuilder()
    builder.configure_latent(
        PopulationInfections,
        gen_int_rv=DeterministicPMF("gen_int", _GEN_INT_PMF),
        I0_rv=DistributionalVariable("I0", dist.Beta(1, 10)),
        log_rt_time_0_rv=DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5)),
        single_rt_process=fixed_ar1(autoreg=0.9, innovation_sd=0.05),
    )
    builder.add_ascertainment(ascertainment)

    hospital_obs = PopulationCounts(
        name="hospital",
        ascertainment_rate_rv=ascertainment.for_signal("hospital"),
        delay_distribution_rv=DeterministicPMF("hosp_delay", hosp_delay_pmf),
        noise=NegativeBinomialNoise(
            DistributionalVariable("hosp_conc", dist.LogNormal(5.0, 1.0))
        ),
        aggregation="weekly",
        reporting_schedule="regular",
        start_dow=MMWR_WEEK,
    )
    builder.add_observation(hospital_obs)

    ed_obs = PopulationCounts(
        name="ed_visits",
        ascertainment_rate_rv=ascertainment.for_signal("ed_visits"),
        delay_distribution_rv=DeterministicPMF("ed_delay", ed_delay_pmf),
        noise=NegativeBinomialNoise(
            DistributionalVariable("ed_conc", dist.LogNormal(4.0, 1.0))
        ),
        day_of_week_rv=DeterministicVariable("ed_dow", ed_day_of_week_effects),
    )
    builder.add_observation(ed_obs)

    return builder.build()


@pytest.fixture(scope="module")
def he_model_state_centered(  # numpydoc ignore=RT01
    hosp_delay_pmf: jnp.ndarray,
    ed_delay_pmf: jnp.ndarray,
    ed_day_of_week_effects: jnp.ndarray,
) -> MultiSignalModel:
    """Build the H+E model with state-centered daily AR1 Rt."""
    return _build_he_population_model(
        single_rt_process=fixed_ar1_state(autoreg=0.9, innovation_sd=0.05),
        hosp_delay_pmf=hosp_delay_pmf,
        ed_delay_pmf=ed_delay_pmf,
        ed_day_of_week_effects=ed_day_of_week_effects,
    )


@pytest.fixture(scope="module")
def he_weekly_rt_model_state_centered(  # numpydoc ignore=RT01
    weekly_rt_true_params: dict[str, Any],
) -> MultiSignalModel:
    """Build a state-centered weekly differenced AR1 model for the weekly-Rt dataset."""
    return _build_weekly_rt_population_model(
        fixed_differenced_ar1_state(autoreg=0.9, innovation_sd=0.05),
        weekly_rt_true_params,
    )


@pytest.fixture(scope="module")
def he_weekly_model_state_centered(  # numpydoc ignore=RT01
    hosp_delay_pmf: jnp.ndarray,
    ed_delay_pmf: jnp.ndarray,
    ed_day_of_week_effects: jnp.ndarray,
) -> MultiSignalModel:
    """Build the weekly-hospital H+E model with state-centered daily AR1 Rt."""
    return _build_he_population_model(
        single_rt_process=fixed_ar1_state(autoreg=0.9, innovation_sd=0.05),
        hosp_delay_pmf=hosp_delay_pmf,
        ed_delay_pmf=ed_delay_pmf,
        ed_day_of_week_effects=ed_day_of_week_effects,
        hospital_weekly=True,
    )


class Wastewater(MeasurementObservation):
    """
    Wastewater viral concentration observation process.

    Transforms infections per resident of each subpopulation into
    predicted log-concentrations via shedding kinetics convolution and
    genome/volume scaling.
    """

    def __init__(
        self,
        name: str,
        shedding_kinetics_rv: RandomVariable,
        log10_genome_per_infection_rv: RandomVariable,
        ml_per_person_per_day: float,
        noise: MeasurementNoise,
    ) -> None:
        """
        Initialize wastewater observation process.

        Parameters
        ----------
        name
            Unique name for this observation process.
        shedding_kinetics_rv
            PMF for viral shedding kinetics over days post-infection.
        log10_genome_per_infection_rv
            Log10 genome copies shed per infection.
        ml_per_person_per_day
            Wastewater volume per person per day in mL.
        noise
            Noise model for wastewater measurements.
        """
        super().__init__(name=name, temporal_pmf_rv=shedding_kinetics_rv, noise=noise)
        self.log10_genome_per_infection_rv = log10_genome_per_infection_rv
        self.ml_per_person_per_day = ml_per_person_per_day

    def validate(self) -> None:
        """Validate shedding PMF and noise model."""
        shedding_pmf = self.temporal_pmf_rv()
        self._validate_pmf(shedding_pmf, "shedding_kinetics_rv")
        self.noise.validate()

    def lookback_days(self) -> int:
        """
        Return required lookback days for shedding kinetics.

        Returns
        -------
        int
            Shedding PMF length minus 1.
        """
        return len(self.temporal_pmf_rv()) - 1

    def _predicted_obs(self, infections: ArrayLike) -> ArrayLike:
        """
        Transform infections into predicted log-concentrations.

        Parameters
        ----------
        infections
            Infection matrix, shape (n_total, n_subpops).

        Returns
        -------
        ArrayLike
            Predicted log-concentrations, shape (n_total, n_subpops).
        """
        shedding_pmf = self.temporal_pmf_rv()
        log10_genome = self.log10_genome_per_infection_rv()

        def convolve_site(site_infections: ArrayLike) -> ArrayLike:
            """
            Convolve a single site's infections with shedding kinetics.

            Parameters
            ----------
            site_infections
                Infection time series for one site, shape (n_total,).

            Returns
            -------
            ArrayLike
                Convolved shedding signal for one site, shape (n_total,).
            """
            convolved, _ = self._convolve_delay_with_alignment(
                site_infections,
                shedding_pmf,
            )
            return convolved

        shedding_signal = jax.vmap(convolve_site, in_axes=1, out_axes=1)(infections)
        genome_copies = 10**log10_genome
        concentration = shedding_signal * genome_copies / self.ml_per_person_per_day
        return jnp.log(concentration)


@pytest.fixture(scope="module")
def ca_hosp_data() -> dict[str, Any]:
    """
    Load California hospital admissions data.

    Returns
    -------
    dict
        Dictionary with daily_admits, population, dates, n_days.
    """
    return load_hospital_data_for_state("CA", "2023-11-06.csv")


@pytest.fixture(scope="module")
def ca_ww_data() -> dict[str, Any]:
    """
    Load California wastewater surveillance data.

    Returns
    -------
    dict
        Dictionary with observed_conc, site_ids, time_indices,
        wwtp_names, n_sites, n_obs.
    """
    return load_wastewater_data_for_state("CA", "fake_nwss.csv")


def _build_hw_model(
    parameterization: Parameterization | None = None,
) -> MultiSignalModel:
    """
    Build a SubpopulationInfections model with hospital + wastewater observations.

    Uses COVID-19 generation interval, AR(1) baseline R(t),
    fixed RandomWalk or inferred weekly AR(1) deviations, infection-to-hospitalization
    delay from literature, and shedding kinetics for wastewater.

    Parameters
    ----------
    parameterization
        Weekly AR(1) parameterization, or None for the fixed random walk.

    Returns
    -------
    MultiSignalModel
        Built model ready for fitting.
    """
    gen_int_pmf = jnp.array([0.16, 0.32, 0.25, 0.14, 0.07, 0.04, 0.02])

    inf_to_hosp_pmf = jnp.array(
        [
            0,
            0.00469,
            0.01452,
            0.02786,
            0.04237,
            0.05581,
            0.06657,
            0.07379,
            0.07729,
            0.07737,
            0.07465,
            0.06988,
            0.06377,
            0.05696,
            0.04996,
            0.04315,
            0.03677,
            0.03097,
            0.02583,
            0.02135,
            0.01751,
            0.01427,
            0.01156,
            0.00931,
            0.00746,
            0.00596,
            0.00474,
            0.00375,
            0.00296,
            0.00233,
            0.00183,
            0.00143,
            0.00107,
            0.00077,
            0.00054,
            0.00036,
            0.00024,
            0.00015,
            0.00009,
            0.00005,
            0.00003,
            0.00002,
            0.00001,
        ]
    )

    shedding_pmf = jnp.array(
        [0.0, 0.02, 0.08, 0.15, 0.20, 0.18, 0.14, 0.10, 0.06, 0.04, 0.02, 0.01]
    )
    shedding_pmf = shedding_pmf / shedding_pmf.sum()

    builder = PyrenewBuilder()
    builder.configure_latent(
        SubpopulationInfections,
        gen_int_rv=DeterministicPMF("gen_int", gen_int_pmf),
        I0_rv=DistributionalVariable("I0", dist.Beta(1, 100)),
        log_rt_time_0_rv=DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5)),
        baseline_rt_process=fixed_ar1(autoreg=0.9, innovation_sd=0.05),
        subpop_rt_deviation_process=(
            fixed_random_walk(innovation_sd=0.025)
            if parameterization is None
            else WeeklyTemporalProcess(
                AR1(
                    autoreg_rv=DistributionalVariable(
                        "deviation_autoreg", dist.Beta(8, 2)
                    ),
                    innovation_sd_rv=DistributionalVariable(
                        "deviation_innovation_sd", dist.LogNormal(jnp.log(0.05), 0.5)
                    ),
                    parameterization=parameterization,
                ),
                start_dow=MMWR_WEEK,
            )
        ),
    )

    ascertainment = IndependentAscertainment(
        name="hospital_ascertainment",
        rate_rvs={"hospital": DeterministicVariable("ihr", 0.01)},
    )
    builder.add_ascertainment(ascertainment)

    hosp_obs = PopulationCounts(
        name="hospital",
        ascertainment_rate_rv=ascertainment.for_signal("hospital"),
        delay_distribution_rv=DeterministicPMF("inf_to_hosp_delay", inf_to_hosp_pmf),
        noise=NegativeBinomialNoise(DeterministicVariable("hosp_concentration", 10.0)),
    )
    builder.add_observation(hosp_obs)

    ww_obs = Wastewater(
        name="wastewater",
        shedding_kinetics_rv=DeterministicPMF("shedding_kinetics", shedding_pmf),
        log10_genome_per_infection_rv=DeterministicVariable(
            "log10_genome_per_inf", 4.1
        ),
        ml_per_person_per_day=1000.0,
        noise=HierarchicalNormalNoise(
            HierarchicalNormalPrior(
                "ww_site_mode",
                sd_rv=DeterministicVariable("site_mode_sd", 0.5),
            ),
            GammaGroupSdPrior(
                "ww_site_sd",
                sd_mean_rv=DeterministicVariable("site_sd_mean", 0.3),
                sd_concentration_rv=DeterministicVariable("site_sd_conc", 4.0),
            ),
        ),
    )
    builder.add_observation(ww_obs)

    return builder.build()


@pytest.fixture(
    scope="module",
    params=[None, "innovation", "state"],
    ids=["fixed_random_walk", "weekly_innovation", "weekly_state"],
)
def hw_model(request: pytest.FixtureRequest) -> MultiSignalModel:
    """Build each hospital–wastewater configuration for fitting and prediction.

    Parameters
    ----------
    request
        Selected deviation process configuration.

    Returns
    -------
    MultiSignalModel
        Model ready for fitting to tutorial data.
    """
    return _build_hw_model(request.param)
