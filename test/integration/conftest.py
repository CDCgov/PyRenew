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

import pyrenew.transformation as transformation
from pyrenew.ascertainment import (
    IndependentAscertainment,
    JointAscertainment,
    RatioLinkedAscertainment,
)
from pyrenew.datasets import (
    load_example_infection_admission_interval,
    load_synthetic_daily_ed_visits,
    load_synthetic_daily_hospital_admissions,
    load_synthetic_daily_infections,
    load_synthetic_true_parameters,
    load_synthetic_weekly_hospital_admissions,
)
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    AR1,
    InfectionsWithFeedback,
    RandomWalk,
    WeeklyTemporalProcess,
)
from pyrenew.latent.infection_process import InfectionProcess
from pyrenew.latent.population_infections import PopulationInfections
from pyrenew.model import MultiSignalModel, PyrenewBuilder
from pyrenew.observation import NegativeBinomialNoise, PopulationCounts
from pyrenew.randomvariable import (
    DistributionalVariable,
    LogitNormalVariable,
    TransformedVariable,
)
from pyrenew.time import MMWR_WEEK
from test.test_helpers import fixed_ar1, fixed_ar1_state, fixed_differenced_ar1_state

_GEN_INT_PMF = jnp.array(
    [0.6326975, 0.2327564, 0.0856263, 0.03150015, 0.01158826, 0.00426308, 0.0015683]
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

    This fixture reproduces the PyRenew structure of the
    ``e_rtdaily_rw_timevary`` model without depending on the external ARM
    model builder. The ascertainment baseline is sampled independently and a
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
    infection_process = InfectionsWithFeedback(
        name="infections_with_feedback",
        infection_feedback_strength=TransformedVariable(
            "inf_feedback",
            DistributionalVariable(
                "inf_feedback_raw",
                dist.LogNormal(jnp.log(50.0), jnp.log(1.5)),
            ),
            transforms=transformation.AffineTransform(loc=0, scale=-1),
        ),
        infection_feedback_pmf=DeterministicPMF(
            "infection_feedback_pmf",
            _GEN_INT_PMF,
        ),
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
        infection_process=infection_process,
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
    single_rt_process: object,
    hosp_delay_pmf: jnp.ndarray,
    ed_delay_pmf: jnp.ndarray,
    ed_day_of_week_effects: jnp.ndarray,
    hospital_weekly: bool = False,
    infection_process: InfectionProcess | None = None,
) -> MultiSignalModel:
    """Build the shared hospital + ED PopulationInfections test model."""
    builder = PyrenewBuilder()
    builder.configure_latent(
        PopulationInfections,
        gen_int_rv=DeterministicPMF("gen_int", _GEN_INT_PMF),
        I0_rv=DistributionalVariable("I0", dist.Beta(1, 10)),
        log_rt_time_0_rv=DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5)),
        single_rt_process=single_rt_process,
        infection_process=infection_process,
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

    This fixture exercises the ``infection_process`` option passed through
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
    infection_process = InfectionsWithFeedback(
        name="infections",
        infection_feedback_strength=DeterministicVariable(
            "infection_feedback_strength",
            -1000.0,
        ),
        infection_feedback_pmf=DeterministicPMF(
            "infection_feedback_pmf",
            _GEN_INT_PMF,
        ),
    )
    return _build_he_population_model(
        single_rt_process=fixed_ar1(autoreg=0.9, innovation_sd=0.05),
        hosp_delay_pmf=hosp_delay_pmf,
        ed_delay_pmf=ed_delay_pmf,
        ed_day_of_week_effects=ed_day_of_week_effects,
        infection_process=infection_process,
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


@pytest.fixture(scope="module")
def he_weekly_rt_model(
    hosp_delay_pmf: jnp.ndarray,
    ed_delay_pmf: jnp.ndarray,
    ed_day_of_week_effects: jnp.ndarray,
) -> MultiSignalModel:
    """
    Build a PopulationInfections model with weekly-parameterized R(t).

    Same observation configuration as ``he_weekly_model`` (weekly hospital
    admissions on the MMWR epiweek grid + daily ED visits with a day-of-week
    effect), but R(t) is sampled weekly and broadcast to daily via
    ``WeeklyTemporalProcess``. This mirrors the production pyrenew-hew
    configuration.

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
        single_rt_process=WeeklyTemporalProcess(
            fixed_ar1(autoreg=0.9, innovation_sd=0.05),
            start_dow=MMWR_WEEK,
        ),
        hosp_delay_pmf=hosp_delay_pmf,
        ed_delay_pmf=ed_delay_pmf,
        ed_day_of_week_effects=ed_day_of_week_effects,
        hospital_weekly=True,
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
    hosp_delay_pmf: jnp.ndarray,
    ed_delay_pmf: jnp.ndarray,
    ed_day_of_week_effects: jnp.ndarray,
) -> MultiSignalModel:
    """Build the H+E model with state-centered weekly differenced AR1 Rt."""
    return _build_he_population_model(
        single_rt_process=WeeklyTemporalProcess(
            fixed_differenced_ar1_state(autoreg=0.9, innovation_sd=0.05),
            start_dow=MMWR_WEEK,
        ),
        hosp_delay_pmf=hosp_delay_pmf,
        ed_delay_pmf=ed_delay_pmf,
        ed_day_of_week_effects=ed_day_of_week_effects,
        hospital_weekly=True,
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
