# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax
import jax.numpy as jnp
import numpy as np
import numpyro as npro
import numpyro.distributions as dist
import polars as pl
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    HospitalAdmissions,
    InfectHospRate,
    Infections,
    Infections0,
)
from pyrenew.metaclass import RandomVariable
from pyrenew.model import HospitalAdmissionsModel
from pyrenew.observation import PoissonObservation
from pyrenew.process import RtRandomWalkProcess


class UniformProbForTest(RandomVariable):
    def __init__(self, pname: str):
        self.name = pname

        return None

    @staticmethod
    def validate(self):
        return None

    def sample(self, **kwargs):
        return (
            npro.sample(name=self.name, fn=dist.Uniform(high=0.99, low=0.01)),
        )


def test_model_hosp_no_obs_model():
    """
    Checks that the partially deterministic
    Hospitalization model runs
    """

    gen_int = DeterministicPMF(jnp.array([0.25, 0.25, 0.25, 0.25]))

    I0 = Infections0(I0_dist=dist.LogNormal(0, 1))

    latent_infections = Infections()
    Rt_process = RtRandomWalkProcess()
    inf_hosp = DeterministicPMF(
        jnp.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.25,
                0.5,
                0.1,
                0.1,
                0.05,
            ]
        ),
    )

    latent_admissions = HospitalAdmissions(
        infection_to_admission_interval=inf_hosp,
        admissions_predicted_varname="observed_admissions",
        infect_hosp_rate_dist=InfectHospRate(
            dist=dist.LogNormal(jnp.log(0.05), 0.05),
        ),
    )

    model0 = HospitalAdmissionsModel(
        gen_int=gen_int,
        I0=I0,
        Rt_process=Rt_process,
        latent_infections=latent_infections,
        latent_admissions=latent_admissions,
        observation_process=DeterministicVariable(0),
    )

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model0_samp = model0.sample(n_timepoints=30)

    model0.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        obs_mean=model0_samp.sampled_admissions,
        n_timepoints=30,
    )

    inf = model0.spread_draws(["observed_admissions"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("observed_admissions").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500


def test_model_hosp_with_obs_model():
    """
    Checks that the random Hospitalization model runs
    """

    gen_int = DeterministicPMF(jnp.array([0.25, 0.25, 0.25, 0.25]))

    I0 = Infections0(I0_dist=dist.LogNormal(0, 1))

    latent_infections = Infections()
    Rt_process = RtRandomWalkProcess()
    observed_admissions = PoissonObservation()

    inf_hosp = DeterministicPMF(
        jnp.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.25,
                0.5,
                0.1,
                0.1,
                0.05,
            ],
        ),
    )

    latent_admissions = HospitalAdmissions(
        infection_to_admission_interval=inf_hosp,
        infect_hosp_rate_dist=InfectHospRate(
            dist=dist.LogNormal(jnp.log(0.05), 0.05),
        ),
    )

    model1 = HospitalAdmissionsModel(
        gen_int=gen_int,
        I0=I0,
        Rt_process=Rt_process,
        latent_infections=latent_infections,
        latent_admissions=latent_admissions,
        observation_process=observed_admissions,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model1_samp = model1.sample(n_timepoints=30)

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        observed_admissions=model1_samp.sampled_admissions,
        n_timepoints=30,
    )

    inf = model1.spread_draws(["predicted_admissions"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("predicted_admissions").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500


def test_model_hosp_with_obs_model_weekday_phosp_2():
    """
    Checks that the random Hospitalization model runs
    """

    gen_int = DeterministicPMF(jnp.array([0.25, 0.25, 0.25, 0.25]))

    I0 = Infections0(I0_dist=dist.LogNormal(0, 1))

    latent_infections = Infections()
    Rt_process = RtRandomWalkProcess()
    observed_admissions = PoissonObservation()

    inf_hosp = DeterministicPMF(
        jnp.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.25,
                0.5,
                0.1,
                0.1,
                0.05,
            ],
        ),
    )

    # Other random components
    weekday = jnp.array([1, 1, 1, 1, 2, 2])
    weekday = weekday / weekday.sum()
    weekday = jnp.tile(weekday, 10)
    weekday = weekday[:31]

    hosp_report_prob_dist = UniformProbForTest("hosp_report_prob_dist")
    weekday = UniformProbForTest("weekday")

    latent_admissions = HospitalAdmissions(
        infection_to_admission_interval=inf_hosp,
        weekday_effect_dist=weekday,
        hosp_report_prob_dist=hosp_report_prob_dist,
        infect_hosp_rate_dist=InfectHospRate(
            dist=dist.LogNormal(jnp.log(0.05), 0.05),
        ),
    )

    model1 = HospitalAdmissionsModel(
        I0=I0,
        gen_int=gen_int,
        Rt_process=Rt_process,
        latent_infections=latent_infections,
        latent_admissions=latent_admissions,
        observation_process=observed_admissions,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model1_samp = model1.sample(n_timepoints=30)

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        observed_admissions=model1_samp.sampled_admissions,
        n_timepoints=30,
    )

    inf = model1.spread_draws(["predicted_admissions"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("predicted_admissions").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500


def test_model_hosp_with_obs_model_weekday_phosp():
    """
    Checks that the random Hospitalization model runs
    """

    gen_int = DeterministicPMF(jnp.array([0.25, 0.25, 0.25, 0.25]))

    I0 = Infections0(I0_dist=dist.LogNormal(0, 1))

    latent_infections = Infections()
    Rt_process = RtRandomWalkProcess()
    observed_admissions = PoissonObservation()

    inf_hosp = DeterministicPMF(
        jnp.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.25,
                0.5,
                0.1,
                0.1,
                0.05,
            ],
        ),
    )

    # Other random components
    weekday = jnp.array([1, 1, 1, 1, 2, 2])
    weekday = jnp.tile(weekday, 10)
    weekday = weekday / weekday.sum()
    weekday = weekday[:31]

    weekday = DeterministicVariable(weekday)

    hosp_report_prob_dist = jnp.array([0.9, 0.8, 0.7, 0.7, 0.6, 0.4])
    hosp_report_prob_dist = jnp.tile(hosp_report_prob_dist, 10)
    hosp_report_prob_dist = hosp_report_prob_dist / hosp_report_prob_dist.sum()

    hosp_report_prob_dist = hosp_report_prob_dist[:31]

    hosp_report_prob_dist = DeterministicVariable(vars=hosp_report_prob_dist)

    latent_admissions = HospitalAdmissions(
        infection_to_admission_interval=inf_hosp,
        weekday_effect_dist=weekday,
        hosp_report_prob_dist=hosp_report_prob_dist,
        infect_hosp_rate_dist=InfectHospRate(
            dist=dist.LogNormal(jnp.log(0.05), 0.05),
        ),
    )

    model1 = HospitalAdmissionsModel(
        I0=I0,
        gen_int=gen_int,
        Rt_process=Rt_process,
        latent_infections=latent_infections,
        latent_admissions=latent_admissions,
        observation_process=observed_admissions,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model1_samp = model1.sample(n_timepoints=30)

    obs = jnp.hstack(
        [jnp.repeat(jnp.nan, 5), model1_samp.sampled_admissions[5:]]
    )

    # Running with padding
    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        observed_admissions=obs,
        n_timepoints=30,
        padding=5,
    )

    inf = model1.spread_draws(["predicted_admissions"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("predicted_admissions").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500
