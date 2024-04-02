# -*- coding: utf-8 -*-

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
from pyrenew.model import HospitalizationsModel
from pyrenew.observation import PoissonObservation
from pyrenew.process import RtRandomWalkProcess


class UniformProbForTest(RandomVariable):
    def __init__(self, pname: str):
        self.name = pname

        return None

    @staticmethod
    def validate(self):
        return None

    def sample(self, random_variables, constants):
        return (
            npro.sample(name=self.name, fn=dist.Uniform(high=0.99, low=0.01)),
        )


def test_model_hosp_no_obs_model():
    """
    Checks that the partially deterministic
    Hospitalization model runs
    """

    gen_int = DeterministicPMF(
        (jnp.array([0.25, 0.25, 0.25, 0.25]),),
    )

    I0 = Infections0(I0_dist=dist.LogNormal(0, 1))

    latent_infections = Infections()
    Rt_process = RtRandomWalkProcess()
    inf_hosp = DeterministicPMF(
        (
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
        ),
    )
    latent_hospitalizations = HospitalAdmissions(
        infection_to_admission_interval=inf_hosp,
        infections_varname="infections",
        hospitalizations_predicted_varname="observed_hospitalizations",
        infect_hosp_rate_dist=InfectHospRate(
            dist=dist.LogNormal(jnp.log(0.05), 0.05),
        ),
    )

    model0 = HospitalizationsModel(
        gen_int=gen_int,
        I0=I0,
        Rt_process=Rt_process,
        latent_infections=latent_infections,
        latent_hospitalizations=latent_hospitalizations,
        observed_hospitalizations=None,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model0_samp = model0.sample(constants={"n_timepoints": 30})

    model0.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        random_variables=dict(observed_hospitalizations=model0_samp.sampled),
        constants=dict(n_timepoints=30),
    )

    inf = model0.spread_draws(["observed_hospitalizations"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("observed_hospitalizations").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500


def test_model_hosp_with_obs_model():
    """
    Checks that the random Hospitalization model runs
    """

    gen_int = DeterministicPMF(
        (jnp.array([0.25, 0.25, 0.25, 0.25]),),
    )

    I0 = Infections0(I0_dist=dist.LogNormal(0, 1))

    latent_infections = Infections()
    Rt_process = RtRandomWalkProcess()
    observed_hospitalizations = PoissonObservation(
        rate_varname="latent",
        counts_varname="observed_hospitalizations",
    )

    inf_hosp = DeterministicPMF(
        (
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
        ),
    )

    latent_hospitalizations = HospitalAdmissions(
        infection_to_admission_interval=inf_hosp,
        infections_varname="infections",
        infect_hosp_rate_dist=InfectHospRate(
            dist=dist.LogNormal(jnp.log(0.05), 0.05),
        ),
    )

    model1 = HospitalizationsModel(
        gen_int=gen_int,
        I0=I0,
        Rt_process=Rt_process,
        latent_infections=latent_infections,
        latent_hospitalizations=latent_hospitalizations,
        observed_hospitalizations=observed_hospitalizations,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model1_samp = model1.sample(constants={"n_timepoints": 30})

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        random_variables=dict(
            observed_hospitalizations=model1_samp.sampled,
            infections=model1_samp.infections,
        ),
        constants=dict(n_timepoints=30),
    )

    inf = model1.spread_draws(["predicted_hospitalizations"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("predicted_hospitalizations").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500


def test_model_hosp_with_obs_model_weekday_phosp_2():
    """
    Checks that the random Hospitalization model runs
    """

    gen_int = DeterministicPMF(
        (jnp.array([0.25, 0.25, 0.25, 0.25]),),
    )

    I0 = Infections0(I0_dist=dist.LogNormal(0, 1))

    latent_infections = Infections()
    Rt_process = RtRandomWalkProcess()
    observed_hospitalizations = PoissonObservation(
        rate_varname="latent",
        counts_varname="observed_hospitalizations",
    )

    inf_hosp = DeterministicPMF(
        (
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
        ),
    )

    # Other random components
    weekday = jnp.array([1, 1, 1, 1, 2, 2])
    weekday = weekday / weekday.sum()
    weekday = jnp.tile(weekday, 10)
    weekday = weekday[:31]

    p_hosp = UniformProbForTest("p_hosp")
    weekday = UniformProbForTest("weekday")

    latent_hospitalizations = HospitalAdmissions(
        infection_to_admission_interval=inf_hosp,
        infections_varname="infections",
        weekday_effect_dist=weekday,
        p_report_dist=p_hosp,
        infect_hosp_rate_dist=InfectHospRate(
            dist=dist.LogNormal(jnp.log(0.05), 0.05),
        ),
    )

    model1 = HospitalizationsModel(
        I0=I0,
        gen_int=gen_int,
        Rt_process=Rt_process,
        latent_infections=latent_infections,
        latent_hospitalizations=latent_hospitalizations,
        observed_hospitalizations=observed_hospitalizations,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model1_samp = model1.sample(constants={"n_timepoints": 30})

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        random_variables=dict(
            observed_hospitalizations=model1_samp.sampled,
            infections=model1_samp.infections,
        ),
        constants=dict(n_timepoints=30),
    )

    inf = model1.spread_draws(["predicted_hospitalizations"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("predicted_hospitalizations").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500


def test_model_hosp_with_obs_model_weekday_phosp():
    """
    Checks that the random Hospitalization model runs
    """

    gen_int = DeterministicPMF(
        (jnp.array([0.25, 0.25, 0.25, 0.25]),),
    )

    I0 = Infections0(I0_dist=dist.LogNormal(0, 1))

    latent_infections = Infections()
    Rt_process = RtRandomWalkProcess()
    observed_hospitalizations = PoissonObservation(
        rate_varname="latent",
        counts_varname="observed_hospitalizations",
    )

    inf_hosp = DeterministicPMF(
        (
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
        ),
    )

    # Other random components
    weekday = jnp.array([1, 1, 1, 1, 2, 2])
    weekday = weekday / weekday.sum()
    weekday = jnp.tile(weekday, 10)
    weekday = weekday[:31]

    p_hosp = DeterministicVariable((weekday,))
    weekday = DeterministicVariable((weekday,))

    latent_hospitalizations = HospitalAdmissions(
        infection_to_admission_interval=inf_hosp,
        infections_varname="infections",
        weekday_effect_dist=weekday,
        p_report_dist=p_hosp,
        infect_hosp_rate_dist=InfectHospRate(
            dist=dist.LogNormal(jnp.log(0.05), 0.05),
        ),
    )

    model1 = HospitalizationsModel(
        I0=I0,
        gen_int=gen_int,
        Rt_process=Rt_process,
        latent_infections=latent_infections,
        latent_hospitalizations=latent_hospitalizations,
        observed_hospitalizations=observed_hospitalizations,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model1_samp = model1.sample(constants={"n_timepoints": 30})

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jax.random.PRNGKey(272),
        random_variables=dict(
            observed_hospitalizations=model1_samp.sampled,
            infections=model1_samp.infections,
        ),
        constants=dict(n_timepoints=30),
    )

    inf = model1.spread_draws(["predicted_hospitalizations"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("predicted_hospitalizations").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500
