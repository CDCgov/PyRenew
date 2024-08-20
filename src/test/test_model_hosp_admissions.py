# -*- coding: utf-8 -*-
# numpydoc ignore=GL08


from test.utils import simple_rt

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
import pytest

from pyrenew.deterministic import (
    DeterministicPMF,
    DeterministicVariable,
    NullObservation,
)
from pyrenew.latent import (
    HospitalAdmissions,
    InfectionInitializationProcess,
    Infections,
    InitializeInfectionsZeroPad,
)
from pyrenew.metaclass import DistributionalRV, RandomVariable, SampledValue
from pyrenew.model import HospitalAdmissionsModel
from pyrenew.observation import PoissonObservation


class UniformProbForTest(RandomVariable):  # numpydoc ignore=GL08
    def __init__(self, pname: str):  # numpydoc ignore=GL08
        self.name = pname

        return None

    @staticmethod
    def validate(self):  # numpydoc ignore=GL08
        return None

    def sample(self, **kwargs):  # numpydoc ignore=GL08
        return (
            SampledValue(
                numpyro.sample(
                    name=self.name, fn=dist.Uniform(high=0.99, low=0.01)
                )
            ),
        )


def test_model_hosp_no_timepoints_or_observations():
    """
    Checks that the hospital admissions model does not run
    without either n_datapoints or observed_admissions
    """

    gen_int = DeterministicPMF(
        name="gen_int", value=jnp.array([0.25, 0.25, 0.25, 0.25])
    )

    I0 = DistributionalRV(name="I0", distribution=dist.LogNormal(0, 1))

    latent_infections = Infections()
    Rt_process = simple_rt()

    observed_admissions = PoissonObservation("poisson_rv")

    inf_hosp = DeterministicPMF(
        name="inf_hosp",
        value=jnp.array(
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
        infection_to_admission_interval_rv=inf_hosp,
        infect_hosp_rate_rv=DistributionalRV(
            name="IHR", distribution=dist.LogNormal(jnp.log(0.05), 0.05)
        ),
    )

    model1 = HospitalAdmissionsModel(
        gen_int_rv=gen_int,
        I0_rv=I0,
        Rt_process_rv=Rt_process,
        latent_infections_rv=latent_infections,
        latent_hosp_admissions_rv=latent_admissions,
        hosp_admission_obs_process_rv=observed_admissions,
    )

    with numpyro.handlers.seed(rng_seed=223):
        with pytest.raises(ValueError, match="Either"):
            model1.sample(n_datapoints=None, data_observed_admissions=None)


def test_model_hosp_both_timepoints_and_observations():
    """
    Checks that the hospital admissions model does not run with
    both n_datapoints and observed_admissions passed
    """

    gen_int = DeterministicPMF(
        name="gen_int",
        value=jnp.array([0.25, 0.25, 0.25, 0.25]),
    )

    I0 = DistributionalRV(name="I0", distribution=dist.LogNormal(0, 1))

    latent_infections = Infections()
    Rt_process = simple_rt()

    observed_admissions = PoissonObservation("poisson_rv")

    inf_hosp = DeterministicPMF(
        name="inf_hosp",
        value=jnp.array(
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
        infection_to_admission_interval_rv=inf_hosp,
        infect_hosp_rate_rv=DistributionalRV(
            name="IHR", distribution=dist.LogNormal(jnp.log(0.05), 0.05)
        ),
    )

    model1 = HospitalAdmissionsModel(
        gen_int_rv=gen_int,
        I0_rv=I0,
        Rt_process_rv=Rt_process,
        latent_infections_rv=latent_infections,
        latent_hosp_admissions_rv=latent_admissions,
        hosp_admission_obs_process_rv=observed_admissions,
    )

    with numpyro.handlers.seed(rng_seed=223):
        with pytest.raises(ValueError, match="Cannot pass both"):
            model1.sample(
                n_datapoints=30,
                data_observed_hosp_admissions=jnp.repeat(jnp.nan, 30),
            )


def test_model_hosp_no_obs_model():
    """
    Checks that the partially deterministic
    Hospitalization model runs
    """

    gen_int = DeterministicPMF(
        name="gen_int",
        value=jnp.array([0.25, 0.25, 0.25, 0.25]),
    )

    I0 = InfectionInitializationProcess(
        "I0_initialization",
        DistributionalRV(name="I0", distribution=dist.LogNormal(0, 1)),
        InitializeInfectionsZeroPad(n_timepoints=gen_int.size()),
        t_unit=1,
    )

    latent_infections = Infections()
    Rt_process = simple_rt()

    inf_hosp = DeterministicPMF(
        name="inf_hosp",
        value=jnp.array(
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
        infection_to_admission_interval_rv=inf_hosp,
        infect_hosp_rate_rv=DistributionalRV(
            name="IHR",
            distribution=dist.LogNormal(jnp.log(0.05), 0.05),
        ),
    )

    model0 = HospitalAdmissionsModel(
        gen_int_rv=gen_int,
        I0_rv=I0,
        Rt_process_rv=Rt_process,
        latent_infections_rv=latent_infections,
        latent_hosp_admissions_rv=latent_admissions,
        hosp_admission_obs_process_rv=None,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    with numpyro.handlers.seed(rng_seed=223):
        model0_samp = model0.sample(n_datapoints=30)

    model0.hosp_admission_obs_process_rv = NullObservation()

    with numpyro.handlers.seed(rng_seed=223):
        model1_samp = model0.sample(n_datapoints=30)

    np.testing.assert_array_almost_equal(
        model0_samp.Rt.value, model1_samp.Rt.value
    )
    np.testing.assert_array_equal(
        model0_samp.latent_infections.value,
        model1_samp.latent_infections.value,
    )
    np.testing.assert_array_equal(
        model0_samp.infection_hosp_rate.value,
        model1_samp.infection_hosp_rate.value,
    )
    np.testing.assert_array_equal(
        model0_samp.latent_hosp_admissions.value,
        model1_samp.latent_hosp_admissions.value,
    )

    # These are supposed to be none, both
    assert model0_samp.observed_hosp_admissions.value is None
    assert model1_samp.observed_hosp_admissions.value is None

    model0.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jr.key(272),
        data_observed_hosp_admissions=model0_samp.latent_hosp_admissions.value,
    )

    inf = model0.spread_draws(["latent_hospital_admissions"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("latent_hospital_admissions").mean())
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
        name="gen_int", value=jnp.array([0.25, 0.25, 0.25, 0.25])
    )

    I0 = InfectionInitializationProcess(
        "I0_initialization",
        DistributionalRV(name="I0", distribution=dist.LogNormal(0, 1)),
        InitializeInfectionsZeroPad(n_timepoints=gen_int.size()),
        t_unit=1,
    )

    latent_infections = Infections()
    Rt_process = simple_rt()
    observed_admissions = PoissonObservation("poisson_rv")

    inf_hosp = DeterministicPMF(
        name="inf_hosp",
        value=jnp.array(
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
        infection_to_admission_interval_rv=inf_hosp,
        infect_hosp_rate_rv=DistributionalRV(
            name="IHR",
            distribution=dist.LogNormal(jnp.log(0.05), 0.05),
        ),
    )

    model1 = HospitalAdmissionsModel(
        gen_int_rv=gen_int,
        I0_rv=I0,
        Rt_process_rv=Rt_process,
        latent_infections_rv=latent_infections,
        latent_hosp_admissions_rv=latent_admissions,
        hosp_admission_obs_process_rv=observed_admissions,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    with numpyro.handlers.seed(rng_seed=233):
        model1_samp = model1.sample(n_datapoints=30)

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jr.key(272),
        data_observed_hosp_admissions=model1_samp.observed_hosp_admissions.value,
    )

    inf = model1.spread_draws(["latent_hospital_admissions"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("latent_hospital_admissions").mean())
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
        name="gen_int",
        value=jnp.array([0.25, 0.25, 0.25, 0.25]),
    )

    I0 = InfectionInitializationProcess(
        "I0_initialization",
        DistributionalRV(name="I0", distribution=dist.LogNormal(0, 1)),
        InitializeInfectionsZeroPad(n_timepoints=gen_int.size()),
        t_unit=1,
    )

    latent_infections = Infections()
    Rt_process = simple_rt()
    observed_admissions = PoissonObservation("poisson_rv")

    inf_hosp = DeterministicPMF(
        name="inf_hosp",
        value=jnp.array(
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
        infection_to_admission_interval_rv=inf_hosp,
        day_of_week_effect_rv=weekday,
        hosp_report_prob_rv=hosp_report_prob_dist,
        infect_hosp_rate_rv=DistributionalRV(
            name="IHR", distribution=dist.LogNormal(jnp.log(0.05), 0.05)
        ),
    )

    model1 = HospitalAdmissionsModel(
        I0_rv=I0,
        gen_int_rv=gen_int,
        Rt_process_rv=Rt_process,
        latent_infections_rv=latent_infections,
        latent_hosp_admissions_rv=latent_admissions,
        hosp_admission_obs_process_rv=observed_admissions,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    with numpyro.handlers.seed(rng_seed=223):
        model1_samp = model1.sample(n_datapoints=30)

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jr.key(272),
        data_observed_hosp_admissions=model1_samp.observed_hosp_admissions.value,
    )

    inf = model1.spread_draws(["latent_hospital_admissions"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("latent_hospital_admissions").mean())
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
        name="gen_int",
        value=jnp.array([0.25, 0.25, 0.25, 0.25]),
    )
    n_obs_to_generate = 30
    pad_size = 5

    I0 = InfectionInitializationProcess(
        "I0_initialization",
        DistributionalRV(name="I0", distribution=dist.LogNormal(0, 1)),
        InitializeInfectionsZeroPad(n_timepoints=gen_int.size()),
        t_unit=1,
    )

    latent_infections = Infections()
    Rt_process = simple_rt()

    observed_admissions = PoissonObservation("poisson_rv")

    inf_hosp = DeterministicPMF(
        name="inf_hosp",
        value=jnp.array(
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
    total_length = n_obs_to_generate + pad_size + gen_int.size()
    weekday = jnp.array([1, 1, 1, 1, 2, 2])
    weekday = weekday / weekday.sum()
    weekday = jnp.tile(weekday, 10)
    weekday = weekday[:total_length]

    weekday = DeterministicVariable(name="weekday", value=weekday)

    hosp_report_prob_dist = jnp.array([0.9, 0.8, 0.7, 0.7, 0.6, 0.4])
    hosp_report_prob_dist = jnp.tile(hosp_report_prob_dist, 10)
    hosp_report_prob_dist = hosp_report_prob_dist[:total_length]
    hosp_report_prob_dist = hosp_report_prob_dist / hosp_report_prob_dist.sum()

    hosp_report_prob_dist = DeterministicVariable(
        name="hosp_report_prob_dist",
        value=hosp_report_prob_dist,
    )

    latent_admissions = HospitalAdmissions(
        infection_to_admission_interval_rv=inf_hosp,
        day_of_week_effect_rv=weekday,
        hosp_report_prob_rv=hosp_report_prob_dist,
        infect_hosp_rate_rv=DistributionalRV(
            name="IHR",
            distribution=dist.LogNormal(jnp.log(0.05), 0.05),
        ),
    )

    model1 = HospitalAdmissionsModel(
        I0_rv=I0,
        gen_int_rv=gen_int,
        Rt_process_rv=Rt_process,
        latent_infections_rv=latent_infections,
        latent_hosp_admissions_rv=latent_admissions,
        hosp_admission_obs_process_rv=observed_admissions,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    with numpyro.handlers.seed(rng_seed=223):
        model1_samp = model1.sample(
            n_datapoints=n_obs_to_generate, padding=pad_size
        )

    # Showed during merge conflict, but unsure if it will be needed
    #  pad_size = 5
    # obs = jnp.hstack(
    #     [
    #         jnp.repeat(jnp.nan, pad_size),
    #         model1_samp.observed_hosp_admissions.value[pad_size:],
    #     ]
    # )
    # Running with padding
    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jr.key(272),
        data_observed_hosp_admissions=model1_samp.observed_hosp_admissions.value,
        padding=pad_size,
    )

    inf = model1.spread_draws(["latent_hospital_admissions"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("latent_hospital_admissions").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500
