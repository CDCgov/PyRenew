# -*- coding: utf-8 -*-

"""
Ensures that models created with the same or
with different random keys behave appropriately.
"""


import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro as npro
import numpyro.distributions as dist
from pyrenew.deterministic import DeterministicPMF
from pyrenew.latent import (
    Infections,
    InfectionSeedingProcess,
    SeedInfectionsZeroPad,
)
from pyrenew.metaclass import DistributionalRV
from pyrenew.model import RtInfectionsRenewalModel
from pyrenew.observation import PoissonObservation
from pyrenew.process import RtRandomWalkProcess


def test_rng_keys_produce_correct_samples():
    """
    Tests that the random keys specified for
    MCMC sampling produce appropriate
    output if left to None or specified directly.
    """

    # set up singular epidemiological process
    pmf_array = jnp.array([0.25, 0.25, 0.25, 0.25])
    gen_int = DeterministicPMF(pmf_array, name="gen_int")
    I0 = InfectionSeedingProcess(
        "I0_seeding",
        DistributionalRV(dist=dist.LogNormal(0, 1), name="I0"),
        SeedInfectionsZeroPad(n_timepoints=gen_int.size()),
    )
    latent_infections = Infections()
    observed_infections = PoissonObservation()
    rt = RtRandomWalkProcess()

    # set up base models for testing
    model_01 = RtInfectionsRenewalModel(
        I0_rv=I0,
        gen_int_rv=gen_int,
        latent_infections_rv=latent_infections,
        infection_obs_process_rv=observed_infections,
        Rt_process_rv=rt,
    )
    model_02 = RtInfectionsRenewalModel(
        I0_rv=I0,
        gen_int_rv=gen_int,
        latent_infections_rv=latent_infections,
        infection_obs_process_rv=observed_infections,
        Rt_process_rv=rt,
    )
    model_03 = RtInfectionsRenewalModel(
        I0_rv=I0,
        gen_int_rv=gen_int,
        latent_infections_rv=latent_infections,
        infection_obs_process_rv=observed_infections,
        Rt_process_rv=rt,
    )
    model_04 = RtInfectionsRenewalModel(
        I0_rv=I0,
        gen_int_rv=gen_int,
        latent_infections_rv=latent_infections,
        infection_obs_process_rv=observed_infections,
        Rt_process_rv=rt,
    )
    model_05 = RtInfectionsRenewalModel(
        I0_rv=I0,
        gen_int_rv=gen_int,
        latent_infections_rv=latent_infections,
        infection_obs_process_rv=observed_infections,
        Rt_process_rv=rt,
    )

    # sample only a single model and use that model's samples
    # as the observed_infections for the rest of the models
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model_01_samp = model_01.sample(n_timepoints_to_simulate=30)

    # run each test model instance;
    model_01.run(
        num_warmup=50,
        num_samples=50,
        observed_infections=model_01_samp.sampled_observed_infections,
        rng_key=jr.key(54),  # rng_key specified, instance 1
        mcmc_args=dict(progress_bar=True),
    )
    model_02.run(
        num_warmup=50,
        num_samples=50,
        observed_infections=model_01_samp.sampled_observed_infections,
        rng_key=jr.key(54),  # rng_key specified, instance 2
        mcmc_args=dict(progress_bar=True),
    )
    model_03.run(
        num_warmup=50,
        num_samples=50,
        observed_infections=model_01_samp.sampled_observed_infections,
        rng_key=None,  # rng_key None, instance 1
        mcmc_args=dict(progress_bar=True),
    )
    model_04.run(
        num_warmup=50,
        num_samples=50,
        observed_infections=model_01_samp.sampled_observed_infections,
        rng_key=None,  # rng_key None, instance 2
        mcmc_args=dict(progress_bar=True),
    )
    model_05.run(
        num_warmup=50,
        num_samples=50,
        observed_infections=model_01_samp.sampled_observed_infections,
        rng_key=jr.key(74),  # different, but specified rng_key
        mcmc_args=dict(progress_bar=True),
    )

    # using same rng_key should get same run samples
    assert np.array_equal(
        model_01.mcmc.get_samples()["Rt"][0],
        model_02.mcmc.get_samples()["Rt"][0],
    )

    # using None for rng_key should get different run samples
    assert not np.array_equal(
        model_03.mcmc.get_samples()["Rt"][0],
        model_04.mcmc.get_samples()["Rt"][0],
    )

    # using None vs preselected rng_key should get different samples
    assert not np.array_equal(
        model_01.mcmc.get_samples()["Rt"][0],
        model_03.mcmc.get_samples()["Rt"][0],
    )

    # using two different non-None rng keys should get different samples
    assert not np.array_equal(
        model_02.mcmc.get_samples()["Rt"][0],
        model_05.mcmc.get_samples()["Rt"][0],
    )
