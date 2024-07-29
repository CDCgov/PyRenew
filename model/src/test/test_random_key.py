# -*- coding: utf-8 -*-

"""
Ensures that models created with the same or
with different random keys behave appropriately.
"""

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pyrenew.transformation as t
from numpy.testing import assert_array_equal, assert_raises
from pyrenew.deterministic import DeterministicPMF
from pyrenew.latent import (
    InfectionInitializationProcess,
    Infections,
    InitializeInfectionsZeroPad,
)
from pyrenew.metaclass import DistributionalRV, TransformedRandomVariable
from pyrenew.model import RtInfectionsRenewalModel
from pyrenew.observation import PoissonObservation
from pyrenew.process import SimpleRandomWalkProcess


def create_test_model():  # numpydoc ignore=GL08
    pmf_array = jnp.array([0.25, 0.25, 0.25, 0.25])
    gen_int = DeterministicPMF(name="gen_int", value=pmf_array)
    I0 = InfectionInitializationProcess(
        "I0_initialization",
        DistributionalRV(name="I0", dist=dist.LogNormal(0, 1)),
        InitializeInfectionsZeroPad(n_timepoints=gen_int.size()),
        t_unit=1,
    )
    latent_infections = Infections()
    observed_infections = PoissonObservation("poisson_rv")
    rt = TransformedRandomVariable(
        "Rt_rv",
        base_rv=SimpleRandomWalkProcess(
            name="log_rt",
            step_rv=DistributionalRV(
                name="rw_step_rv", dist=dist.Normal(0, 0.025)
            ),
            init_rv=DistributionalRV(
                name="init_log_rt", dist=dist.Normal(0, 0.2)
            ),
        ),
        transforms=t.ExpTransform(),
    )
    model = RtInfectionsRenewalModel(
        I0_rv=I0,
        gen_int_rv=gen_int,
        latent_infections_rv=latent_infections,
        infection_obs_process_rv=observed_infections,
        Rt_process_rv=rt,
    )
    return model


def run_test_model(
    test_model, observed_infections, rng_key
):  # numpydoc ignore=GL08
    test_model.run(
        num_warmup=50,
        num_samples=50,
        data_observed_infections=observed_infections,
        rng_key=rng_key,
        mcmc_args=dict(progress_bar=True),
    )


def prior_predictive_test_model(
    test_model, n_datapoints, rng_key
):  # numpydoc ignore=GL08
    prior_predictive_samples = test_model.prior_predictive(
        rng_key=rng_key,
        numpyro_predictive_args={"num_samples": 20},
        n_datapoints=n_datapoints,
    )
    return prior_predictive_samples


def posterior_predictive_test_model(
    test_model, n_datapoints, rng_key
):  # numpydoc ignore=GL08
    posterior_predictive_samples = test_model.posterior_predictive(
        rng_key=rng_key, n_datapoints=n_datapoints
    )
    return posterior_predictive_samples


def test_rng_keys_produce_correct_samples():
    """
    Tests that the random keys specified for
    MCMC sampling produce appropriate
    output if left to None or specified directly.
    """

    # set up singular epidemiological process

    # set up base models for testing
    models = [create_test_model() for _ in range(5)]
    n_datapoints = [30] * len(models)
    n_timepoints_posterior_predictive = [
        x + models[0].gen_int_rv.size() for x in n_datapoints
    ]
    # sample only a single model and use that model's samples
    # as the observed_infections for the rest of the models
    with numpyro.handlers.seed(rng_seed=223):
        model_sample = models[0].sample(n_datapoints=n_datapoints[0])
    obs_infections = [model_sample.observed_infections.value] * len(models)
    rng_keys = [jr.key(54), jr.key(54), None, None, jr.key(74)]

    # run test models with the different keys

    for elt in list(zip(models, obs_infections, rng_keys)):
        run_test_model(*elt)

    prior_predictive_list = [
        prior_predictive_test_model(*elt)
        for elt in list(zip(models, n_datapoints, rng_keys))
    ]

    posterior_predictive_list = [
        posterior_predictive_test_model(*elt)
        for elt in list(
            zip(models, n_timepoints_posterior_predictive, rng_keys)
        )
    ]
    # using same rng_key should get same run samples
    assert_array_equal(
        models[0].mcmc.get_samples()["Rt"][0],
        models[1].mcmc.get_samples()["Rt"][0],
    )

    assert_array_equal(
        prior_predictive_list[0]["Rt"][0],
        prior_predictive_list[1]["Rt"][0],
    )

    assert_array_equal(
        posterior_predictive_list[0]["Rt"][0],
        posterior_predictive_list[1]["Rt"][0],
    )

    # using None for rng_key should get different run samples
    assert_raises(  # negate assert_array_equal
        AssertionError,
        assert_array_equal,
        models[2].mcmc.get_samples()["Rt"][0],
        models[3].mcmc.get_samples()["Rt"][0],
    )

    assert_raises(
        AssertionError,
        assert_array_equal,
        prior_predictive_list[2]["Rt"][0],
        prior_predictive_list[3]["Rt"][0],
    )

    assert_raises(
        AssertionError,
        assert_array_equal,
        posterior_predictive_list[2]["Rt"][0],
        posterior_predictive_list[3]["Rt"][0],
    )

    # using None vs preselected rng_key should get different samples
    assert_raises(
        AssertionError,
        assert_array_equal,
        models[0].mcmc.get_samples()["Rt"][0],
        models[2].mcmc.get_samples()["Rt"][0],
    )

    assert_raises(
        AssertionError,
        assert_array_equal,
        prior_predictive_list[0]["Rt"][0],
        prior_predictive_list[2]["Rt"][0],
    )

    assert_raises(
        AssertionError,
        assert_array_equal,
        posterior_predictive_list[0]["Rt"][0],
        posterior_predictive_list[2]["Rt"][0],
    )

    # using two different non-None rng keys should get different samples
    assert_raises(
        AssertionError,
        assert_array_equal,
        models[1].mcmc.get_samples()["Rt"][0],
        models[4].mcmc.get_samples()["Rt"][0],
    )

    assert_raises(
        AssertionError,
        assert_array_equal,
        prior_predictive_list[1]["Rt"][0],
        prior_predictive_list[4]["Rt"][0],
    )

    assert_raises(
        AssertionError,
        assert_array_equal,
        posterior_predictive_list[1]["Rt"][0],
        posterior_predictive_list[4]["Rt"][0],
    )
