# numpydoc ignore=GL08


from test.utils import SimpleRt

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
import pytest

from pyrenew.deterministic import DeterministicPMF, NullObservation
from pyrenew.latent import (
    InfectionInitializationProcess,
    Infections,
    InitializeInfectionsZeroPad,
)
from pyrenew.model import RtInfectionsRenewalModel
from pyrenew.observation import PoissonObservation
from pyrenew.randomvariable import DistributionalVariable


def test_model_basicrenewal_no_timepoints_or_observations():
    """
    Test that the basic renewal model does not run
    without either n_datapoints or
    observed_admissions
    """

    gen_int = DeterministicPMF(
        name="gen_int", value=jnp.array([0.25, 0.25, 0.25, 0.25])
    )

    I0_init_rv = InfectionInitializationProcess(
        "I0_initialization",
        DistributionalVariable(name="I0", distribution=dist.LogNormal(0, 1)),
        InitializeInfectionsZeroPad(n_timepoints=gen_int.size()),
        t_unit=1,
    )

    latent_infections = Infections()

    observed_infections = PoissonObservation("poisson_rv")

    rt = SimpleRt()

    model1 = RtInfectionsRenewalModel(
        I0_rv=I0_init_rv,
        gen_int_rv=gen_int,
        latent_infections_rv=latent_infections,
        infection_obs_process_rv=observed_infections,
        Rt_process_rv=rt,
    )

    with numpyro.handlers.seed(rng_seed=223):
        with pytest.raises(ValueError, match="Either"):
            model1.sample(n_datapoints=None, data_observed_infections=None)


def test_model_basicrenewal_both_timepoints_and_observations():
    """
    Test that the basic renewal model does not run with both n_datapoints and observed_admissions passed
    """

    gen_int = DeterministicPMF(
        name="gen_int",
        value=jnp.array([0.25, 0.25, 0.25, 0.25]),
    )

    I0_init_rv = InfectionInitializationProcess(
        "I0_initialization",
        DistributionalVariable(name="I0", distribution=dist.LogNormal(0, 1)),
        InitializeInfectionsZeroPad(n_timepoints=gen_int.size()),
        t_unit=1,
    )

    latent_infections = Infections()

    observed_infections = PoissonObservation("possion_rv")

    rt = SimpleRt()

    model1 = RtInfectionsRenewalModel(
        I0_rv=I0_init_rv,
        gen_int_rv=gen_int,
        latent_infections_rv=latent_infections,
        infection_obs_process_rv=observed_infections,
        Rt_process_rv=rt,
    )

    with numpyro.handlers.seed(rng_seed=223):
        with pytest.raises(ValueError, match="Cannot pass both"):
            model1.sample(
                n_datapoints=30,
                data_observed_infections=jnp.repeat(jnp.nan, 30),
            )


def test_model_basicrenewal_no_obs_model():
    """
    Test the basic semi-deterministic renewal model runs. Semi-deterministic
    from the perspective of the infections. It returns expected, not sampled.
    """

    gen_int = DeterministicPMF(
        name="gen_int",
        value=jnp.array([0.25, 0.25, 0.25, 0.25]),
    )

    with pytest.raises(ValueError):
        _ = DistributionalVariable(name="I0", distribution=1)

    I0_init_rv = InfectionInitializationProcess(
        "I0_initialization",
        DistributionalVariable(name="I0", distribution=dist.LogNormal(0, 1)),
        InitializeInfectionsZeroPad(n_timepoints=gen_int.size()),
        t_unit=1,
    )

    latent_infections = Infections()

    rt = SimpleRt()

    model0 = RtInfectionsRenewalModel(
        gen_int_rv=gen_int,
        I0_rv=I0_init_rv,
        latent_infections_rv=latent_infections,
        Rt_process_rv=rt,
        # Explicitly use None, this should call the NullObservation
        infection_obs_process_rv=None,
    )

    # Sampling and fitting model 0 (with no obs for infections)
    with numpyro.handlers.seed(rng_seed=223):
        model0_samp = model0.sample(n_datapoints=30)
    model0_samp.Rt
    model0_samp.latent_infections
    model0_samp.observed_infections

    # Generating
    model0.infection_obs_process_rv = NullObservation()
    with numpyro.handlers.seed(rng_seed=223):
        model1_samp = model0.sample(n_datapoints=30)

    np.testing.assert_array_equal(model0_samp.Rt, model1_samp.Rt)
    np.testing.assert_array_equal(
        model0_samp.latent_infections,
        model1_samp.latent_infections,
    )
    np.testing.assert_array_equal(
        model0_samp.observed_infections,
        model1_samp.observed_infections,
    )

    model0.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jr.key(272),
        data_observed_infections=model0_samp.latent_infections,
    )

    inf = model0.spread_draws(["all_latent_infections"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("all_latent_infections").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500


def test_model_basicrenewal_with_obs_model():
    """
    Test the basic random renewal model runs. Random
    from the perspective of the infections. It returns sampled, not expected.
    """

    gen_int = DeterministicPMF(
        name="gen_int", value=jnp.array([0.25, 0.25, 0.25, 0.25])
    )

    I0_init_rv = InfectionInitializationProcess(
        "I0_initialization",
        DistributionalVariable(name="I0", distribution=dist.LogNormal(0, 1)),
        InitializeInfectionsZeroPad(n_timepoints=gen_int.size()),
        t_unit=1,
    )

    latent_infections = Infections()

    observed_infections = PoissonObservation("poisson_rv")

    rt = SimpleRt()

    model1 = RtInfectionsRenewalModel(
        I0_rv=I0_init_rv,
        gen_int_rv=gen_int,
        latent_infections_rv=latent_infections,
        infection_obs_process_rv=observed_infections,
        Rt_process_rv=rt,
    )

    # Sampling and fitting model 1 (with obs infections)
    with numpyro.handlers.seed(rng_seed=223):
        model1_samp = model1.sample(n_datapoints=30)

    print(model1_samp)
    print(model1_samp.Rt.size)
    print(model1_samp.latent_infections.size)
    print(model1_samp.observed_infections.size)

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jr.key(22),
        data_observed_infections=model1_samp.observed_infections,
    )

    inf = model1.spread_draws(["all_latent_infections"])
    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("all_latent_infections").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500


def test_model_basicrenewal_padding() -> None:  # numpydoc ignore=GL08
    gen_int = DeterministicPMF(
        name="gen_int", value=jnp.array([0.25, 0.25, 0.25, 0.25])
    )

    I0_init_rv = InfectionInitializationProcess(
        "I0_initialization",
        DistributionalVariable(name="I0", distribution=dist.LogNormal(0, 1)),
        InitializeInfectionsZeroPad(n_timepoints=gen_int.size()),
        t_unit=1,
    )

    latent_infections = Infections()

    observed_infections = PoissonObservation("poisson_rv")

    rt = SimpleRt()

    model1 = RtInfectionsRenewalModel(
        I0_rv=I0_init_rv,
        gen_int_rv=gen_int,
        latent_infections_rv=latent_infections,
        infection_obs_process_rv=observed_infections,
        Rt_process_rv=rt,
    )

    pad_size = 5

    with numpyro.handlers.seed(rng_seed=223):
        model1_samp = model1.sample(n_datapoints=30, padding=pad_size)

    model1.run(
        num_warmup=500,
        num_samples=500,
        rng_key=jr.key(22),
        data_observed_infections=model1_samp.observed_infections,
        padding=pad_size,
    )

    inf = model1.spread_draws(["all_latent_infections"])

    inf_mean = (
        inf.group_by("draw")
        .agg(pl.col("all_latent_infections").mean())
        .sort(pl.col("draw"))
    )

    # For now the assertion is only about the expected number of rows
    # It should be about the MCMC inference.
    assert inf_mean.to_numpy().shape[0] == 500
