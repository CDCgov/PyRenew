# numpydoc ignore=GL08

from test.utils import get_default_rt

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpy.testing import assert_array_equal
from pyrenew.deterministic import DeterministicPMF
from pyrenew.latent import (
    InfectionInitializationProcess,
    Infections,
    InitializeInfectionsZeroPad,
)
from pyrenew.metaclass import DistributionalRV
from pyrenew.model import RtInfectionsRenewalModel
from pyrenew.observation import PoissonObservation


def test_forecast():
    """Check that forecasts are the right length and match the posterior up until forecast begins."""
    pmf_array = jnp.array([0.25, 0.25, 0.25, 0.25])
    gen_int = DeterministicPMF(name="gen_int", value=pmf_array)
    I0 = InfectionInitializationProcess(
        "I0_initialization",
        DistributionalRV(name="I0", dist=dist.LogNormal(0, 1)),
        InitializeInfectionsZeroPad(n_timepoints=gen_int.size()),
        t_unit=1,
    )
    latent_infections = Infections()
    observed_infections = PoissonObservation(name="poisson_rv")
    rt = get_default_rt()

    model = RtInfectionsRenewalModel(
        I0_rv=I0,
        gen_int_rv=gen_int,
        latent_infections_rv=latent_infections,
        infection_obs_process_rv=observed_infections,
        Rt_process_rv=rt,
    )

    n_datapoints = 30
    n_forecast_points = 10
    with numpyro.handlers.seed(rng_seed=223):
        model_sample = model.sample(n_datapoints=n_datapoints)

    model.run(
        num_warmup=5,
        num_samples=5,
        data_observed_infections=model_sample.observed_infections.value,
        rng_key=jr.key(54),
    )

    posterior_predictive_samples = model.posterior_predictive(
        n_datapoints=n_datapoints + n_forecast_points,
    )

    # Check the length of the predictive distribution
    assert (
        len(posterior_predictive_samples["poisson_rv"][0])
        == n_datapoints + n_forecast_points
    )

    # Check the first elements of the posterior predictive Rt are the same as the
    # posterior Rt
    assert_array_equal(
        model.mcmc.get_samples()["Rt"][0],
        posterior_predictive_samples["Rt"][0][
            : len(model.mcmc.get_samples()["Rt"][0])
        ],
    )
