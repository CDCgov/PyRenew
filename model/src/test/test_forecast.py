# numpydoc ignore=GL08

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro as npro
import numpyro.distributions as dist
import pyrenew.transformation as t
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
from pyrenew.process import RtRandomWalkProcess


def test_forecast():
    """Check that forecasts are the right length and match the posterior up until forecast begins."""
    pmf_array = jnp.array([0.25, 0.25, 0.25, 0.25])
    gen_int = DeterministicPMF(pmf_array, name="gen_int")
    I0 = InfectionInitializationProcess(
        "I0_initialization",
        DistributionalRV(dist=dist.LogNormal(0, 1), name="I0"),
        InitializeInfectionsZeroPad(n_timepoints=gen_int.size()),
        t_unit=1,
    )
    latent_infections = Infections()
    observed_infections = PoissonObservation("poisson_rv")
    rt = RtRandomWalkProcess(
        Rt0_dist=dist.TruncatedNormal(loc=1.2, scale=0.2, low=0),
        Rt_transform=t.ExpTransform().inv,
        Rt_rw_dist=dist.Normal(0, 0.025),
    )
    model = RtInfectionsRenewalModel(
        I0_rv=I0,
        gen_int_rv=gen_int,
        latent_infections_rv=latent_infections,
        infection_obs_process_rv=observed_infections,
        Rt_process_rv=rt,
    )

    n_timepoints_to_simulate = 30
    n_forecast_points = 10
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        model_sample = model.sample(
            n_timepoints_to_simulate=n_timepoints_to_simulate
        )

    model.run(
        num_warmup=5,
        num_samples=5,
        data_observed_infections=model_sample.observed_infections.array,
        rng_key=jr.key(54),
    )

    posterior_predictive_samples = model.posterior_predictive(
        n_timepoints_to_simulate=n_timepoints_to_simulate + n_forecast_points,
    )

    # Check the length of the predictive distribution
    assert (
        len(posterior_predictive_samples["poisson_rv"][0])
        == n_timepoints_to_simulate + n_forecast_points
    )

    # Check the first elements of the posterior predictive Rt are the same as the
    # posterior Rt
    assert_array_equal(
        model.mcmc.get_samples()["Rt"][0],
        posterior_predictive_samples["Rt"][0][
            : len(model.mcmc.get_samples()["Rt"][0])
        ],
    )
