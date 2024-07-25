# numpydoc ignore=GL08

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pyrenew.transformation as t
from numpy.testing import assert_array_equal
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
    rt = TransformedRandomVariable(
        "Rt_rv",
        base_rv=SimpleRandomWalkProcess(
            name="log_rt",
            step_rv=DistributionalRV(dist.Normal(0, 0.025), "rw_step_rv"),
            init_rv=DistributionalRV(dist.Normal(0, 0.2), "init_log_Rt_rv"),
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

    n_datapoints = 30
    n_forecast_points = 10
    with numpyro.handlers.seed(rng_seed=223):
        model_sample = model.sample(n_datapoints=n_datapoints)

    model.run(
        num_warmup=5,
        num_samples=5,
        data_observed_infections=model_sample.observed_infections,
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
