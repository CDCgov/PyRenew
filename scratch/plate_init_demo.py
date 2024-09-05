# numpydoc ignore=GL08

import string

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax import random
from numpyro.infer import MCMC, NUTS

from pyrenew.latent import InitializeInfectionsExponentialGrowth
from pyrenew.randomvariable import DistributionalVariable

n_groups = 3  # works fine
# n_groups = 4 # totally fails
rates = 1 + jnp.pow(10.0, -(jnp.arange(n_groups) + 1))
n_timepoints = jnp.arange(n_groups) + 10
i0s = jnp.arange(n_groups) + 1


input_data = pl.DataFrame(
    {
        "group": pl.Series(
            np.array(list(string.ascii_lowercase))[
                np.repeat(np.arange(n_groups), n_timepoints)
            ],
            dtype=pl.Categorical,
        ),
        "time": np.concatenate(
            [np.arange(n_timepoint) for n_timepoint in n_timepoints]
        ),
        "obs": np.concatenate(
            [
                i0 * np.exp(rate * np.arange(n_timepoints))
                for rate, i0, n_timepoints in zip(rates, i0s, n_timepoints)
            ]
        ),
    }
).filter(~((pl.col("group") == "a") & (pl.col("time") == 4)))
# some implicitly missing data

y_group = input_data["group"].to_numpy()
y_time = input_data["time"].to_numpy()
y_obs = input_data["obs"].to_numpy()


# This would be done at model instantiation:
y_group_ind = input_data["group"].to_physical().to_numpy()
y_time_max = input_data["time"].max()

rate_rv = DistributionalVariable("rate", dist.HalfNormal())
i0_rv = DistributionalVariable("i0", dist.HalfNormal())


def my_model(y_group, y_time, y_obs):
    # numpydoc ignore=GL08
    with numpyro.plate("group", n_groups):
        i0 = i0_rv()[0].value
        mean_infec = InitializeInfectionsExponentialGrowth(
            n_timepoints=y_time_max + 1, rate_rv=rate_rv, t_pre_init=0
        )(i0)

    numpyro.sample("obs", dist.Poisson(mean_infec[y_time, y_group]), obs=y_obs)


# Posterior Sampling
nuts_kernel = NUTS(my_model, find_heuristic_step_size=True)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, y_group=y_group_ind, y_time=y_time, y_obs=y_obs)

# Check results
mcmc.print_summary()
