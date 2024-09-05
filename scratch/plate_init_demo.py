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

groups = jnp.arange(4)
min_timepoints = 10
rates = 1 + (groups + 2) * 0.01
n_timepoints = groups + min_timepoints
i0s = 1 + (groups + 1) * 0.01


input_data = pl.DataFrame(
    {
        "group": pl.Series(
            np.array(list(string.ascii_lowercase))[
                np.repeat(groups, n_timepoints)
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
    with numpyro.plate("group", len(groups)):
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
