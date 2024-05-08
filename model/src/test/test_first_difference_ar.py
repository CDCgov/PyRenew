# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax.numpy as jnp
import numpyro
from pyrenew.process import FirstDifferenceARProcess


def test_fd_ar_can_be_sampled():
    """
    Check that stochastic process
    with AR(1) first differences
    can be initialized and sampled
    from
    """
    ar_fd = FirstDifferenceARProcess(0.5, 0.5)

    with numpyro.handlers.seed(rng_seed=62):
        # can sample with and without inits
        # for the rate of change
        ar_fd.sample(3532, init_val=jnp.array([50.0]))
        ar_fd.sample(
            3532,
            init_val=jnp.array([50.0]),
            init_rate_of_change=jnp.array([0.25]),
        )
