# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from test.utils import simple_rt

import jax.numpy as jnp
import numpy.testing as testing
import numpyro
import pytest
from pyrenew.latent import Infections


def test_infections_as_deterministic():
    """
    Test that the Infections class samples the same infections when
    the same seed is used.
    """

    rt = simple_rt()

    with numpyro.handlers.seed(rng_seed=223):
        sim_rt, *_ = rt(n_steps=30)

    gen_int = jnp.array([0.25, 0.25, 0.25, 0.25])

    inf1 = Infections()

    obs = dict(
        Rt=sim_rt.value,
        I0=jnp.zeros(gen_int.size),
        gen_int=gen_int,
    )
    with numpyro.handlers.seed(rng_seed=223):
        inf_sampled1 = inf1(**obs)
        inf_sampled2 = inf1(**obs)

    testing.assert_array_equal(
        inf_sampled1.post_initialization_infections.value,
        inf_sampled2.post_initialization_infections.value,
    )

    # Check that Initial infections vector must be at least as long as the generation interval.
    with numpyro.handlers.seed(rng_seed=223):
        with pytest.raises(ValueError):
            obs["I0"] = jnp.array([1])
            inf1(**obs)
