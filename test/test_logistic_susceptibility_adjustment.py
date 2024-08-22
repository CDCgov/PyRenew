# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax.numpy as jnp

from pyrenew.latent import logistic_susceptibility_adjustment


def test_logistic_susceptibility_adjustment():  # numpydoc ignore=GL08
    new_I_raw = 1000000
    population = 100

    assert (
        logistic_susceptibility_adjustment(new_I_raw, 1, population)
        == population
    )

    assert logistic_susceptibility_adjustment(new_I_raw, 0, population) == 0

    new_I_raw = 7.2352
    assert (
        logistic_susceptibility_adjustment(new_I_raw, 0.75, population)
        == (1 - jnp.exp(-new_I_raw / population)) * 0.75 * population
    )
