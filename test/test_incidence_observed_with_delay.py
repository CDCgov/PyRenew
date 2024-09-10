# numpydoc ignore=GL08

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from pyrenew.convolve import compute_delay_ascertained_incidence


@pytest.mark.parametrize(
    ["obs_rate", "latent_incidence", "delay_interval", "expected_output"],
    [
        [
            jnp.array([1.0]),
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([1.0]),
            jnp.array([1.0, 2.0, 3.0]),
        ],
        [
            jnp.array([1.0, 0.1, 1.0]),
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([1.0]),
            jnp.array([1.0, 0.2, 3.0]),
        ],
        [
            jnp.array([1.0]),
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([0.5, 0.5]),
            jnp.array([1.5, 2.5]),
        ],
        [
            jnp.array([1.0]),
            jnp.array([0, 2.0, 4.0]),
            jnp.array([0.25, 0.5, 0.25]),
            jnp.array([2]),
        ],
        [
            jnp.array([1.0]),
            jnp.array([0, 2.0, 4.0]),
            jnp.array([0.25, 0.5, 0.25]),
            jnp.array([2]),
        ],
    ],
)
def test(obs_rate, latent_incidence, delay_interval, expected_output):
    """
    Tests for helper function to compute
    incidence observed with a delay
    """
    result = compute_delay_ascertained_incidence(
        latent_incidence,
        delay_interval,
        obs_rate,
    )
    assert_array_equal(result, expected_output)


def test_default_obs_rate():
    """
    Compute incidence observed with a delay and default observation rate
    """
    result = compute_delay_ascertained_incidence(
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array([1.0]),
    )
    assert_array_equal(result, jnp.array([1.0, 2.0, 3.0]))
