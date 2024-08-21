# numpydoc ignore=GL08

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from pyrenew.metaclass import compute_incidence_observed_with_delay


@pytest.mark.parametrize(
    ["obs_rate", "latent_incidence", "delay_interval", "expected_output"],
    [
        [
            jnp.array([1.0]),
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([1.0]),
            jnp.array([1.0, 2.0, 3.0]),
        ],
    ],
)
def test(obs_rate, latent_incidence, delay_interval, expected_output):
    """
    Tests for helper function to compute
    incidence observed with a delay
    """
    result = compute_incidence_observed_with_delay(
        obs_rate,
        latent_incidence,
        delay_interval,
    )

    assert_array_equal(result, expected_output)
