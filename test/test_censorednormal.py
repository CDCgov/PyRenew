# numpydoc ignore=GL08

import jax
import jax.numpy as jnp
import numpyro
import pytest
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

from pyrenew.distributions import CensoredNormal


@pytest.mark.parametrize(
    ["loc", "scale", "lower_limit", "upper_limit", "in_val", "l_val", "h_val"],
    [
        [
            jnp.array([0]),
            jnp.array([2.0, 1.0]),
            -1,
            1,
            jnp.array([0, 0.5]),
            -2,
            2,
        ],
        [
            jnp.array([0, 1]),
            jnp.array([1.0]),
            -1,
            2,
            jnp.array([0, 0.5]),
            -2,
            3,
        ],
    ],
)
def test_interval_censored_normal_distribution(
    loc,
    scale,
    lower_limit,
    upper_limit,
    in_val,
    l_val,
    h_val,
):
    """
    Tests the censored normal distribution samples
    within the limit and calculation of log probability
    """
    censored_dist = CensoredNormal(
        loc=loc, scale=scale, lower_limit=lower_limit, upper_limit=upper_limit
    )
    normal_dist = numpyro.distributions.Normal(loc=loc, scale=scale)

    # test samples within the bounds
    samp = censored_dist.sample(jax.random.PRNGKey(0), sample_shape=(100,))
    assert jnp.all(samp >= lower_limit)
    assert jnp.all(samp <= upper_limit)

    # test log prob of values within bounds
    assert_array_equal(censored_dist.log_prob(in_val), normal_dist.log_prob(in_val))

    # test log prob of values lower than the limit
    assert_array_almost_equal(
        censored_dist.log_prob(l_val),
        jax.scipy.special.log_ndtr((lower_limit - loc) / scale),
    )

    # test log prob of values higher than the limit
    assert_array_almost_equal(
        censored_dist.log_prob(h_val),
        jax.scipy.special.log_ndtr(-(upper_limit - loc) / scale),
    )

    # test_broadcasting
    assert_equal(samp.shape[-1], max(loc.shape[0], scale.shape[0]))

    # test support of the distribution
    assert_equal(censored_dist.support.lower_bound, lower_limit)
    assert_equal(censored_dist.support.upper_bound, upper_limit)


@pytest.mark.parametrize(
    ["loc", "scale", "lower_limit", "in_val", "l_val"],
    [
        [0, 1, -5, jnp.array([-2, 1]), -6],
    ],
)
def test_left_censored_normal_distribution(
    loc,
    scale,
    lower_limit,
    in_val,
    l_val,
):
    """
    Tests the lower censored normal distribution samples
    within the limit and calculation of log probability
    """
    censored_dist = CensoredNormal(
        loc=loc,
        scale=scale,
        lower_limit=lower_limit,
    )
    normal_dist = numpyro.distributions.Normal(loc=loc, scale=scale)

    # test samples within the bounds
    samp = censored_dist.sample(jax.random.PRNGKey(0), sample_shape=(100,))
    assert jnp.all(samp >= lower_limit)

    # test log prob of values within bounds
    assert_array_equal(censored_dist.log_prob(in_val), normal_dist.log_prob(in_val))

    # test log prob of values lower than the limit
    assert_array_almost_equal(
        censored_dist.log_prob(l_val),
        jax.scipy.special.log_ndtr((lower_limit - loc) / scale),
    )

    # test support of the distribution
    assert_equal(censored_dist.support.lower_bound, lower_limit)
    assert censored_dist.support.upper_bound == jnp.inf


@pytest.mark.parametrize(
    ["loc", "scale", "upper_limit", "in_val", "h_val"],
    [
        [0, 1, 3, jnp.array([1, 2]), 5],
    ],
)
def test_right_censored_normal_distribution(
    loc,
    scale,
    upper_limit,
    in_val,
    h_val,
):
    """
    Tests the upper censored normal distribution samples
    within the limit and calculation of log probability
    """
    censored_dist = CensoredNormal(loc=loc, scale=scale, upper_limit=upper_limit)
    normal_dist = numpyro.distributions.Normal(loc=loc, scale=scale)

    # test samples within the bounds
    samp = censored_dist.sample(jax.random.PRNGKey(0), sample_shape=(100,))
    assert jnp.all(samp <= upper_limit)

    # test log prob of values within bounds
    assert_array_equal(censored_dist.log_prob(in_val), normal_dist.log_prob(in_val))

    # test log prob of values higher than the limit
    assert_array_almost_equal(
        censored_dist.log_prob(h_val),
        jax.scipy.special.log_ndtr(-(upper_limit - loc) / scale),
    )

    # test support of the distribution
    assert_equal(censored_dist.support.upper_bound, upper_limit)
    assert censored_dist.support.lower_bound == -jnp.inf
