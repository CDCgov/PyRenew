# numpydoc ignore=GL08

from contextlib import nullcontext

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


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize(
    ["loc", "scale"],
    [
        [jnp.array([0]), jnp.array([2.5, 1.0])],
        [
            jnp.array([0, 1]),
            jnp.array([1.0]),
        ],
    ],
)
@pytest.mark.parametrize(
    ["lower_limit", "upper_limit", "inbounds_test_val"],
    [
        [
            -1.5552,
            1.0235235,
            jnp.array([0, 0.5]),
        ],
        [
            -1,
            2,
            jnp.array([0, 0.5]),
        ],
    ],
)
def test_censored_normal_distribution(
    loc, scale, lower_limit, upper_limit, inbounds_test_val, validate
):
    """
    Test forward sampling, support, and log probability calculations
    for the CensoredNormal, with and without validation.
    """
    censored_dist = CensoredNormal(
        loc=loc,
        scale=scale,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        validate_args=validate,
    )
    normal_dist = numpyro.distributions.Normal(loc=loc, scale=scale)

    lb_vals = jnp.full_like(inbounds_test_val, lower_limit)
    ub_vals = jnp.full_like(inbounds_test_val, upper_limit)

    oob_vals_lower = jnp.nextafter(lb_vals, -jnp.inf)
    oob_vals_upper = jnp.nextafter(ub_vals, jnp.inf)

    expected_lp_inbounds = normal_dist.log_prob(inbounds_test_val)
    expected_lp_lb = jax.scipy.special.log_ndtr((lb_vals - loc) / scale)
    expected_lp_ub = jax.scipy.special.log_ndtr(-(ub_vals - loc) / scale)
    if validate:
        expected_lp_oob_low = jnp.full_like(expected_lp_lb, -jnp.inf)
        expected_lp_oob_high = jnp.full_like(expected_lp_ub, -jnp.inf)
        oob_context = pytest.warns(match="Out-of-support")
    else:
        expected_lp_oob_low = expected_lp_lb
        expected_lp_oob_high = expected_lp_ub
        oob_context = nullcontext()

    # test samples within the bounds
    samp = censored_dist.sample(jax.random.PRNGKey(0), sample_shape=(100,))
    assert jnp.all(samp >= lower_limit)
    assert jnp.all(samp <= upper_limit)

    # test log prob of values within bounds
    assert_array_equal(censored_dist.log_prob(inbounds_test_val), expected_lp_inbounds)

    # test log prob of values at the lower limit
    assert_array_almost_equal(censored_dist.log_prob(lb_vals), expected_lp_lb)

    # test log prob of values at the upper limit
    assert_array_almost_equal(censored_dist.log_prob(ub_vals), expected_lp_ub)

    # out of bounds values should return -inf log prob
    # with a warning if validation enabled, otherwise
    # should return limit values

    with oob_context:
        assert_array_almost_equal(
            censored_dist.log_prob(oob_vals_lower), expected_lp_oob_low
        )

    with oob_context:
        assert_array_almost_equal(
            censored_dist.log_prob(oob_vals_upper), expected_lp_oob_high
        )

    # log prob of a mix of in, at, and oob values should mix -infs
    # and correct inbounds values, with a warning
    test_vals = jnp.vstack(
        [oob_vals_lower, jnp.array(inbounds_test_val), lb_vals, oob_vals_upper, ub_vals]
    )
    expected_lps = jnp.vstack(
        [
            expected_lp_oob_low,
            expected_lp_inbounds,
            expected_lp_lb,
            expected_lp_oob_high,
            expected_lp_ub,
        ],
    )

    with oob_context:
        assert_array_almost_equal(censored_dist.log_prob(test_vals), expected_lps)

    # test_broadcasting
    assert_equal(samp.shape[-1], max(loc.shape[0], scale.shape[0]))

    # test support of the distribution
    assert_equal(censored_dist.support.lower_bound, lower_limit)
    assert_equal(censored_dist.support.upper_bound, upper_limit)


@pytest.mark.parametrize(
    ["loc", "scale", "lower_limit"],
    [
        [0, 1, -5.000001],
    ],
)
def test_left_censored_normal_distribution(
    loc,
    scale,
    lower_limit,
):
    """
    If only lower (left) limit provided, upper (right) limit is inf
    """
    censored_dist = CensoredNormal(
        loc=loc,
        scale=scale,
        lower_limit=lower_limit,
    )
    assert_equal(censored_dist.support.lower_bound, lower_limit)
    assert censored_dist.support.upper_bound == jnp.inf


@pytest.mark.parametrize(
    ["loc", "scale", "upper_limit"],
    [
        [0, 1, jnp.array([1.0353, 2.52])],
    ],
)
def test_right_censored_normal_distribution(loc, scale, upper_limit):
    """
    If only upper (right) limit provided, the lower (left) limit is -inf
    """
    censored_dist = CensoredNormal(loc=loc, scale=scale, upper_limit=upper_limit)
    assert_array_equal(censored_dist.support.upper_bound, upper_limit)
    assert censored_dist.support.lower_bound == -jnp.inf
