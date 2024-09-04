import jax.numpy as jnp
import jax
import numpyro
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from pyrenew.metaclass import CensoredNormal


@pytest.mark.parametrize(
    ["loc", "scale", "lower_limit", "upper_limit", "in_val", "l_val", "h_val"],
    [0, 1, -1, 1, jnp.array([0, 0.5]), -2, 2],
)
def test_censored_normal_distribution(
    loc, scale, lower_limit, upper_limit, in_val, l_val, h_val
):  # numpydoc ignore=GL08
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
