# numpydoc ignore=GL08

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import DistributionalRV
from pyrenew.process import RandomWalk


@pytest.mark.parametrize("init_value", [50.0, -3, jnp.array(3)])
def test_rw_can_be_sampled(init_value):
    """
    Check that a simple random walk
    can be initialized and sampled from
    """
    init_rv = DeterministicVariable(name="init_rv_fixed", value=init_value)

    rw = RandomWalk(
        "rw",
        DistributionalRV(name="rw_step_rv", distribution=dist.Normal(0, 0.27)),
    )

    with numpyro.handlers.seed(rng_seed=62):
        # can sample with a fixed init
        # and with a random init
        init_vals = init_rv()[0].value
        ans_long = rw(n=5023, init_vals=init_vals)
        ans_short = rw(n=1, init_vals=init_vals)

        # Providing more than one init val should
        # raise an error.
        with pytest.raises(ValueError, match="differencing order"):
            rw(n=523, init_vals=jnp.hstack([init_vals, 0.25]))
    # check that the samples are of the right shape
    assert ans_long[0].value.shape == (5023,)
    assert ans_short[0].value.shape == (1,)
    # check that the first n_inits samples are the inits
    n_inits = jnp.atleast_1d(init_vals).size
    assert_array_almost_equal(
        ans_long[0].value[0:n_inits], jnp.atleast_1d(init_vals)
    )
    assert_array_almost_equal(
        ans_short[0].value, jnp.atleast_1d(init_vals)[:1]
    )


@pytest.mark.parametrize(
    ["step_mean", "step_sd"],
    [
        [0, 1],
        [2.253, 0.025],
        [-3.2521, 1],
        [1052, 3],
        [1e-6, 0.02],
    ],
)
def test_rw_samples_correctly_distributed(step_mean, step_sd):
    """
    Check that a simple random walk has steps
    distributed according to the target distribution
    """

    n_samples = 10000
    rw_init_val = jnp.array([532.0])
    rw_normal = RandomWalk(
        name="rw_normal_test",
        step_rv=DistributionalRV(
            name="rw_step_dist",
            distribution=dist.Normal(loc=step_mean, scale=step_sd),
        ),
    )

    with numpyro.handlers.seed(rng_seed=62):
        samples, *_ = rw_normal(n=n_samples, init_vals=rw_init_val)
        samples = samples.value

        # Checking the shape
        assert samples.shape == (n_samples,)

        # diffs should not be greater than
        # 5 sigma
        diffs = jnp.diff(samples)
        assert jnp.all(jnp.abs(diffs - step_mean) < 5 * step_sd)

        # sample mean of diffs should be
        # approximately equal to the
        # step mean, according to
        # the Law of Large Numbers
        deviation_threshold = 4 * jnp.sqrt((step_sd**2) / n_samples)
        assert jnp.abs(jnp.mean(diffs) - step_mean) < deviation_threshold

        # sample sd of diffs
        # should be approximately equal
        # to the step sd
        assert jnp.abs(jnp.log(jnp.std(diffs) / step_sd)) < jnp.log(1.1)

        # first value should be the init value
        assert_almost_equal(samples[0], rw_init_val)
