# numpydoc ignore=GL08

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpy.testing import assert_almost_equal
from pyrenew.process import SimpleRandomWalkProcess


def test_rw_can_be_sampled():
    """
    Check that a simple random walk
    can be initialized and sampled from
    """
    rw_normal = SimpleRandomWalkProcess(dist.Normal(0, 1))

    with numpyro.handlers.seed(rng_seed=62):
        # can sample with and without inits
        ans0 = rw_normal.sample(3532, init=50.0)
        ans1 = rw_normal.sample(5023)

        # check that the samples are of the right shape
        assert ans0[0].shape == (3532,)
        assert ans1[0].shape == (5023,)


def test_rw_samples_correctly_distributed():
    """
    Check that a simple random walk has steps
    distributed according to the target distribution
    """

    n_samples = 10000
    for step_mean, step_sd in zip(
        [0, 2.253, -3.2521, 1052, 1e-6], [1, 0.025, 3, 1, 0.02]
    ):
        rw_normal = SimpleRandomWalkProcess(dist.Normal(step_mean, step_sd))
        rw_init = 532.0
        with numpyro.handlers.seed(rng_seed=62):
            samples, *_ = rw_normal.sample(n_samples, init=rw_init)

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
            assert_almost_equal(samples[0], rw_init)
