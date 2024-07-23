# numpydoc ignore=GL08

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpy.testing import assert_almost_equal
from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import DistributionalRV
from pyrenew.process import SimpleRandomWalkProcess


def test_rw_can_be_sampled():
    """
    Check that a simple random walk
    can be initialized and sampled from
    """
    init_rv_rand = DistributionalRV(dist.Normal(1, 0.5), "init_rv_rand")
    init_rv_fixed = DeterministicVariable(50.0, "init_rv_fixed")

    step_rv = DistributionalRV(dist.Normal(0, 1), "rw_step")

    rw_init_rand = SimpleRandomWalkProcess(
        "rw_rand_init", step_rv=step_rv, init_rv=init_rv_rand
    )

    rw_init_fixed = SimpleRandomWalkProcess(
        "rw_fixed_init", step_rv=step_rv, init_rv=init_rv_fixed
    )

    with numpyro.handlers.seed(rng_seed=62):
        # can sample with a fixed init
        # and with a random init
        ans_rand = rw_init_rand(n_steps=3532)
        ans_fixed = rw_init_fixed(n_steps=5023)

    # check that the samples are of the right shape
    assert ans_rand[0].value.shape == (3532,)
    assert ans_fixed[0].value.shape == (5023,)

    # check that fixing inits works
    assert_almost_equal(ans_fixed[0].value[0], init_rv_fixed.vars)
    assert ans_rand[0].value[0] != init_rv_fixed.vars


def test_rw_samples_correctly_distributed():
    """
    Check that a simple random walk has steps
    distributed according to the target distribution
    """

    n_samples = 10000
    for step_mean, step_sd in zip(
        [0, 2.253, -3.2521, 1052, 1e-6], [1, 0.025, 3, 1, 0.02]
    ):
        rw_init_val = 532.0
        rw_normal = SimpleRandomWalkProcess(
            name="rw_normal_test",
            step_rv=DistributionalRV(
                dist=dist.Normal(loc=step_mean, scale=step_sd),
                name="rw_normal_dist",
            ),
            init_rv=DeterministicVariable(rw_init_val, "init_rv_fixed"),
        )

        with numpyro.handlers.seed(rng_seed=62):
            samples, *_ = rw_normal(n_steps=n_samples)
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
