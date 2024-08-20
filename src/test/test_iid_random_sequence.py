# numpydoc ignore=GL08

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from scipy.stats import kstest

from pyrenew.metaclass import (
    DistributionalRV,
    SampledValue,
    StaticDistributionalRV,
)
from pyrenew.process import IIDRandomSequence, StandardNormalSequence


@pytest.mark.parametrize(
    ["distribution", "n"],
    [
        [dist.Normal(0, 1), 1000],
        [dist.Cauchy(2, 325.0), 13532],
        [dist.Normal(jnp.array([2.0, 3.0, -5.235]), 0.25), 622],
    ],
)
def test_iidrandomsequence_with_dist_rv(distribution, n):
    """
    Check that an IIDRandomSequence can be
    initialized and sampled from when the element_rv is
    a distributional RV, including with array-valued
    distributions
    """
    element_rv = DistributionalRV("el_rv", distribution=distribution)
    rseq = IIDRandomSequence(name="randseq", element_rv=element_rv)

    with numpyro.handlers.seed(rng_seed=62):
        ans, *_ = rseq.sample(n=n)

    # check that samples are the right type
    assert isinstance(ans, SampledValue)

    # check that the samples are of the right shape
    expected_shape = distribution.batch_shape
    if expected_shape == () or expected_shape == (1,):
        expected_shape = (n,)
    else:
        expected_shape = tuple([n] + [x for x in expected_shape])
    assert ans.value.shape == expected_shape


def test_standard_normal_sequence():
    """
    Test the StandardNormalSequence RandomVariable
    class.
    """
    norm_seq = StandardNormalSequence("test_norm")

    # should be implemented with a DistributionalRV
    # that is a standard normal
    assert isinstance(norm_seq.element_rv, StaticDistributionalRV)
    assert isinstance(norm_seq.element_rv.distribution, dist.Normal)
    assert norm_seq.element_rv.distribution.loc == 0
    assert norm_seq.element_rv.distribution.scale == 1

    # should be sampleable
    with numpyro.handlers.seed(rng_seed=67):
        ans, *_ = norm_seq.sample(n=50000)

    assert isinstance(ans, SampledValue)
    # samples should be approximately standard normal
    kstest_out = kstest(ans.value, "norm", (0, 1))
    assert kstest_out.pvalue > 0.001
