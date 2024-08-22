# numpydoc ignore=GL08

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from scipy.stats import kstest

from pyrenew.metaclass import (
    DistributionalVariable,
    SampledValue,
    StaticDistributionalVariable,
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
    element_rv = DistributionalVariable("el_rv", distribution=distribution)
    rseq = IIDRandomSequence(element_rv=element_rv)
    if distribution.batch_shape == () or distribution.batch_shape == (1,):
        expected_shape = (n,)
    else:
        expected_shape = tuple([n] + [x for x in distribution.batch_shape])

    with numpyro.handlers.seed(rng_seed=62):
        ans_vec, *_ = rseq.sample(n=n, vectorize=True)
        ans_serial, *_ = rseq.sample(n=n, vectorize=False)

    # check that samples are the right type
    for ans in [ans_serial, ans_vec]:
        assert isinstance(ans, SampledValue)
        # check that the samples are of the right shape
        assert ans.value.shape == expected_shape

    # vectorized and unvectorized sampling should
    # not give the same answer
    # but they should give similar distributions
    assert all(ans_serial.value.flatten() != ans_vec.value.flatten())

    if expected_shape == (n,):
        kstest_out = kstest(ans_serial.value, ans_vec.value)
        assert kstest_out.pvalue > 0.01


def test_standard_normal_sequence():
    """
    Test the StandardNormalSequence RandomVariable
    class.
    """
    norm_seq = StandardNormalSequence("test_norm_elements")

    # should be implemented with a DistributionalVariable
    # that is a standard normal
    assert isinstance(norm_seq.element_rv, StaticDistributionalVariable)
    assert isinstance(norm_seq.element_rv.distribution, dist.Normal)
    assert norm_seq.element_rv.distribution.loc == 0.0
    assert norm_seq.element_rv.distribution.scale == 1.0

    # should be sampleable
    with numpyro.handlers.seed(rng_seed=67):
        ans, *_ = norm_seq.sample(n=50000)

    assert isinstance(ans, SampledValue)
    # samples should be approximately standard normal
    kstest_out = kstest(ans.value, "norm", (0, 1))
    assert kstest_out.pvalue > 0.01
