# numpydoc ignore=GL08

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from pyrenew.metaclass import DistributionalRV
from pyrenew.process import IIDRandomSequence


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
        ans = rseq.sample(n=n)
    # check that the samples are of the right shape
    expected_shape = distribution.batch_shape
    if expected_shape == () or expected_shape == (1,):
        expected_shape = (n,)
    else:
        expected_shape = tuple([n] + [x for x in expected_shape])
    assert ans[0].value.shape == expected_shape
