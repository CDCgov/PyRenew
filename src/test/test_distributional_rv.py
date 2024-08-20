"""
Tests for the distributional RV classes
"""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpy.testing import assert_array_equal

from pyrenew.metaclass import (
    DistributionalRV,
    DynamicDistributionalRV,
    StaticDistributionalRV,
)


class NonCallableTestClass:
    """
    Generic non-callable object to test
    callable checking for DynamicDistributionalRV.
    """

    def __init__(self):
        """
        Initialization method for generic non-callable
        object
        """
        pass


@pytest.mark.parametrize("not_a_dist", [1, "test", NonCallableTestClass()])
def test_invalid_constructor_args(not_a_dist):
    """
    Test that the constructor errors
    appropriately when given incorrect input
    """

    with pytest.raises(
        ValueError, match="distribution argument to DistributionalRV"
    ):
        DistributionalRV(name="this should fail", distribution=not_a_dist)
    with pytest.raises(
        ValueError,
        match=(
            "distribution should be an instance of "
            "numpyro.distributions.Distribution"
        ),
    ):
        StaticDistributionalRV.validate(not_a_dist)
    with pytest.raises(ValueError, match="must provide a Callable"):
        DynamicDistributionalRV.validate(not_a_dist)


@pytest.mark.parametrize(
    ["valid_static_dist_arg", "valid_dynamic_dist_arg"],
    [
        [dist.Normal(0, 1), dist.Normal],
        [dist.Cauchy(3.0, 5.0), dist.Cauchy],
        [dist.Poisson(0.25), dist.Poisson],
    ],
)
def test_factory_triage(valid_static_dist_arg, valid_dynamic_dist_arg):
    """
    Test that passing a numpyro.distributions.Distribution
    instance to the DistributionalRV factory instaniates
    a StaticDistributionalRV, while passing a callable
    instaniates a DynamicDistributionalRV
    """
    static = DistributionalRV(
        name="test static", distribution=valid_static_dist_arg
    )
    assert isinstance(static, StaticDistributionalRV)
    dynamic = DistributionalRV(
        name="test dynamic", distribution=valid_dynamic_dist_arg
    )
    assert isinstance(dynamic, DynamicDistributionalRV)


@pytest.mark.parametrize(
    ["dist", "params"],
    [
        [dist.Normal, {"loc": 0.0, "scale": 0.5}],
        [dist.Poisson, {"rate": 0.35265}],
        [
            dist.Cauchy,
            {
                "loc": jnp.array([1.0, 5.0, -0.25]),
                "scale": jnp.array([0.02, 0.15, 2]),
            },
        ],
    ],
)
def test_sampling_equivalent(dist, params):
    """
    Test that sampling a DynamicDistributionalRV
    with a given parameterization is equivalent to
    sampling a StaticDistributionalRV with the
    same parameterization and the same random seed
    """
    static = DistributionalRV(name="static", distribution=dist(**params))
    dynamic = DistributionalRV(name="dynamic", distribution=dist)
    assert isinstance(static, StaticDistributionalRV)
    assert isinstance(dynamic, DynamicDistributionalRV)
    with numpyro.handlers.seed(rng_seed=5):
        static_samp, *_ = static()
    with numpyro.handlers.seed(rng_seed=5):
        dynamic_samp, *_ = dynamic(**params)
    assert_array_equal(static_samp.value, dynamic_samp.value)
