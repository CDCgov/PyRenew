"""
Tests for the distributional RV classes
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpy.testing import assert_array_equal
from numpyro.distributions import ExpandedDistribution

from pyrenew.metaclass import (
    DistributionalVariable,
    DynamicDistributionalVariable,
    StaticDistributionalVariable,
)


class NonCallableTestClass:
    """
    Generic non-callable object to test
    callable checking for DynamicDistributionalVariable.
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
        ValueError, match="distribution argument to DistributionalVariable"
    ):
        DistributionalVariable(
            name="this should fail", distribution=not_a_dist
        )
    with pytest.raises(
        ValueError,
        match=(
            "distribution should be an instance of "
            "numpyro.distributions.Distribution"
        ),
    ):
        StaticDistributionalVariable.validate(not_a_dist)
    with pytest.raises(ValueError, match="must provide a Callable"):
        DynamicDistributionalVariable.validate(not_a_dist)


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
    instance to the DistributionalVariable factory instaniates
    a StaticDistributionalVariable, while passing a callable
    instaniates a DynamicDistributionalVariable
    """
    static = DistributionalVariable(
        name="test static", distribution=valid_static_dist_arg
    )
    assert isinstance(static, StaticDistributionalVariable)
    dynamic = DistributionalVariable(
        name="test dynamic", distribution=valid_dynamic_dist_arg
    )
    assert isinstance(dynamic, DynamicDistributionalVariable)


@pytest.mark.parametrize(
    ["dist", "params", "expand_by_shape"],
    [
        [dist.Normal, {"loc": 0.0, "scale": 0.5}, (5,)],
        [dist.Poisson, {"rate": 0.35265}, (20, 25)],
        [
            dist.Cauchy,
            {
                "loc": jnp.array([1.0, 5.0, -0.25]),
                "scale": jnp.array([0.02, 0.15, 2]),
            },
            (10, 10, 3),
        ],
    ],
)
def test_expand_by(dist, params, expand_by_shape):
    """
    Test the expand_by method for static
    distributional RVs.
    """
    static = DistributionalVariable(name="static", distribution=dist(**params))
    dynamic = DistributionalVariable(name="dynamic", distribution=dist)
    expanded_static = static.expand_by(expand_by_shape)
    expanded_dynamic = dynamic.expand_by(expand_by_shape)

    assert isinstance(expanded_dynamic, DynamicDistributionalVariable)
    assert dynamic.expand_by_shape is None
    assert isinstance(expanded_dynamic.expand_by_shape, tuple)
    assert expanded_dynamic.expand_by_shape == expand_by_shape
    assert dynamic.reparam_dict == expanded_dynamic.reparam_dict
    assert (
        dynamic.distribution_constructor
        == expanded_dynamic.distribution_constructor
    )

    assert isinstance(expanded_static, StaticDistributionalVariable)
    assert isinstance(expanded_static.distribution, ExpandedDistribution)
    assert expanded_static.distribution.batch_shape == (
        expand_by_shape + static.distribution.batch_shape
    )

    with pytest.raises(ValueError):
        dynamic.expand_by("not a tuple")
    with pytest.raises(ValueError):
        static.expand_by("not a tuple")


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
    Test that sampling a DynamicDistributionalVariable
    with a given parameterization is equivalent to
    sampling a StaticDistributionalVariable with the
    same parameterization and the same random seed
    """
    static = DistributionalVariable(name="static", distribution=dist(**params))
    dynamic = DistributionalVariable(name="dynamic", distribution=dist)
    assert isinstance(static, StaticDistributionalVariable)
    assert isinstance(dynamic, DynamicDistributionalVariable)
    with numpyro.handlers.seed(rng_seed=5):
        static_samp, *_ = static()
    with numpyro.handlers.seed(rng_seed=5):
        dynamic_samp, *_ = dynamic(**params)
    assert_array_equal(static_samp.value, dynamic_samp.value)
