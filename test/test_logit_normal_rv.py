"""Tests for the LogitNormalVariable factory."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpy.testing import assert_allclose
from numpyro.infer import Predictive
from numpyro.infer.reparam import LocScaleReparam

import pyrenew.transformation as transformation
from pyrenew.randomvariable import (
    LogitNormalVariable,
    StaticDistributionalVariable,
    TransformedVariable,
)


def test_logit_normal_variable_construction():
    """The factory constructs the expected transformed Normal variable."""
    median = 0.004
    scale = 0.3

    rv = LogitNormalVariable(name="iedr", median=median, scale=scale)

    assert isinstance(rv, TransformedVariable)
    assert isinstance(rv.base_rv, StaticDistributionalVariable)
    assert isinstance(rv.base_rv.distribution, dist.Normal)
    assert_allclose(
        rv.base_rv.distribution.loc,
        transformation.SigmoidTransform().inv(median),
    )
    assert_allclose(rv.base_rv.distribution.scale, scale)
    assert len(rv.transforms) == 1
    assert isinstance(rv.transforms[0], transformation.SigmoidTransform)


def test_logit_normal_variable_samples_are_probabilities():
    """Samples from a logit-normal variable lie strictly between zero and one."""
    rv = LogitNormalVariable(name="iedr", median=0.004, scale=0.3)

    def model():  # numpydoc ignore=GL08
        return rv.sample()

    samples = Predictive(model, num_samples=100)(jax.random.key(0))
    values = samples["iedr"][0]

    assert (values > 0).all()
    assert (values < 1).all()


@pytest.mark.parametrize(
    ("base_name", "expected_base_name"),
    [
        (None, "logit_iedr"),
        ("iedr_unconstrained", "iedr_unconstrained"),
    ],
)
def test_logit_normal_variable_names(base_name, expected_base_name):
    """Default and explicit names are used for object and trace sites."""
    rv = LogitNormalVariable(
        name="iedr",
        median=0.004,
        scale=0.3,
        base_name=base_name,
    )

    assert rv.name == "iedr"
    assert rv.base_rv.name == expected_base_name

    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.trace() as trace:
        rv.sample(record=True)

    assert trace[expected_base_name]["type"] == "sample"
    assert trace["iedr"]["type"] == "deterministic"


def test_logit_normal_variable_reparameterization():
    """A reparameterizer is applied to the underlying Normal sample site."""
    reparam = LocScaleReparam(0)
    rv = LogitNormalVariable(
        name="iedr",
        median=0.004,
        scale=0.3,
        reparam=reparam,
    )

    assert rv.base_rv.reparam_dict == {"logit_iedr": reparam}

    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.trace() as trace:
        value = rv.sample()

    assert trace["logit_iedr_decentered"]["type"] == "sample"
    assert trace["logit_iedr"]["type"] == "deterministic"
    assert 0 < value < 1


@pytest.mark.parametrize(
    "median",
    [
        -0.1,
        0.0,
        1.0,
        1.1,
        float("-inf"),
        float("inf"),
        float("nan"),
        jnp.array([0.2, 1.0]),
    ],
)
def test_logit_normal_variable_rejects_invalid_median(median):
    """Medians must be finite and strictly inside the unit interval."""
    with pytest.raises(ValueError, match="median must contain only finite values"):
        LogitNormalVariable(name="invalid", median=median, scale=0.3)


@pytest.mark.parametrize(
    "scale",
    [
        -0.1,
        0.0,
        float("-inf"),
        float("inf"),
        float("nan"),
        jnp.array([0.3, 0.0]),
    ],
)
def test_logit_normal_variable_rejects_invalid_scale(scale):
    """Scales must be finite and strictly positive."""
    with pytest.raises(ValueError, match="scale must contain only finite positive"):
        LogitNormalVariable(name="invalid", median=0.4, scale=scale)
