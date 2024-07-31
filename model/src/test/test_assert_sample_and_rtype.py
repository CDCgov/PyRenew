"""
Tests for _assert_sample_and_rtype method
"""

import re

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpy.testing import assert_equal
from pyrenew.deterministic import DeterministicVariable, NullObservation
from pyrenew.metaclass import (
    DistributionalRV,
    RandomVariable,
    SampledValue,
    _assert_sample_and_rtype,
)


class RVreturnsTuple(RandomVariable):
    """
    Class for a RandomVariable with
    sample value 1
    """

    def sample(self, **kwargs) -> tuple:
        """
        Deterministic sampling method that returns 1

        Returns
        -------
        (
        SampledValue(1, t_start=self.t_start, t_unit=self.t_unit),
        )
        """

        return (
            SampledValue(value=1, t_start=self.t_start, t_unit=self.t_unit),
        )

    def validate(self):
        """
        No validation.

        Returns
        -------
        None
        """
        return None


class RVnoAnnotation(RandomVariable):
    """
    Class for a RandomVariable with
    sample value 1
    """

    def sample(self, **kwargs):
        """
        Deterministic sampling method that returns 1

        Returns
        -------
        (
        SampledValue(1, t_start=self.t_start, t_unit=self.t_unit),
        )
        """

        return (
            SampledValue(value=1, t_start=self.t_start, t_unit=self.t_unit),
        )

    def validate(self):
        """
        No validation.

        Returns
        -------
        None
        """
        return None


def test_none_rv():  # numpydoc ignore=GL08
    assert_equal(_assert_sample_and_rtype(None), None)

    with pytest.raises(
        Exception, match="None is not an instance of RandomVariable"
    ):
        _assert_sample_and_rtype(None, skip_if_none=False)


def test_input_rv():  # numpydoc ignore=GL08
    valid_rv = [
        NullObservation(),
        DeterministicVariable(name="rv1", value=jnp.array([1, 2, 3, 4])),
        DistributionalRV(name="rv2", dist=dist.Normal(0, 1)),
    ]
    not_rv = jnp.array([1])

    for rv in valid_rv:
        _assert_sample_and_rtype(rv)

    with pytest.raises(
        Exception,
        match=re.escape(f"{not_rv} is not an instance of RandomVariable"),
    ):
        _assert_sample_and_rtype(not_rv)


def test_sample_return():  # numpydoc ignore=GL08
    """
    Test that RandomVariable has a sample method with return type tuple
    """

    rv3 = RVreturnsTuple()
    _assert_sample_and_rtype(rv3)

    rv4 = RVnoAnnotation()
    with pytest.raises(
        Exception,
        match=f"The RandomVariable {rv4} does not have return type annotation",
    ):
        _assert_sample_and_rtype(rv4)
