"""
Tests for _assert_sample_and_rtype method
"""

import pytest
from numpy.testing import assert_equal
from pyrenew.metaclass import (
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

        return (SampledValue(value=1, t_start=self.t_start, t_unit=self.t_unit),)

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

        return (SampledValue(value=1, t_start=self.t_start, t_unit=self.t_unit),)

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

    with pytest.raises(Exception, match="None is not an instance of RandomVariable"):
        _assert_sample_and_rtype(None, skip_if_none=False)


def test_not_a_rv():  # numpydoc ignore=GL08
    with pytest.raises(Exception, match="is not an instance of RandomVariable"):
        _assert_sample_and_rtype(1)


def test_sample_return():  # numpydoc ignore=GL08
    """
    Test that RandomVariable has a sample method with return type tuple
    """

    rv1 = RVreturnsTuple()
    _assert_sample_and_rtype(rv1)

    rv2 = RVnoAnnotation()
    with pytest.raises(Exception, match="does not have return type annotation"):
        _assert_sample_and_rtype(rv2)
