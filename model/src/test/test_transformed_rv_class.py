# -*- coding: utf-8 -*-

"""
Tests for TransformedRandomVariable class
"""

import numpyro
import numpyro.distributions as dist
import pyrenew.transformation as t
import pytest
from numpy.testing import assert_almost_equal
from pyrenew.metaclass import (
    DistributionalRV,
    RandomVariable,
    SampledValue,
    TransformedRandomVariable,
)


class LengthTwoRV(RandomVariable):
    """
    Class for a RandomVariable
    with sample_length 2
    and values 1 and 5
    """

    def sample(self, **kwargs):
        """
        Deterministic sampling method
        that returns a length-2 tuple

        Returns
        -------
        tuple
           (SampledValue(1, t_start=self.t_start, t_unit=self.t_unit), SampledValue(5, t_start=self.t_start, t_unit=self.t_unit))
        """
        return (
            SampledValue(1, t_start=self.t_start, t_unit=self.t_unit),
            SampledValue(5, t_start=self.t_start, t_unit=self.t_unit),
        )

    def sample_length(self):
        """
        Report the sample length as 2

        Returns
        -------
        int
           2
        """
        return 2

    def validate(self):
        """
        No validation.

        Returns
        -------
        None
        """
        return None


def test_transform_rv_validation():
    """
    Test that a TransformedRandomVariable validation
    works as expected.
    """

    base_rv = DistributionalRV(name="test_normal", dist=dist.Normal(0, 1))
    base_rv.sample_length = lambda: 1  # numpydoc ignore=GL08

    l2_rv = LengthTwoRV()

    test_transforms = [t.IdentityTransform(), t.ExpTransform()]

    for tr in test_transforms:
        my_rv = TransformedRandomVariable("test_transformed_rv", base_rv, tr)
        assert isinstance(my_rv.transforms, tuple)
        assert len(my_rv.transforms) == 1
        assert my_rv.sample_length() == 1
        not_callable_err = "All entries in self.transforms " "must be callable"
        sample_length_err = "There must be exactly as many transformations"
        with pytest.raises(ValueError, match=sample_length_err):
            _ = TransformedRandomVariable(
                "should_error_due_to_too_many_transforms", base_rv, (tr, tr)
            )
        with pytest.raises(ValueError, match=sample_length_err):
            _ = TransformedRandomVariable(
                "should_error_due_to_too_few_transforms", l2_rv, tr
            )
        with pytest.raises(ValueError, match=sample_length_err):
            _ = TransformedRandomVariable(
                "should_also_error_due_to_too_few_transforms", l2_rv, (tr,)
            )
        with pytest.raises(ValueError, match=not_callable_err):
            _ = TransformedRandomVariable(
                "should_error_due_to_not_callable", l2_rv, (1,)
            )
        with pytest.raises(ValueError, match=not_callable_err):
            _ = TransformedRandomVariable(
                "should_error_due_to_not_callable", base_rv, (1,)
            )


def test_transforms_applied_at_sampling():
    """
    Test that TransformedRandomVariable
    instances correctly apply their specified
    transformations at sampling
    """
    norm_rv = DistributionalRV(name="test_normal", dist=dist.Normal(0, 1))
    norm_rv.sample_length = lambda: 1

    l2_rv = LengthTwoRV()

    for tr in [
        t.IdentityTransform(),
        t.ExpTransform(),
        t.ExpTransform().inv,
        t.ScaledLogitTransform(5),
    ]:
        tr_norm = TransformedRandomVariable("transformed_normal", norm_rv, tr)

        tr_l2 = TransformedRandomVariable(
            "transformed_length_2", l2_rv, (tr, t.ExpTransform())
        )

        with numpyro.handlers.seed(rng_seed=5):
            norm_base_sample = norm_rv.sample()
            l2_base_sample = l2_rv.sample()
        with numpyro.handlers.seed(rng_seed=5):
            norm_transformed_sample = tr_norm.sample()
            l2_transformed_sample = tr_l2.sample()

        assert_almost_equal(
            tr(norm_base_sample[0].value), norm_transformed_sample[0].value
        )
        assert_almost_equal(
            (
                tr(l2_base_sample[0].value),
                t.ExpTransform()(l2_base_sample[1].value),
            ),
            (l2_transformed_sample[0].value, l2_transformed_sample[1].value),
        )
