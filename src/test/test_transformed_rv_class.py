# -*- coding: utf-8 -*-

"""
Tests for TransformedRandomVariable class
"""

from typing import NamedTuple

import jax
import numpyro
import numpyro.distributions as dist
import pytest
from numpy.testing import assert_almost_equal

import pyrenew.transformation as t
from pyrenew.metaclass import (
    DistributionalRV,
    Model,
    RandomVariable,
    SampledValue,
    TransformedRandomVariable,
)


class LengthTwoRV(RandomVariable):
    """
    Class for a RandomVariable
    with sample_length 2
    """

    def sample(self, **kwargs):
        """
        Sampling method
        that returns a length-2 tuple

        Returns
        -------
        tuple
           (SampledValue(val, t_start=self.t_start, t_unit=self.t_unit),
            SampledValue(val, t_start=self.t_start, t_unit=self.t_unit))
        """
        val = numpyro.sample("my_normal", dist.Normal(0, 1))
        return (
            SampledValue(val, t_start=self.t_start, t_unit=self.t_unit),
            SampledValue(val, t_start=self.t_start, t_unit=self.t_unit),
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


class RVSamples(NamedTuple):
    """
    A container to hold the output of `NamedBaseRV()`.
    """

    rv1: SampledValue | None = None
    rv2: SampledValue | None = None

    def __repr__(self):
        return f"RVSamples(rv1={self.rv1},rv2={self.rv2})"


class NamedBaseRV(RandomVariable):
    """
    Class for a RandomVariable
    returning NamedTuples "rv1", and "rv2"
    """

    def sample(self, **kwargs):
        """
        Sampling method that returns two named tuples

        Returns
        -------
        tuple
           (rv1= SampledValue(val, t_start=self.t_start, t_unit=self.t_unit),
            rv2= SampledValue(val, t_start=self.t_start, t_unit=self.t_unit))
        """
        val = numpyro.sample("my_normal", dist.Normal(0, 1))
        return RVSamples(
            rv1=SampledValue(val, t_start=self.t_start, t_unit=self.t_unit),
            rv2=SampledValue(val, t_start=self.t_start, t_unit=self.t_unit),
        )

    def validate(self):
        """
        No validation.

        Returns
        -------
        None
        """
        return None


class MyModel(Model):
    """
    Model class to create and run variable name recording
    """

    def __init__(self, rv):  # numpydoc ignore=GL08
        self.rv = rv

    def validate(self):  # numpydoc ignore=GL08
        pass

    def sample(self, **kwargs):  # numpydoc ignore=GL08
        return self.rv(record=True, **kwargs)


def test_transform_rv_validation():
    """
    Test that a TransformedRandomVariable validation
    works as expected.
    """

    base_rv = DistributionalRV(
        name="test_normal", distribution=dist.Normal(0, 1)
    )
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
    norm_rv = DistributionalRV(
        name="test_normal", distribution=dist.Normal(0, 1)
    )
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


def test_transforms_variable_naming():
    """
    Tests TransformedRandomVariable name
    recording is as expected.
    """
    transformed_dist_named_base_rv = TransformedRandomVariable(
        "transformed_rv",
        NamedBaseRV(),
        (t.ExpTransform(), t.IdentityTransform()),
    )

    transformed_dist_unnamed_base_rv = TransformedRandomVariable(
        "transformed_rv",
        DistributionalRV(name="my_normal", distribution=dist.Normal(0, 1)),
        (t.ExpTransform(), t.IdentityTransform()),
    )

    transformed_dist_unnamed_base_l2_rv = TransformedRandomVariable(
        "transformed_rv",
        LengthTwoRV(),
        (t.ExpTransform(), t.IdentityTransform()),
    )

    mymodel1 = MyModel(transformed_dist_named_base_rv)
    mymodel1.run(num_samples=1, num_warmup=10, rng_key=jax.random.key(4))

    assert "transformed_rv_rv1" in mymodel1.mcmc.get_samples()
    assert "transformed_rv_rv2" in mymodel1.mcmc.get_samples()

    mymodel2 = MyModel(transformed_dist_unnamed_base_rv)
    mymodel2.run(num_samples=1, num_warmup=10, rng_key=jax.random.key(5))

    assert "transformed_rv" in mymodel2.mcmc.get_samples()

    mymodel3 = MyModel(transformed_dist_unnamed_base_l2_rv)
    mymodel3.run(num_samples=1, num_warmup=10, rng_key=jax.random.key(4))

    assert "transformed_rv_0" in mymodel3.mcmc.get_samples()
    assert "transformed_rv_1" in mymodel3.mcmc.get_samples()