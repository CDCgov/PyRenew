#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for regression functionality
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pyrenew.regression as r
import pyrenew.transform as t
from numpy.testing import assert_array_almost_equal


def test_glm_prediction():
    """
    Test generalized linear model
    prediction functionality
    """
    intercept_custom_suffix = "_523sdgahbf"
    coefficient_custom_suffix = "_gad23562g%"
    fixed_predictor_values = jnp.array([[2, 0.5, -7, 3], [1, 20, -15, 0]])

    glm_pred = r.GLMPrediction(
        "test_GLM_prediction",
        fixed_predictor_values=fixed_predictor_values,
        intercept_prior=dist.Normal(0, 1.5),
        coefficient_priors=dist.Normal(0, 0.5).expand((4,)),
        transform=None,
        intercept_suffix=intercept_custom_suffix,
        coefficient_suffix=coefficient_custom_suffix,
    )

    # if not set, transform should be identity
    assert isinstance(glm_pred.transform, t.IdentityTransform)

    # deterministic predictions should work as
    # matrix algebra
    fixed_pred_coeff = jnp.array([1, 35235, -5232.2532, 0])
    fixed_pred_intercept = jnp.array([5.2])
    assert_array_almost_equal(
        glm_pred.predict(fixed_pred_intercept, fixed_pred_coeff),
        fixed_pred_intercept + fixed_predictor_values @ fixed_pred_coeff,
    )

    # all coefficients and intercept equal to zero
    # should make all predictions zero
    assert_array_almost_equal(
        glm_pred.predict(
            jnp.zeros(1), jnp.zeros(fixed_predictor_values.shape[1])
        ),
        jnp.zeros(fixed_predictor_values.shape[0]),
    )

    # sampling should work
    with numpyro.handlers.seed(rng_seed=5):
        preds = glm_pred.sample()

    assert isinstance(preds, dict)

    ## check prediction output
    ## is of expected type and shape
    assert "prediction" in preds.keys()
    assert isinstance(preds["prediction"], jnp.ndarray)
    assert preds["prediction"].shape[0] == fixed_predictor_values.shape[0]

    ## check coeffficients
    assert "coefficients" in preds.keys()

    # check results agree with manual calculation
    assert_array_almost_equal(
        preds["prediction"],
        preds["intercept"] + fixed_predictor_values @ preds["coefficients"],
    )
