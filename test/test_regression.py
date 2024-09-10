"""
Tests for regression functionality
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpy.testing import assert_array_almost_equal

import pyrenew.regression as r
import pyrenew.transformation as t


def test_glm_prediction():
    """
    Test generalized linear model
    prediction functionality
    """
    intercept_custom_suffix = "_523sdgahbf"
    coefficient_custom_suffix = "_gad23562g%"
    predictor_values = jnp.array([[2, 0.5, -7, 3], [1, 20, -15, 0]])

    glm_pred = r.GLMPrediction(
        "test_GLM_prediction",
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
        glm_pred.predict(
            fixed_pred_intercept, fixed_pred_coeff, predictor_values
        ),
        fixed_pred_intercept + predictor_values @ fixed_pred_coeff,
    )

    # all coefficients and intercept equal to zero
    # should make all predictions zero
    assert_array_almost_equal(
        glm_pred.predict(
            jnp.zeros(1),
            jnp.zeros(predictor_values.shape[1]),
            predictor_values,
        ),
        jnp.zeros(predictor_values.shape[0]),
    )

    # sampling should work
    with numpyro.handlers.seed(rng_seed=5):
        preds = glm_pred(predictor_values=predictor_values)

    ## check prediction output
    ## is of expected type and shape
    assert preds.prediction.shape[0] == predictor_values.shape[0]

    ## check coeffficients and intercept

    # check results agree with manual calculation
    assert_array_almost_equal(
        preds.prediction,
        preds.intercept + predictor_values @ preds.coefficients,
    )
