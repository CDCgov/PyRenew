# -*- coding: utf-8 -*-

"""
Helper classes for regression problems
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpyro
import numpyro.distributions as dist
import pyrenew.transformation as nt
from jax.typing import ArrayLike


class AbstractRegressionPrediction(metaclass=ABCMeta):  # numpydoc ignore=GL08
    @abstractmethod
    def predict(self):
        """
        Make a regression prediction
        """
        pass

    @abstractmethod
    def sample(self, obs=None):
        """
        Observe or sample from the regression
        problem according to the specified
        distributions
        """
        pass


class GLMPrediction(AbstractRegressionPrediction):
    """
    Generalized linear model regression
    predictions
    """

    def __init__(
        self,
        name: str,
        fixed_predictor_values: ArrayLike,
        intercept_prior: dist.Distribution,
        coefficient_priors: dist.Distribution,
        transform: nt.Transform = None,
        intercept_suffix="_intercept",
        coefficient_suffix="_coefficients",
    ) -> None:
        """
        Default class constructor for GLMObservation

        Parameters
        ----------
        name : str
            The name of the observation process,
            which will be used to name the constituent
            sampled parameters in calls to `numpyro.sample`

        fixed_predictor_values : ArrayLike (n_predictors, n_observations)
            Matrix of fixed values of the predictor variables
            (covariates) for the regression problem. Each
            row should represent the predictor values corresponding
            to an observation; each column should represent
            a predictor variable. You do not include values of
            1 for the intercept; these will be added automatically.

        intercept_prior : numypro.distributions.Distribution
            Prior distribution for the regression intercept
            value

        coefficient_priors : numpyro.distributions.Distribution
            Vectorized prior distribution for the regression
            coefficient values

        transform : numpyro.distributions.transforms.Transform, optional
            Transform linking the scale of the
            regression to the scale of the observation.
            If `None`, use an identity transform. Default
            `None`.

        intercept_suffix : str, optional
            Suffix for naming the intercept random variable in
            class to numpyro.sample(). Default `"_intercept"`.

        coefficient_suffix : str, optional
            Suffix for naming the regression coefficient
            random variables in calls to numpyro.sample().
            Default `"_coefficients"`.
        """
        if transform is None:
            transform = nt.IdentityTransform()

        self.name = name
        self.fixed_predictor_values = fixed_predictor_values
        self.transform = transform
        self.intercept_prior = intercept_prior
        self.coefficient_priors = coefficient_priors
        self.intercept_suffix = intercept_suffix
        self.coefficient_suffix = coefficient_suffix

    def predict(
        self, intercept: ArrayLike, coefficients: ArrayLike
    ) -> ArrayLike:
        """
        Generates a transformed prediction w/ intercept, coefficients, and
        fixed predictor values

        Parameters
        ----------
        intercept : ArrayLike
            Sampled numpyro distribution generated from intercept priors.
        coefficients : ArrayLike
            Sampled prediction coefficients distribution generated
            from coefficients priors.

        Returns
        -------
        ArrayLike
            Array of transformed predictions.
        """
        transformed_prediction = (
            intercept + self.fixed_predictor_values @ coefficients
        )
        return self.transform.inv(transformed_prediction)

    def sample(self) -> dict:
        """
        Sample generalized linear model

        Returns
        -------
        dict
            A dictionary containing transformed predictions, and
            the intercept and coefficients sample distributions.
        """
        intercept = numpyro.sample(
            self.name + self.intercept_suffix, self.intercept_prior
        )
        coefficients = numpyro.sample(
            self.name + self.coefficient_suffix, self.coefficient_priors
        )
        prediction = self.predict(intercept, coefficients)
        return dict(
            prediction=prediction,
            intercept=intercept,
            coefficients=coefficients,
        )

    def __repr__(self):
        return "GLMPrediction " + str(self.name)
