# -*- coding: utf-8 -*-

"""
Helper classes for regression problems
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpyro
import numpyro.distributions as dist
import pyrenew.transformation as t
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
        intercept_prior: dist.Distribution,
        coefficient_priors: dist.Distribution,
        transform: t.Transform = None,
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
            transform = t.IdentityTransform()

        self.name = name
        self.transform = transform
        self.intercept_prior = intercept_prior
        self.coefficient_priors = coefficient_priors
        self.intercept_suffix = intercept_suffix
        self.coefficient_suffix = coefficient_suffix

    def predict(
        self,
        intercept: ArrayLike,
        coefficients: ArrayLike,
        predictor_values: ArrayLike,
    ) -> ArrayLike:
        """
        Generates a transformed prediction w/ intercept, coefficients, and
        predictor values

        Parameters
        ----------
        intercept : ArrayLike
            Sampled numpyro distribution generated from intercept priors.
        coefficients : ArrayLike
            Sampled prediction coefficients distribution generated
            from coefficients priors.
        predictor_values : ArrayLike(n_predictors, n_observations)
            Matrix of predictor variables (covariates) for the
            regression problem. Each row should represent the
            predictor values corresponding to an observation;
            each column should represent a predictor variable.
            You do not include values of 1 for the intercept;
            these will be added automatically.

        Returns
        -------
        ArrayLike
            Array of transformed predictions.
        """
        transformed_prediction = intercept + predictor_values @ coefficients
        return self.transform.inv(transformed_prediction)

    def sample(self, predictor_values: ArrayLike, **kwargs) -> dict:
        """
        Sample generalized linear model

        Parameters
        -----------
        predictor_values : ArrayLike(n_predictors, n_observations)
            Matrix of predictor variables (covariates) for the
            regression problem. Each row should represent the
            predictor values corresponding to an observation;
            each column should represent a predictor variable.
            Do not include values of 1 for the intercept;
            these will be added automatically. Passed as the 
            `predictor_values` argument to 
            :meth:`GLMPrediction.predict()`

        **kwargs : dict
             Additional keyword arguments. Ignored.
            kwargs containing additional arguments including predictor_values

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

        predictor_values = kwargs.get("predictor_values")
        prediction = self.predict(intercept, coefficients, predictor_values)
        return dict(
            prediction=prediction,
            intercept=intercept,
            coefficients=coefficients,
        )

    def __call__(self, **kwargs):
        """
        Alias for `sample()`.
        """
        return self.sample(**kwargs)

    def __repr__(self):
        return "GLMPrediction " + str(self.name)
