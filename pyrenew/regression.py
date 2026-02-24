"""
Helper classes for regression problems
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import NamedTuple

import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

import pyrenew.transformation as t


class AbstractRegressionPrediction(metaclass=ABCMeta):  # numpydoc ignore=GL08
    @abstractmethod
    def predict(self) -> ArrayLike:
        """
        Make a regression prediction
        """
        pass

    @abstractmethod
    def sample(self, obs: ArrayLike | None = None) -> ArrayLike:
        """
        Observe or sample from the regression
        problem according to the specified
        distributions
        """
        pass


class GLMPredictionSample(NamedTuple):
    """
    A container for holding the output from `GLMPrediction.sample()`.

    Attributes
    ----------
    prediction
        Transformed predictions. Defaults to None.
    intercept
        Sampled intercept from intercept priors.
        Defaults to None.
    coefficients
        Prediction coefficients generated
        from coefficients priors. Defaults to None.
    """

    prediction: ArrayLike | None = None
    intercept: ArrayLike | None = None
    coefficients: ArrayLike | None = None

    def __repr__(self) -> str:
        return (
            f"GLMPredictionSample("
            f"prediction={self.prediction}, "
            f"intercept={self.intercept}, "
            f"coefficients={self.coefficients})"
        )


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
        intercept_suffix: str = "_intercept",
        coefficient_suffix: str = "_coefficients",
    ) -> None:
        """
        Default class constructor for GLMPrediction

        Parameters
        ----------
        name
            The name of the prediction process,
            which will be used to name the constituent
            sampled parameters in calls to [`numpyro.primitives.sample`][]

        intercept_prior
            Prior distribution for the regression intercept
            value

        coefficient_priors
            Vectorized prior distribution for the regression
            coefficient values

        transform
            Transform linking the scale of the
            regression to the scale of the observation.
            If `None`, use an identity transform. Default
            `None`.

        intercept_suffix
            Suffix for naming the intercept random variable in
            class to numpyro.sample(). Default `"_intercept"`.

        coefficient_suffix
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
        intercept
            Sampled numpyro distribution generated from intercept priors.
        coefficients
            Sampled prediction coefficients distribution generated
            from coefficients priors.
        predictor_values
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

    def sample(self, predictor_values: ArrayLike) -> GLMPredictionSample:
        """
        Sample generalized linear model

        Parameters
        ----------
        predictor_values
            Matrix of predictor variables (covariates) for the
            regression problem. Each row should represent the
            predictor values corresponding to an observation;
            each column should represent a predictor variable.
            Do not include values of 1 for the intercept;
            these will be added automatically. Passed as the
            `predictor_values` argument to
            [`pyrenew.regression.GLMPrediction.predict`][].

        Returns
        -------
        GLMPredictionSample
        """

        intercept = numpyro.sample(
            self.name + self.intercept_suffix, self.intercept_prior
        )
        coefficients = numpyro.sample(
            self.name + self.coefficient_suffix, self.coefficient_priors
        )
        prediction = self.predict(intercept, coefficients, predictor_values)

        return GLMPredictionSample(
            prediction=prediction,
            intercept=intercept,
            coefficients=coefficients,
        )

    def __call__(self, *args: object, **kwargs: object) -> GLMPredictionSample:
        """
        Alias for `sample()`.
        """
        return self.sample(*args, **kwargs)

    def __repr__(self) -> str:
        return "GLMPrediction " + str(self.name)
