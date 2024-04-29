# -*- coding: utf-8 -*-

"""
Transform classes for PyRenew
"""
from abc import ABCMeta, abstractmethod

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


class AbstractTransform(metaclass=ABCMeta):
    """
    Abstract base class for transformations
    """

    def __call__(self, x):
        return self.transform(x)

    @abstractmethod
    def transform(self, x):
        """
        Transform generated predictions
        """
        pass

    @abstractmethod
    def inverse(self, x):
        """
        Take the inverse of transformed predictions
        """
        pass


class IdentityTransform(AbstractTransform):
    """
    Identity transformation, which
    is its own inverse.

    f(x) = x
    f^-1(x) = x
    """

    def transform(self, x: any):
        """
        Parameters
        ----------
        x : any
            Input, usually ArrayLike

        Returns
        -------
        any
            The same object that was inputted.
        """
        return x

    def inverse(self, x: any):
        """
        Parameters
        ----------
        x : any
            Input, usually ArrayLike

        Returns
        -------
        any
            The same object that was inputted.
        """
        return x


class LogTransform(AbstractTransform):
    """
    Logarithmic (base e) transformation, whose
    inverse is exponentiation.

    f(x) = log(x)
    f^-1(x) = exp(x)
    """

    def transform(self, x: ArrayLike):
        """
        Parameters
        ----------
        x : ArrayLike
            Input, usually predictions array..

        Returns
        -------
        ArrayLike
            Log-transformed input
        """
        return jnp.log(x)

    def inverse(self, x: ArrayLike):
        """
        Parameters
        ----------
        x : ArrayLike
            Input, usually log-scale predictions array.

        Returns
        -------
        ArrayLike
            Exponentiated input
        """
        return jnp.exp(x)


class LogitTransform(AbstractTransform):
    """
    Logistic transformation, whose
    inverse is the inverse logit or
    'expit' function:

    f(x) = log(x) - log(1 - x)
    f^-1(x) = 1 / (1 + exp(-x))
    """

    def transform(self, x: ArrayLike):
        """
        Parameters
        ----------
        x : ArrayLike
            Input, usually predictions array.

        Returns
        -------
        ArrayLike
            Logit transformed input.
        """
        return jax.scipy.special.logit(x)

    def inverse(self, x: ArrayLike):
        """
        Parameters
        ----------
        x : ArrayLike
            Input, usually logit-transformed predictions array.

        Returns
        -------
        ArrayLike
            Inversed logit transformed input.
        """
        return jax.scipy.special.expit(x)


class ScaledLogitTransform(AbstractTransform):
    """
    Scaled logistic transformation from the
    interval (0, X_max) to the interval
    (-infinity, +infinity).
    It's inverse is the inverse logit or
    'expit' function multiplied by X_max
    f(x) = log(x/X_max) - log(1 - x/X_max)
    f^-1(x) = X_max / (1 + exp(-x))
    """

    def __init__(self, x_max: float):
        """
        Default constructor

        Parameters
        ----------
        x_max : float
            Maximum value on the untransformed scale
            (will be transformed to +infinity)
        """
        self.x_max = x_max

    def transform(self, x: ArrayLike):
        """
        Parameters
        ----------
        x : ArrayLike
            Input, usually predictions array.

        Returns
        -------
        ArrayLike
            x_max scaled logit transformed input.
        """
        return jax.scipy.special.logit(x / self.x_max)

    def inverse(self, x: ArrayLike):
        """
        Parameters
        ----------
        x : ArrayLike
            Input, usually scaled logit predictions array.

        Returns
        -------
        ArrayLike
            Inverse of x_max scaled logit transformed input.
        """
        return self.x_max * jax.scipy.special.expit(x)
