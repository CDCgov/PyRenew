# -*- coding: utf-8 -*-

"""
distutil

Utilities for working with commonly-
encountered probability distributions
found in renewal equation modeling,
such as discrete time-to-event distributions
"""
import jax.numpy as jnp
from jax.typing import ArrayLike


def validate_discrete_dist_vector(
    discrete_dist: ArrayLike, tol: float = 1e-20
) -> ArrayLike:
    """
    Validate that a vector represents a discrete
    probability distribution to within a specified
    tolerance, raising a ValueError if not.

    Parameters
    ----------
    discrete_dist : ArrayLike
        An jax array containing non-negative values that
        represent a discrete probability distribution. The values
        must sum to 1 within the specified tolerance.
    tol : float, optional
        The tolerance within which the sum of the distribution must
        be 1. Defaults to 1e-20.

    Returns
    -------
    ArrayLike
        The normalized distribution array if the input is valid.

    Raises
    ------
    ValueError
        If any value in discrete_dist is negative or if the sum of the
        distribution does not equal 1 within the specified tolerance.
    """
    discrete_dist = discrete_dist.flatten()
    if not jnp.all(discrete_dist >= 0):
        raise ValueError(
            "Discrete distribution "
            "vector must have "
            "only non-negative "
            "entries; got {}"
            "".format(discrete_dist)
        )
    dist_norm = jnp.sum(discrete_dist)
    if not jnp.abs(dist_norm - 1) < tol:
        raise ValueError(
            "Discrete generation interval "
            "distributions must sum to 1"
            "with a tolerance of {}"
            "".format(tol)
        )
    return discrete_dist / dist_norm


def reverse_discrete_dist_vector(dist: ArrayLike) -> ArrayLike:
    """
    Reverse a discrete distribution
    vector (useful for discrete
    time-to-event distributions).

    Parameters
    ----------
    dist : ArrayLike
        A discrete distribution vector (likely discrete time-to-event distribution)

    Returns
    -------
    ArrayLike
        A reversed (jnp.flip) discrete distribution vector
    """
    return jnp.flip(dist)
