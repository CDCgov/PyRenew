#!/usr/bin/env/python
# -*- coding: utf-8 -*-

"""
distutil

Utilities for working with commonly-
encountered probability distributions
found in renewal equation modeling,
such as discrete time-to-event distributions
"""
import jax.numpy as jnp


def validate_discrete_dist_vector(
    discrete_dist: jnp.ndarray, tol: float = 1e-20
) -> bool:
    """
    Validate that a vector represents a discrete
    probability distribution to within a specified
    tolerance, raising a ValueError if not.
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


def reverse_discrete_dist_vector(dist):
    """
    Reverse a discrete distribution
    vector (useful for discrete
    time-to-event distributions).
    """
    return jnp.flip(dist)
