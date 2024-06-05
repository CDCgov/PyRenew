# -*- coding: utf-8 -*-

"""
convolve

Factory functions for
calculating convolutions of timeseries
with discrete distributions
of times-to-event using
jax.lax.scan. Factories generate functions
that can be passed to scan with an
appropriate array to scan.
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax.typing import ArrayLike


def new_convolve_scanner(discrete_dist_flipped: ArrayLike) -> Callable:
    """
    Factory function to create a scanner function for
    convolving a discrete distribution
    over a time series data subset.

    Parameters
    ----------
    discrete_dist_flipped : ArrayLike
        A 1D jax array representing the discrete
        distribution flipped for convolution.

    Returns
    -------
    Callable
        A scanner function that can be used with
        jax.lax.scan for convolution.
        This function takes a history subset and
        a multiplier, computes the dot product,
        then updates and returns the new history
        subset and the convolution result.
    """

    def _new_scanner(
        history_subset: ArrayLike, multiplier: float
    ) -> tuple[ArrayLike, float]:  # numpydoc ignore=GL08
        new_val = multiplier * jnp.dot(discrete_dist_flipped, history_subset)
        latest = jnp.hstack([history_subset[1:], new_val])
        return latest, new_val

    return _new_scanner


def new_double_scanner(
    dists: tuple[ArrayLike, ArrayLike],
    transforms: tuple[Callable, Callable],
) -> Callable:
    """
    Factory function to create a scanner function that
    applies two sequential transformations
    and convolutions using two discrete distributions.

    Parameters
    ----------
    dists : tuple[ArrayLike, ArrayLike]
        A tuple of two 1D jax arrays, each representing a
        discrete distribution for the
        two stages of convolution.
    transforms : tuple[Callable, Callable]
        A tuple of two functions, each transforming the
        output of the dot product at each
        convolution stage.

    Returns
    -------
    Callable
        A scanner function that applies two sequential
        convolutions and transformations. It takes a history
        subset and a tuple of multipliers,
        computes the transformations and dot products,
        and returns the updated history
        subset and a tuple of new values.
    """
    d1, d2 = dists
    t1, t2 = transforms

    def _new_scanner(
        history_subset: ArrayLike,
        multipliers: tuple[float, float],
    ) -> tuple[ArrayLike, tuple[float, float]]:  # numpydoc ignore=GL08
        m1, m2 = multipliers
        m_net1 = t1(m1 * jnp.dot(d1, history_subset))
        new_val = t2(m2 * m_net1 * jnp.dot(d2, history_subset))
        latest = jnp.hstack([history_subset[1:], new_val])
        return latest, (new_val, m_net1)

    return _new_scanner
