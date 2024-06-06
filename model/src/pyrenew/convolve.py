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
from pyrenew.transformation import IdentityTransform


def new_convolve_scanner(
    discrete_dist_flipped: ArrayLike, transform: Callable = None
) -> Callable:
    r"""
    Factory function to create a "scanner" function
    that can be used with :py:func:`jax.lax.scan` to
    construct an array via backward-looking iterative
    convolution.

    Parameters
    ----------
    discrete_dist_flipped : ArrayLike
        A 1D jax array representing the discrete
        distribution flipped for convolution.

    transform : Callable
        A transformation to apply to the result
        of the dot product and multiplication.
        If None, use the identity transformation.
        Default None.

    Returns
    -------
    Callable
        A scanner function that can be used with
        jax.lax.scan for convolution.
        This function takes a history subset and
        a multiplier, computes the dot product,
        then updates and returns the new history
        subset and the convolution result.

    Notes
    -----
    The following iterative operation is found often
    in renewal processes:

    .. math::
        X(t) = f\left(m(t) * \left[X(t - n),
        X(t - n + 1), ... X(t - 1)\right] \dot \vec{d} \right)

    Where `math`:\vec{d} is a vector of length `math`:n,
    `math`:m(t) is a scalar for each value of time `math`:t,
    and `math`:f is a scalar-valued function.

    Given `math`:\vec{d}, and optionally `math`:f,
    this factory function returns a new function that
    peforms one step of this process while scanning along
    an array of  multipliers (i.e. an array
    giving the values of `math`:m(t)) using :py:func:jax.lax.scan.
    """
    if transform is None:
        transform = IdentityTransform()

    def _new_scanner(
        history_subset: ArrayLike, multiplier: float
    ) -> tuple[ArrayLike, float]:  # numpydoc ignore=GL08
        new_val = transform(
            multiplier * jnp.dot(discrete_dist_flipped, history_subset)
        )
        latest = jnp.hstack([history_subset[1:], new_val])
        return latest, new_val

    return _new_scanner


def new_double_convolve_scanner(
    dists: tuple[ArrayLike, ArrayLike],
    transforms: tuple[Callable, Callable] = (None, None),
) -> Callable:
    """
    Factory function to create a scanner function
    that iteratively constructs arrays by applying
    the dot-product/multiply/transform operation
    twice per history subset, with the first yielding
    operation yielding an additional scalar multiplier
    for the second.

    Parameters
    ----------
    dists : tuple[ArrayLike, ArrayLike]
        A tuple of two 1D jax arrays, each representing a
        discrete distribution for the
        two stages of convolution.
    transforms : tuple[Callable, Callable]
        A tuple of two functions, each transforming the
        output of the dot product at each
        convolution stage. If either is None,
        the identity transformation will be used
        at that step. Default (None, None)

    Returns
    -------
    Callable
        A scanner function that applies two sets of
        convolution, multiply, and transform operations
        in sequence to construct a new array by scanning
        along a pair of input arrays that are equal in
        length to each other.
    """
    d1, d2 = dists
    t1, t2 = [x if x is not None else IdentityTransform() for x in transforms]

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
