"""
Factory functions for
calculating convolutions of timeseries
with discrete distributions
of times-to-event using
[`jax.lax.scan`][].
Factories generate functions
that can be passed to
[`jax.lax.scan`][] or
[`numpyro.contrib.control_flow.scan`][]
with an appropriate array to scan along.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax.typing import ArrayLike


def new_convolve_scanner(
    array_to_convolve: ArrayLike,
    transform: Callable,
) -> Callable:
    r"""
    Factory function to create a "scanner" function
    that can be used with [`jax.lax.scan`][] or
    [`numpyro.contrib.control_flow.scan`][] to
    construct an array via backward-looking iterative
    convolution.

    Parameters
    ----------
    array_to_convolve
        A 1D jax array to convolve with subsets of the
        iteratively constructed history array.

    transform
        A transformation to apply to the result
        of the dot product and multiplication.

    Returns
    -------
    Callable
        A scanner function that can be used with
        [`jax.lax.scan`][] or
        [`numpyro.contrib.control_flow.scan`][]
        for convolution.
        This function takes a history subset array and
        a scalar, computes the dot product of
        the supplied convolution array with the history
        subset array, multiplies by the scalar, and
        returns the resulting value and a new history subset
        array formed by the 2nd-through-last entries
        of the old history subset array followed by that same
        resulting value.

    Notes
    -----
    The following iterative operation is found often
    in renewal processes:

    ```math
    X(t) = f\left(m(t) \begin{bmatrix} X(t - n) \\ X(t - n + 1) \\
    \vdots{} \\ X(t - 1)\end{bmatrix} \cdot{} \mathbf{d} \right)
    ```

    Where $\mathbf{d}$ is a vector of length $n$,
    $m(t)$ is a scalar for each value of time $t$,
    and $f$ is a scalar-valued function.

    Given $\mathbf{d}$, and optionally $f$,
    this factory function returns a new function that
    performs one step of this process while scanning along
    an array of  multipliers (i.e. an array
    giving the values of $m(t)$) using [`jax.lax.scan`][].
    """

    def _new_scanner(
        history_subset: ArrayLike, multiplier: float
    ) -> tuple[ArrayLike, float]:  # numpydoc ignore=GL08
        new_val = transform(
            multiplier * jnp.einsum("i...,i...->...", array_to_convolve, history_subset)
        )
        latest = jnp.concatenate([history_subset[1:], new_val[jnp.newaxis]], axis=0)
        return latest, new_val

    return _new_scanner


def new_double_convolve_scanner(
    arrays_to_convolve: tuple[ArrayLike, ArrayLike],
    transforms: tuple[Callable, Callable],
) -> Callable:
    r"""
    Factory function to create a scanner function
    that iteratively constructs arrays by applying
    the dot-product/multiply/transform operation
    twice per history subset, with the first yielding
    operation yielding an additional scalar multiplier
    for the second.

    Parameters
    ----------
    arrays_to_convolve
        A tuple of two 1D jax arrays, one for
        each of the two stages of convolution.
        The first entry in the arrays_to_convolve
        tuple will be convolved with the
        current history subset array first, the
        the second entry will be convolved with
        it second.
    transforms
        A tuple of two functions, each transforming the
        output of the dot product at each
        convolution stage. The first entry in the transforms
        tuple will be applied first, then the second will
        be applied.

    Returns
    -------
    Callable
        A scanner function that applies two sets of
        convolution, multiply, and transform operations
        in sequence to construct a new array by scanning
        along a pair of input arrays that are equal in
        length to each other.

    Notes
    -----
    Using the same notation as in the documentation for
    [`pyrenew.convolve.new_convolve_scanner`][], this function aids in
    applying the iterative operation:

    ```math
    \begin{aligned}
    Y(t) &= f_1 \left(m_1(t)
        \begin{bmatrix}
            X(t - n) \\
            X(t - n + 1) \\
            \vdots{} \\
            X(t - 1)
    \end{bmatrix} \cdot{} \mathbf{d}_1 \right) \\ \\
    X(t) &= f_2 \left(
        m_2(t) Y(t)
    \begin{bmatrix} X(t - n) \\ X(t - n + 1) \\
    \vdots{} \\ X(t - 1)\end{bmatrix} \cdot{} \mathbf{d}_2 \right)
    \end{aligned}
    ```

    Where $\mathbf{d}_1$ and $\mathbf{d}_2$ are vectors of
    length $n$, $m_1(t)$ and $m_2(t)$ are scalars
    for each value of time $t$, and $f_1$ and $f_2$
    are scalar-valued functions.
    """
    arr1, arr2 = arrays_to_convolve
    t1, t2 = transforms

    def _new_scanner(
        history_subset: ArrayLike,
        multipliers: tuple[float, float],
    ) -> tuple[ArrayLike, tuple[float, float]]:  # numpydoc ignore=GL08
        m1, m2 = multipliers
        m_net1 = t1(m1 * jnp.einsum("i...,i...->...", arr1, history_subset))
        new_val = t2(m2 * m_net1 * jnp.einsum("i...,i...->...", arr2, history_subset))
        latest = jnp.concatenate([history_subset[1:], new_val[jnp.newaxis]], axis=0)
        return latest, (new_val, m_net1)

    return _new_scanner
