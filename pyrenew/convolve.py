"""
convolve

Factory functions for
calculating convolutions of timeseries
with discrete distributions
of times-to-event using
:py:func:`jax.lax.scan`.
Factories generate functions
that can be passed to
:func:`jax.lax.scan` or
:func:`numpyro.contrib.control_flow.scan`
with an appropriate array to scan along.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax.typing import ArrayLike


def new_convolve_scanner(
    array_to_convolve: ArrayLike,
    transform: Callable,
) -> Callable:
    r"""
    Factory function to create a "scanner" function
    that can be used with :func:`jax.lax.scan` or
    :func:`numpyro.contrib.control_flow.scan` to
    construct an array via backward-looking iterative
    convolution.

    Parameters
    ----------
    array_to_convolve : ArrayLike
        A 1D jax array to convolve with subsets of the
        iteratively constructed history array.

    transform : Callable
        A transformation to apply to the result
        of the dot product and multiplication.

    Returns
    -------
    Callable
        A scanner function that can be used with
        :func:`jax.lax.scan` or
        :func:`numpyro.contrib.control_flow.scan`
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

    .. math::
        X(t) = f\left(m(t) \begin{bmatrix} X(t - n) \\ X(t - n + 1) \\
        \vdots{} \\ X(t - 1)\end{bmatrix} \cdot{} \mathbf{d} \right)

    Where :math:`\mathbf{d}` is a vector of length :math:`n`,
    :math:`m(t)` is a scalar for each value of time :math:`t`,
    and :math:`f` is a scalar-valued function.

    Given :math:`\mathbf{d}`, and optionally :math:`f`,
    this factory function returns a new function that
    peforms one step of this process while scanning along
    an array of  multipliers (i.e. an array
    giving the values of :math:`m(t)`) using :py:func:`jax.lax.scan`.
    """

    def _new_scanner(
        history_subset: ArrayLike, multiplier: float
    ) -> tuple[ArrayLike, float]:  # numpydoc ignore=GL08
        new_val = transform(
            multiplier
            * jnp.einsum("i...,i...->...", array_to_convolve, history_subset)
        )
        latest = jnp.concatenate(
            [history_subset[1:], new_val[jnp.newaxis]], axis=0
        )
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
    arrays_to_convolve : tuple[ArrayLike, ArrayLike]
        A tuple of two 1D jax arrays, one for
        each of the two stages of convolution.
        The first entry in the arrays_to_convolve
        tuple will be convolved with the
        current history subset array first, the
        the second entry will be convolved with
        it second.
    transforms : tuple[Callable, Callable]
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
    :func:`new_convolve_scanner`, this function aids in
    applying the iterative operation:

    .. math::
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

    Where :math:`\mathbf{d}_1` and :math:`\mathbf{d}_2` are vectors of
    length :math:`n`, :math:`m_1(t)` and :math:`m_2(t)` are scalars
    for each value of time :math:`t`, and :math:`f_1` and :math:`f_2`
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
        new_val = t2(
            m2 * m_net1 * jnp.einsum("i...,i...->...", arr2, history_subset)
        )
        latest = jnp.concatenate(
            [history_subset[1:], new_val[jnp.newaxis]], axis=0
        )
        return latest, (new_val, m_net1)

    return _new_scanner


def compute_delay_ascertained_incidence(
    latent_incidence: ArrayLike,
    delay_incidence_to_observation_pmf: ArrayLike,
    p_observed_given_incident: ArrayLike = 1,
    return_offset: bool = False,
) -> ArrayLike | tuple[ArrayLike, int]:
    """
    Computes incidences observed according
    to a given observation rate and based
    on a delay interval.

    Parameters
    ----------
    p_observed_given_incident: ArrayLike
        The rate at which latent incident counts translate into observed counts.
        For example, setting ``p_observed_given_incident=0.001``
        when the incident counts are infections and the observed counts are
        reported hospital admissions could be used to model disease and population
        for which the probability of a latent infection leading to a reported
        hospital admission is 0.001.
    latent_incidence: ArrayLike
        Incidence values based on the true underlying process.
    delay_incidence_to_observation_pmf: ArrayLike
        Probability mass function of delay interval from incidence to
        observation with support on the interval 0 to the length of the
        array's first dimensino. The :math:`i`\th entry represents the
        probability mass for a delay
        of :math:`i` time units, i.e
        ``delay_incidence_to_observation_pmf[0]`` represents
        the fraction of observations that are delayed 0 time unit,
        ``delay_incidence_to_observation_pmf[1]`` represents the fraction
        that are delayed 1 time units, et cetera.

    return_offset: bool
        In addition to the output array, return the offset between
        (number of time units) separating the first entry of the
        the input ``latent_incidence`` array and the output
        array? If ``True``, return a tuple of the output array
        and the offset. If ``False`` return just the output array.
        Boolean, default ``False``.

    Returns
    --------
    ArrayLike
        The predicted timeseries of delayed observations.
    """
    delay_obs_incidence = jnp.convolve(
        p_observed_given_incident * latent_incidence,
        delay_incidence_to_observation_pmf,
        mode="valid",
    )

    if return_offset:
        offset = jnp.shape(delay_incidence_to_observation_pmf)[0] - 1
        result = (delay_obs_incidence, offset)
    else:
        result = delay_obs_incidence
    return result
