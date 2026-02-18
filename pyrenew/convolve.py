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


def compute_delay_ascertained_incidence(
    latent_incidence: ArrayLike,
    delay_incidence_to_observation_pmf: ArrayLike,
    p_observed_given_incident: ArrayLike = 1,
    pad: bool = False,
) -> tuple[ArrayLike, int]:
    """
    Computes incidences observed according
    to a given observation rate and based
    on a delay interval.

    In addition to the output array, returns the offset
    (number of time units) separating the first entry of the
    the input `latent_incidence` array from the first entry
    of the output (delay ascertained incidence) array.
    Note that if the `pad` keyword argument is `True`,
    the offset will be always `0`.

    Parameters
    ----------
    latent_incidence
        Incidence values based on the true underlying process.

    delay_incidence_to_observation_pmf
        Probability mass function of delay interval from incidence to
        observation with support on the interval 0 to the length of the
        array's first dimension. The $i$\th entry represents the
        probability mass for a delay
        of $i$ time units, i.e
        ``delay_incidence_to_observation_pmf[0]`` represents
        the fraction of observations that are delayed 0 time unit,
        ``delay_incidence_to_observation_pmf[1]`` represents the fraction
        that are delayed 1 time units, et cetera.

    p_observed_given_incident
        The rate at which latent incident counts translate into observed
        counts. For example, setting ``p_observed_given_incident=0.001``
        when the incident counts are infections and the observed counts
        are reported hospital admissions could be used to model disease
        and population for which the probability of a latent infection
        leading to a reported hospital admission is 0.001. Default `1`.

    pad
        Return an output array that has been nan-padded so that its
        first entry represents the same timepoint as the first timepoint
        of the input `latent_incidence` array? Boolean, default `False`.

    Returns
    -------
    tuple[ArrayLike, int]
        Tuple whose first entry is the predicted timeseries of
        delayed observations and whose second entry is the offset.
    """
    delay_obs_incidence = jnp.convolve(
        p_observed_given_incident * latent_incidence,
        delay_incidence_to_observation_pmf,
        mode="valid",
    )

    offset = jnp.shape(delay_incidence_to_observation_pmf)[0] - 1

    if pad:
        delay_obs_incidence = jnp.pad(
            1.0 * delay_obs_incidence,  # ensure float since
            # nans pad as zeros for ints
            (offset, 0),
            mode="constant",
            constant_values=jnp.nan,
        )
        offset = 0
    return (delay_obs_incidence, offset)


def compute_prop_already_reported(
    reporting_delay_pmf: ArrayLike,
    n_timepoints: int,
    right_truncation_offset: int,
) -> ArrayLike:
    """
    Compute the proportion of events already reported at each timepoint,
    given a reporting delay PMF and a right-truncation offset.

    For right-truncated data, recent timepoints have lower expected counts
    because not all events have been reported yet. This function computes
    the cumulative proportion reported for each timepoint.

    Parameters
    ----------
    reporting_delay_pmf : ArrayLike
        PMF of reporting delays. The i-th entry is the probability that
        an event is reported with a delay of i time units.
    n_timepoints : int
        Number of timepoints in the output array.
    right_truncation_offset : int
        Number of additional timepoints beyond the last observation
        for which reports could still arrive. An offset of 0 means
        the last timepoint has only had time for delay-0 reports.

    Returns
    -------
    ArrayLike
        Array of shape (n_timepoints,) where each entry is the
        proportion of events already reported at that timepoint.
        Earlier timepoints are 1.0 (fully reported); recent
        timepoints approach reporting_delay_pmf[0] (minimally reported).
    """
    cdf = jnp.cumsum(reporting_delay_pmf)
    tail = jnp.flip(cdf[right_truncation_offset:])
    n_pad = n_timepoints - tail.shape[0]
    return jnp.concatenate([jnp.ones(n_pad), tail])
