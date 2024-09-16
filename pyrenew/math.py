"""
Helper functions for doing analytical
and/or numerical calculations.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.lax import scan
from jax.typing import ArrayLike

from pyrenew.distutil import validate_discrete_dist_vector


def get_leslie_matrix(
    R: float, generation_interval_pmf: ArrayLike
) -> ArrayLike:
    """
    Create the Leslie matrix
    corresponding to a basic
    renewal process with the
    given R value and discrete
    generation interval pmf
    vector.

    Parameters
    ----------
    R : float
       The reproduction number of the renewal process
    generation_interval_pmf: ArrayLike
       The discrete generation interval probability
       mass vector of the renewal process

    Returns
    -------
    ArrayLike
        The Leslie matrix for the
        renewal process, as a jax array.
    """
    validate_discrete_dist_vector(generation_interval_pmf)
    gen_int_len = generation_interval_pmf.size
    aging_matrix = jnp.hstack(
        [
            jnp.identity(gen_int_len - 1),
            jnp.zeros(gen_int_len - 1)[..., jnp.newaxis],
        ]
    )

    return jnp.vstack([R * generation_interval_pmf, aging_matrix])


def get_asymptotic_growth_rate_and_age_dist(
    R: float, generation_interval_pmf: ArrayLike
) -> tuple[float, ArrayLike]:
    """
    Get the asymptotic per-timestep growth
    rate of the renewal process (the dominant
    eigenvalue of its Leslie matrix) and the
    associated stable age distribution
    (a normalized eigenvector associated to
    that eigenvalue).

    Parameters
    ----------
    R : float
       The reproduction number of the renewal process
    generation_interval_pmf: ArrayLike
       The discrete generation interval probability
       mass vector of the renewal process

    Returns
    -------
    tuple[float, ArrayLike]
        A tuple consisting of the asymptotic growth rate of
        the process, as jax float, and the stable age distribution
        of the process, as a jax array probability vector of the
        same shape as the generation interval probability vector.

    Raises
    ------
    ValueError
        If an age distribution vector with non-zero imaginary part is produced.
    """
    L = get_leslie_matrix(R, generation_interval_pmf)
    eigenvals, eigenvecs = jnp.linalg.eig(L)
    d = jnp.argmax(jnp.abs(eigenvals))  # index of dominant eigenvalue
    d_vec, d_val = eigenvecs[:, d], eigenvals[d]
    d_vec_real, d_val_real = jnp.real(d_vec), jnp.real(d_val)
    if not all(d_vec_real == d_vec):
        raise ValueError(
            "get_asymptotic_growth_rate_and_age_dist() "
            "produced an age distribution vector with "
            "non-zero imaginary part. "
            "Check your generation interval distribution "
            "vector and R value"
        )
    if not d_val_real == d_val:
        raise ValueError(
            "get_asymptotic_growth_rate_and_age_dist() "
            "produced an asymptotic growth rate with "
            "non-zero imaginary part. "
            "Check your generation interval distribution "
            "vector and R value"
        )
    d_vec_norm = d_vec_real / jnp.sum(d_vec_real)
    return d_val_real, d_vec_norm


def get_stable_age_distribution(
    R: float, generation_interval_pmf: ArrayLike
) -> ArrayLike:
    """
    Get the stable age distribution for a
    renewal process with a given value of
    R and a given discrete generation
    interval probability mass vector.

    This function computes that stable age
    distribution by finding and then normalizing
    an eigenvector associated to the dominant
    eigenvalue of the renewal process's
    Leslie matrix.

    Parameters
    ----------
    R : float
       The reproduction number of the renewal process
    generation_interval_pmf: ArrayLike
       The discrete generation interval probability
       mass vector of the renewal process

    Returns
    -------
    ArrayLike
        The stable age distribution for the
        process, as a jax array probability vector of
        the same shape as the generation interval
        probability vector.
    """
    return get_asymptotic_growth_rate_and_age_dist(R, generation_interval_pmf)[
        1
    ]


def get_asymptotic_growth_rate(
    R: float, generation_interval_pmf: ArrayLike
) -> float:
    """
    Get the asymptotic per timestep growth rate
    for a renewal process with a given value of
    R and a given discrete generation interval
    probability mass vector.

    This function computes that growth rate
    finding the dominant eigenvalue of the
    renewal process's Leslie matrix.

    Parameters
    ----------
    R : float
       The reproduction number of the renewal process
    generation_interval_pmf: ArrayLike
       The discrete generation interval probability
       mass vector of the renewal process

    Returns
    -------
    float
        The asymptotic growth rate of the renewal process,
        as a jax float.
    """
    return get_asymptotic_growth_rate_and_age_dist(R, generation_interval_pmf)[
        0
    ]


def integrate_discrete(
    init_diff_vals: ArrayLike, highest_order_diff_vals: ArrayLike
) -> ArrayLike:
    """
    Integrate (de-difference) the differenced process,
    obtaining the process values :math:`X(t=0), X(t=1), ... X(t)`
    from the :math:`n^{th}` differences and a set of
    initial process / difference values
    :math:`X(t=0), X^1(t=1), X^2(t=2), ... X^{(n-1)}(t=n-1)`,
    where :math:`X^k(t)` is the value of the :math:`n^{th}`
    difference at index :math:`t` of the process,
    obtaining a sequence of length equal to the length of
    the provided `highest_order_diff_vals` vector plus
    the order of the process.

    Parameters
    ----------
    init_diff_vals : ArrayLike
        Values of
        :math:`X(t=0), X^1(t=1), X^2(t=2) ... X^{(n-1)}(t=n-1)`.

    highest_order_diff_vals : ArrayLike
        Array of differences at the highest order of
        differencing, i.e. the order of the overall process,
        starting with :math:`X^{n}(t=n)`

    Returns
    -------
    ArrayLike
        The integrated (de-differenced) sequence of values,
        of length n_diffs + order, where n_diffs is the
        number of highest_order_diff_vals and order is the
        order of the process.
    """
    order_level_init_vals = jnp.atleast_1d(init_diff_vals)
    order = order_level_init_vals.shape[0]

    if not (
        highest_order_diff_vals.shape[1:] == order_level_init_vals.shape[1:]
    ):
        raise ValueError(
            "highest_order_diff_vals must have the same "
            "non-time dimension batch shape (i.e. shape after "
            "the first dimension) as the order_level_init_vals. "
            "Got highest_order_diff_vals of batch shape "
            f"{highest_order_diff_vals.shape[1:]} and "
            "order_level_init_vals of batch shape "
            f"{order_level_init_vals.shape[1:]}"
        )

    highest_diffs = jnp.concatenate(
        [
            jnp.zeros_like(order_level_init_vals),
            jnp.atleast_1d(highest_order_diff_vals),
        ],
        axis=0,
    )

    def _integrate_one_step(
        current_diffs: ArrayLike,
        next_order_and_init: tuple[int, ArrayLike],
    ) -> tuple[ArrayLike, None]:
        """
        Perform one step of integration
        (de-differencing) of the process.

        Parameters
        ----------
        current_diffs: ArrayLike
            Array of differences at the current
            de-differencing order

        next_order_and_init: tuple
            Tuple containing with two entries.
            First entry: the next order of de-differencing
            (the current order - 1) as an integer.
            Second entry: the initial value at
            that the next order of de-differencing
            as an ArrayLike of appropriate shape.

        Returns
        -------
        tuple[ArrayLike, None]
            A tuple whose first entry contains the
            values at the next order of (de)-differencing
            and whose second entry is None.
        """
        next_order, next_init = next_order_and_init
        next_diffs = jnp.cumsum(
            current_diffs.at[next_order, ...].set(next_init)
        )
        return next_diffs, None

    scan_arrays = (
        jnp.arange(start=order - 1, stop=-1, step=-1),
        jnp.flip(order_level_init_vals, axis=0),
    )

    integrated, _ = scan(
        f=_integrate_one_step, init=highest_diffs, xs=scan_arrays
    )

    return integrated
