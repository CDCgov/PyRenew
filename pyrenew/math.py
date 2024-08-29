"""
Helper functions for doing analytical
and/or numerical calculations about
a given renewal process.
"""

from __future__ import annotations

import jax.numpy as jnp
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
