"""
Helper functions for doing analytical
and/or numerical calculations.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.lax import broadcast_shapes, scan
from jax.typing import ArrayLike

from pyrenew.distutil import validate_discrete_dist_vector


def _positive_ints_like(vec: ArrayLike) -> jnp.ndarray:
    """
    Given an array of size n, return the 1D Jax array
    ``[1, ... n]``.

    Parameters
    ----------
    vec: ArrayLike
        The template array

    Returns
    -------
    jnp.ndarray
        The resulting array ``[1, ..., n]``.
    """
    return jnp.arange(1, jnp.size(vec) + 1)


def neg_MGF(r: float, w: ArrayLike) -> float:
    """
    Compute the negative moment generating function (MGF)
    for a given rate ``r`` and weights ``w``.

    Parameters
    ----------
    r: float
       The rate parameter.

    w: ArrayLike
       An array of weights.

    Returns
    -------
    float
        The value of the negative MGF evaluated at ``r``
        and ``w``.

    Notes
    -----
    For a finite discrete random variable :math:`X` supported on
    the first :math:`n` positive integers (:math:`\\{1, 2, ..., n \\}`),
    the moment generating function (MGF) :math:`M_+(r)` is defined
    as the expected value of :math:`\\exp(rX)`. Similarly, the negative
    moment generating function :math:`M_-(r)` is the expected value of
    :math:`\\exp(-rX)`. So if we represent the PMF of :math:`X` as a
    "weights" vector :math:`w` of length :math:`n`, the negative MGF
    :math:`M_-(r)` is given by:

    .. math::
        M_-(r) = \\sum_{t = 1}^{n} w_i \\exp(-rt)
    """
    return jnp.sum(w * jnp.exp(-r * _positive_ints_like(w)))


def neg_MGF_del_r(r: float, w: ArrayLike) -> float:
    """
    Compute the value of the partial deriative of
    :func:`neg_MGF` with respect to ``r``
    evaluated at a particular ``r`` and ``w`` pair.

    Parameters
    ----------
    r: float
       The rate parameter.

    w: ArrayLike
       An array of weights.

    Returns
    -------
    float
        The value of the partial derivative evaluated at ``r``
        and ``w``.
    """
    t_vec = _positive_ints_like(w)
    return -jnp.sum(w * t_vec * jnp.exp(-r * t_vec))


def r_approx_from_R(R: float, g: ArrayLike, n_newton_steps: int) -> ArrayLike:
    """
    Get the approximate asymptotic geometric growth rate ``r``
    for a renewal process with a fixed reproduction number ``R``
    and discrete generation interval PMF ``g``.

    Uses Newton's method with a fixed number of steps.

    Parameters
    ----------
    R: float
        The reproduction number

    g: ArrayLike
        The probability mass function of the generation
        interval.

    n_newton_steps: int
        Number of steps to take when performing Newton's method.

    Returns
    -------
    float
        The approximate value of ``r``.

    Notes
    -----
    For a fixed value of :math:`\\mathcal{R}`, a renewal process
    has an asymptotic geometric growth rate :math:`r` that satisfies

    .. math::
        M_{-}(r) - \\frac{1}{\\mathcal{R}} = 0

    where :math:`M_-(r)` is the negative moment generating function
    for a random variable :math:`\\tau` representing the (discrete)
    generation interval. See :func:`neg_MGF` for details.

    We obtain a value for :math:`r` via approximate numerical solution
    of this implicit equation.

    We first make an initial guess based on the mean generation interval
    :math:`\\bar{\\tau} = \\mathbb{E}(\\tau)`:

    .. math::
        r \\approx \\frac{\\mathcal{R} - 1}{\\mathcal{R} \\bar{\\tau}}

    We then refine this approximation by applying Newton's method for
    a fixed number of steps.
    """
    mean_gi = jnp.dot(g, _positive_ints_like(g))
    init_r = (R - 1) / (R * mean_gi)

    def _r_next(r, _) -> tuple[ArrayLike, None]:  # numpydoc ignore=GL08
        return (
            r - ((R * neg_MGF(r, g) - 1) / (R * neg_MGF_del_r(r, g))),
            None,
        )

    result, _ = scan(f=_r_next, init=init_r, xs=None, length=n_newton_steps)
    return result


def get_leslie_matrix(R: float, generation_interval_pmf: ArrayLike) -> ArrayLike:
    """
    Create the Leslie matrix
    corresponding to a basic
    renewal process with the
    given :math:`\\mathcal{R}`
    value and discrete
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
        If an age distribution vector with non-zero
        imaginary part is produced.
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
    return get_asymptotic_growth_rate_and_age_dist(R, generation_interval_pmf)[1]


def get_asymptotic_growth_rate(R: float, generation_interval_pmf: ArrayLike) -> float:
    """
    Get the asymptotic per timestep growth rate
    for a renewal process with a given value of
    :math:`\\mathcal{R}` and a given discrete
    generation interval probability mass vector.

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
    return get_asymptotic_growth_rate_and_age_dist(R, generation_interval_pmf)[0]


def integrate_discrete(
    init_diff_vals: ArrayLike, highest_order_diff_vals: ArrayLike
) -> ArrayLike:
    """
    Integrate (de-difference) the differenced process,
    obtaining the process values :math:`X(t=0), X(t=1), ... , X(t)`
    from the :math:`n^{th}` differences and a set of
    initial process / difference values
    :math:`X(t=0), X^1(t=1), X^2(t=2), ..., X^{(n-1)}(t=n-1)`,
    where :math:`X^k(t)` is the value of the :math:`n^{th}`
    difference at index :math:`t` of the process,
    obtaining a sequence of length equal to the length of
    the provided `highest_order_diff_vals` vector plus
    the order of the process.

    Parameters
    ----------
    init_diff_vals : ArrayLike
        Values of
        :math:`X(t=0), X^1(t=1), X^2(t=2) ..., X^{(n-1)}(t=n-1)`.

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
    inits_by_order = jnp.atleast_1d(init_diff_vals)
    highest_diffs = jnp.atleast_1d(highest_order_diff_vals)
    order = inits_by_order.shape[0]
    n_diffs = highest_diffs.shape[0]

    try:
        batch_shape = broadcast_shapes(
            highest_diffs.shape[1:], inits_by_order.shape[1:]
        )
    except Exception as e:
        raise ValueError(
            "Non-time dimensions "
            "(i.e. dimensions after the first) "
            "for highest_order_diff_vals and init_diff_vals "
            "must be broadcastable together. "
            "Got highest_order_diff_vals of shape "
            f"{highest_diffs.shape} and "
            "init_diff_vals of shape "
            f"{inits_by_order.shape}"
        ) from e

    highest_diffs = jnp.broadcast_to(highest_diffs, (n_diffs,) + batch_shape)
    inits_by_order = jnp.broadcast_to(inits_by_order, (order,) + batch_shape)

    highest_diffs = jnp.concatenate(
        [jnp.zeros_like(inits_by_order), highest_diffs],
        axis=0,
    )

    scan_arrays = (
        jnp.arange(start=order - 1, stop=-1, step=-1),
        jnp.flip(inits_by_order, axis=0),
    )

    integrated, _ = scan(f=_integrate_one_step, init=highest_diffs, xs=scan_arrays)

    return integrated


def _integrate_one_step(
    current_diffs: ArrayLike,
    next_order_and_init: tuple[int, ArrayLike],
) -> tuple[ArrayLike, None]:
    """
    Perform one step of integration
    (de-differencing) for integrate_discrete().

    Helper function passed to :func:`jax.lax.scan()`.

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
    next_diffs = jnp.cumsum(current_diffs.at[next_order, ...].set(next_init), axis=0)
    return next_diffs, None
