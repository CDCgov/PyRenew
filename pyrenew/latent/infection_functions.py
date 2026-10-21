# numpydoc ignore=GL08

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from pyrenew.convolve import new_convolve_scanner, new_double_convolve_scanner
from pyrenew.transformation import ExpTransform, IdentityTransform


def compute_infections_from_rt(
    I0: ArrayLike,
    Rt: ArrayLike,
    reversed_generation_interval_pmf: ArrayLike,
) -> jnp.ndarray:
    """
    Generate infections according to a
    renewal process with a time-varying
    reproduction number $\\mathcal{R}(t)$

    Parameters
    ----------
    I0
        Array of initial infections of the
        same length as the generation interval
        pmf vector.
    Rt
        Timeseries of $\\mathcal{R}(t)$ values
    reversed_generation_interval_pmf
        discrete probability mass vector
        representing the generation interval
        of the infection process, where the final
        entry represents an infection 1 time unit in the
        past, the second-to-last entry represents
        an infection two time units in the past, etc.

    Returns
    -------
    jnp.ndarray
        The timeseries of infections.
    """
    incidence_func = new_convolve_scanner(
        reversed_generation_interval_pmf, IdentityTransform()
    )

    latest, all_infections = jax.lax.scan(f=incidence_func, init=I0, xs=Rt)

    return all_infections


def logistic_susceptibility_adjustment(
    I_raw_t: float,
    frac_susceptible: float,
    n_population: float,
) -> float:
    """
    Apply the logistic susceptibility
    adjustment to a potential new
    incidence `I_raw_t` proposed in
    equation 6 of `Bhatt et al 2023
    <https://doi.org/10.1093/jrsssa/qnad030>`_.

    Parameters
    ----------
    I_raw_t
        The "unadjusted" incidence at time t,
        i.e. the incidence given an infinite
        number of available susceptible individuals.
    frac_susceptible
        fraction of remaining susceptible individuals
        in the population
    n_population
        Total size of the population.

    Returns
    -------
    float
        The adjusted value of $I(t)$.
    """
    approx_frac_infected = 1 - jnp.exp(-I_raw_t / n_population)
    return n_population * frac_susceptible * approx_frac_infected


def compute_infections_from_rt_with_feedback(
    I0: ArrayLike,
    Rt_raw: ArrayLike,
    infection_feedback_strength: ArrayLike,
    reversed_generation_interval_pmf: ArrayLike,
    reversed_infection_feedback_pmf: ArrayLike,
) -> tuple:
    r"""
    Generate infections according to
    a renewal process with infection
    feedback (generalizing `Asher 2018
    <https://doi.org/10.1016/j.epidem.2017.02.009>`_).

    Parameters
    ----------
    I0
        Array of initial infections of the
        same length as the generation interval
        pmf vector.
    Rt_raw
        Timeseries of raw $\mathcal{R}(t)$ values not
        adjusted by infection feedback
    infection_feedback_strength
        Strength of the infection feedback.
        Either a scalar (constant feedback
        strength in time) or a vector representing
        the infection feedback strength at a
        given point in time.
    reversed_generation_interval_pmf
        discrete probability mass vector
        representing the generation interval
        of the infection process, where the final
        entry represents an infection 1 time unit in the
        past, the second-to-last entry represents
        an infection two time units in the past, etc.
    reversed_infection_feedback_pmf
        discrete probability mass vector
        representing the infection feedback
        process, where the final entry represents
        the relative contribution to infection
        feedback from infections that occurred
        1 time unit in the past, the second-to-last
        entry represents the contribution from infections
        that occurred 2 time units in the past, etc.

    Returns
    -------
    tuple
        A tuple ``(infections, Rt_adjusted)``,
        where `Rt_adjusted` is the infection-feedback-adjusted
        timeseries of the reproduction number $\mathcal{R}(t)$
        and `infections` is the incident infection timeseries.

    Notes
    -----
    This function implements the following renewal process:

    ```math
    \begin{aligned}
    I(t) & = \mathcal{R}(t)\sum_{\tau=1}^{T_g}I(t - \tau)g(\tau) \\
    \mathcal{R}(t) & = \mathcal{R}^u(t)\exp\left(\gamma(t)\
        \sum_{\tau=1}^{T_f}I(t - \tau)f(\tau)\right)
    \end{aligned}
    ```

    where $\mathcal{R}(t)$ is the reproductive number,
    $\gamma(t)$ is the infection feedback strength,
    $T_g$ is the max-length of the
    generation interval, $\mathcal{R}^u(t)$ is the raw reproduction
    number, $f(t)$ is the infection feedback pmf, and $T_f$
    is the max-length of the infection feedback pmf.

    Note that negative $\gamma(t)$ implies
    that recent incident infections reduce $\mathcal{R}(t)$
    below its raw value in the absence of feedback, while
    positive $\gamma$ implies that recent incident infections
    *increase* $\mathcal{R}(t)$ above its raw value, and
    $\gamma(t)=0$ implies no feedback.

    In general, negative $\gamma$ is the more common modeling
    choice, as it can be used to model susceptible depletion,
    reductions in contact rate due to awareness of high incidence,
    et cetera.
    """
    feedback_scanner = new_double_convolve_scanner(
        arrays_to_convolve=(
            reversed_infection_feedback_pmf,
            reversed_generation_interval_pmf,
        ),
        transforms=(ExpTransform(), IdentityTransform()),
    )
    latest, infs_and_R_adj = jax.lax.scan(
        f=feedback_scanner,
        init=I0,
        xs=(infection_feedback_strength, Rt_raw),
    )

    infections, R_adjustment = infs_and_R_adj
    Rt_adjusted = R_adjustment * Rt_raw
    return infections, Rt_adjusted


def _mobility_renewal_step(
    state: tuple[ArrayLike, ArrayLike],
    rt_raw: ArrayLike,
    mobility_matrix: ArrayLike,
    share_present: ArrayLike,
    reversed_generation_interval_pmf: ArrayLike,
    susceptible_depletion: bool,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Advance the mobility-coupled renewal process by one day.

    Parameters
    ----------
    state
        Recent new infections per resident, shape ``(n_gen, n)``, and
        cumulative new infections per resident, shape ``(n,)``.
    rt_raw
        Reproduction number at each location on this day, shape ``(n,)``.
    mobility_matrix
        Fraction of residents of each home region present at each location,
        shape ``(n, n)``.
    share_present
        Share of the people present at each location who live in each home
        region, shape ``(n, n)``.
    reversed_generation_interval_pmf
        Generation interval PMF in reverse order.
    susceptible_depletion
        Whether new infections are scaled by the susceptible fraction.

    Returns
    -------
    tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
        The updated state, and this day's new infections per resident and
        adjusted reproduction number at each location.
    """
    history, cumulative = state
    infectiousness = reversed_generation_interval_pmf @ history
    if susceptible_depletion:
        susceptible = jnp.maximum(1.0 - cumulative, 0.0)
        rt_adjusted = rt_raw * (share_present @ susceptible)
    else:
        susceptible = jnp.ones_like(cumulative)
        rt_adjusted = rt_raw
    new_infections = susceptible * (
        mobility_matrix.T @ (rt_raw * (share_present @ infectiousness))
    )
    history = jnp.concatenate([history[1:], new_infections[jnp.newaxis]], axis=0)
    return (history, cumulative + new_infections), (new_infections, rt_adjusted)


def compute_infections_from_rt_with_mobility(
    I0: ArrayLike,
    Rt_raw: ArrayLike,
    reversed_generation_interval_pmf: ArrayLike,
    mobility_matrix: ArrayLike,
    subpop_fractions: ArrayLike,
    susceptible_depletion: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""
    Generate infections from a renewal process coupled by a fixed mobility matrix.

    Residents of each home region spend part of each day at other locations.
    Transmission happens at each location, at that location's reproduction
    number, among the people present there. Each infection is assigned to
    the home region of the person infected.

    Parameters
    ----------
    I0
        New infections per resident on each day of the initialization
        period, shape ``(n_init, n)``, with ``n_init`` at least the length of
        the generation interval. Every row counts toward susceptible
        depletion. Only the last ``len(reversed_generation_interval_pmf)``
        rows enter the generation-interval sum.
    Rt_raw
        Reproduction number at each location on each day after the
        initialization period, shape ``(n_days, n)``.
    reversed_generation_interval_pmf
        Generation interval PMF in reverse order: the last entry is one day
        back, the second-to-last two days back, and so on.
    mobility_matrix
        Matrix used on every day, shape ``(n, n)``.
        ``mobility_matrix[j, k]`` is the fraction of residents of home
        region ``k`` present at location ``j``. Each column sums to 1.
    subpop_fractions
        Each home region's share of the total population, shape ``(n,)``.
        Sums to 1.
    susceptible_depletion
        If True (default), new infections among residents of each region
        are scaled by the fraction of those residents not yet infected.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        New infections per resident, shape ``(n_days, n)``, and the
        reproduction number at each location after depletion, the same
        shape. Without depletion, the second array is ``Rt_raw``.

    Notes
    -----
    This is Equation 2 of Roy, Clapham, and Mishra (2025),
    <https://doi.org/10.1371/journal.pcbi.1013642>. In infection counts (before
    converting to infections per resident), the equation is:

    ```math
    I_{t,i} = \sum_{j=1}^{n} \frac{m_{ji} S_{t,i}}{\sum_{l=1}^{n} m_{jl} N_l}
    R_{t,j} \left[ \sum_{k=1}^{n} m_{jk} \sum_{\tau} I_{\tau,k}\, g_{t-\tau} \right]
    ```

    where $m_{jk}$ is ``mobility_matrix[j, k]``, $N_k$ is the number of
    residents of region $k$, $R_{t,j}$ is the reproduction number at
    location $j$, $g$ is the generation interval, and $S_{t,i}$ is the
    number of residents of region $i$ not infected before day $t$.

    With ``M = mobility_matrix`` and ``p = subpop_fractions``, one day is:

    ```python
    share_present = M * p[None, :] / (M @ p)[:, None]
    infectiousness = reversed_generation_interval_pmf @ history
    susceptible = jnp.maximum(1 - cumulative, 0)
    new_infections = susceptible * (M.T @ (rt * (share_present @ infectiousness)))
    ```

    ``share_present[j, k]`` is the share of the people present at location
    ``j`` who live in region ``k``. ``cumulative`` counts every day of
    ``I0`` and every earlier day of the result.

    The susceptible fraction is floored at zero. On a day with extreme
    reproduction numbers, new infections can exceed the residents still
    susceptible. That day is not capped, but later days stay non-negative.
    """
    share_present = (
        mobility_matrix
        * subpop_fractions[jnp.newaxis, :]
        / (mobility_matrix @ subpop_fractions)[:, jnp.newaxis]
    )
    n_gen = reversed_generation_interval_pmf.size
    initial_state = (I0[-n_gen:], jnp.sum(I0, axis=0))
    step = functools.partial(
        _mobility_renewal_step,
        mobility_matrix=mobility_matrix,
        share_present=share_present,
        reversed_generation_interval_pmf=reversed_generation_interval_pmf,
        susceptible_depletion=susceptible_depletion,
    )
    _, (infections, Rt_adjusted) = jax.lax.scan(step, initial_state, Rt_raw)
    return infections, Rt_adjusted
