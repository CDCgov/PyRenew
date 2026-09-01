"""
Utilities for handling delays
"""

import jax.numpy as jnp
from jax.typing import ArrayLike


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
    reporting_delay_pmf
        PMF of reporting delays. The i-th entry is the probability that
        an event is reported with a delay of i time units.
    n_timepoints
        Number of timepoints in the output array.
    right_truncation_offset
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
