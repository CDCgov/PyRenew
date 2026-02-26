# numpydoc ignore=GL08
"""
Count observations with composable noise models.

Ascertainment x delay convolution with pluggable noise (Poisson, Negative Binomial, etc.).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from pyrenew.arrayutils import tile_until_n
from pyrenew.convolve import compute_prop_already_reported
from pyrenew.metaclass import RandomVariable
from pyrenew.observation.base import BaseObservationProcess
from pyrenew.observation.noise import CountNoise
from pyrenew.observation.types import ObservationSample
from pyrenew.time import validate_dow


class CountBase(BaseObservationProcess):
    """
    Base class for count observation processes.

    Implements ascertainment x delay convolution with pluggable noise model.
    """

    def __init__(
        self,
        name: str,
        ascertainment_rate_rv: RandomVariable,
        delay_distribution_rv: RandomVariable,
        noise: CountNoise,
        right_truncation_rv: RandomVariable | None = None,
        day_of_week_rv: RandomVariable | None = None,
    ) -> None:
        """
        Initialize count observation base.

        Parameters
        ----------
        name : str
            Unique name for this observation process. Used to prefix all
            numpyro sample and deterministic site names.
        ascertainment_rate_rv : RandomVariable
            Ascertainment rate in [0, 1] (e.g., IHR, IER).
        delay_distribution_rv : RandomVariable
            Delay distribution PMF (must sum to ~1.0).
        noise : CountNoise
            Noise model for count observations (Poisson, NegBin, etc.).
        right_truncation_rv : RandomVariable | None
            Optional reporting delay PMF for right-truncation adjustment.
            When provided (along with ``right_truncation_offset`` at sample
            time), predicted counts are scaled down for recent timepoints
            to account for incomplete reporting.
        day_of_week_rv : RandomVariable | None
            Optional day-of-week multiplicative effect. Must sample to
            shape (7,) with non-negative values, where entry j is the
            multiplier for day-of-week j (0=Monday, 6=Sunday, ISO
            convention). An effect of 1.0 means no adjustment for that
            day. Values summing to 7.0 preserve weekly totals and keep
            the ascertainment rate interpretable; other sums rescale
            overall predicted counts. When provided (along with
            ``first_day_dow`` at sample time), predicted counts are
            scaled by a periodic weekly pattern.
        """
        super().__init__(name=name, temporal_pmf_rv=delay_distribution_rv)
        self.ascertainment_rate_rv = ascertainment_rate_rv
        self.noise = noise
        self.right_truncation_rv = right_truncation_rv
        self.day_of_week_rv = day_of_week_rv

    def validate(self) -> None:
        """
        Validate observation parameters.

        Raises
        ------
        ValueError
            If delay PMF invalid, ascertainment rate outside [0,1],
            or noise params invalid.
        """
        delay_pmf = self.temporal_pmf_rv()
        self._validate_pmf(delay_pmf, "delay_distribution_rv")

        ascertainment_rate = self.ascertainment_rate_rv()
        if jnp.any(ascertainment_rate < 0) or jnp.any(ascertainment_rate > 1):
            raise ValueError(
                "ascertainment_rate_rv must be in [0, 1], "
                "got value(s) outside this range"
            )

        self.noise.validate()

        if self.right_truncation_rv is not None:
            rt_pmf = self.right_truncation_rv()
            self._validate_pmf(rt_pmf, "right_truncation_rv")

        if self.day_of_week_rv is not None:
            dow_effect = self.day_of_week_rv()
            self._validate_dow_effect(dow_effect, "day_of_week_rv")

    def lookback_days(self) -> int:
        """
        Return required lookback days for this observation.

        Delay PMFs are 0-indexed (delay can be 0), so a PMF of length L
        covers delays 0 to L-1, requiring L-1 initialization points.

        Returns
        -------
        int
            Length of delay distribution PMF minus 1.
        """
        return len(self.temporal_pmf_rv()) - 1

    def _predicted_obs(
        self,
        infections: ArrayLike,
    ) -> ArrayLike:
        """
        Compute predicted counts via ascertainment x delay convolution.

        Parameters
        ----------
        infections : ArrayLike
            Infections from the infection process.
            Shape: (n_days,) for aggregate
            Shape: (n_days, n_subpops) for subpop-level

        Returns
        -------
        ArrayLike
            Predicted counts with timeline alignment.
            Same shape as input.
            First len(delay_pmf)-1 days are NaN.
        """
        delay_pmf = self.temporal_pmf_rv()
        ascertainment_rate = self.ascertainment_rate_rv()

        if infections.ndim == 1:
            return self._convolve_with_alignment(
                infections, delay_pmf, ascertainment_rate
            )[0]
        return jax.vmap(
            lambda col: self._convolve_with_alignment(
                col, delay_pmf, ascertainment_rate
            )[0],
            in_axes=1,
            out_axes=1,
        )(infections)

    def _apply_right_truncation(
        self,
        predicted: ArrayLike,
        right_truncation_offset: int,
    ) -> ArrayLike:
        """
        Apply right-truncation adjustment to predicted counts.

        Scales predicted counts by the proportion already reported
        at each timepoint.

        Parameters
        ----------
        predicted : ArrayLike
            Predicted counts. Shape: (n_timepoints,) or
            (n_timepoints, n_subpops).
        right_truncation_offset : int
            Number of additional reporting days that have occurred since the last observation. 0 implies only 0-delay reports have arrived for the last observed timepoint, 1 implies 0 and 1 delay reports have arrived, et cetera.

        Returns
        -------
        ArrayLike
            Adjusted predicted counts, same shape as input.

        Notes
        -----
        Assumes a single truncation PMF shared across all subpopulations.
        The 1D proportion array is broadcast to match 2D predicted counts.
        """
        trunc_pmf = self.right_truncation_rv()
        n_timepoints = predicted.shape[0]
        delay_support = trunc_pmf.shape[0] - right_truncation_offset
        if n_timepoints < delay_support:
            raise ValueError(
                f"Observation window length ({n_timepoints}) must be >= "
                f"delay distribution support minus right_truncation_offset "
                f"({delay_support})."
            )
        prop = compute_prop_already_reported(
            trunc_pmf, n_timepoints, right_truncation_offset
        )
        self._deterministic("prop_already_reported", prop)
        if predicted.ndim == 2:
            prop = prop[:, None]
        return predicted * prop

    def _apply_day_of_week(
        self,
        predicted: ArrayLike,
        first_day_dow: int,
    ) -> ArrayLike:
        """
        Apply day-of-week multiplicative adjustment to predicted counts.

        Tiles a 7-element effect vector across the full time axis,
        aligned to the calendar via ``first_day_dow``. NaN values
        in the initialization period propagate unchanged (NaN * effect = NaN),
        which is correct since masked days are excluded from the likelihood.

        Parameters
        ----------
        predicted : ArrayLike
            Predicted counts. Shape: (n_timepoints,) or
            (n_timepoints, n_subpops).
        first_day_dow : int
            Day of the week for element 0 of the time axis
            (0=Monday, 6=Sunday, ISO convention).

        Returns
        -------
        ArrayLike
            Adjusted predicted counts, same shape as input.
        """
        validate_dow(first_day_dow, "first_day_dow")
        dow_effect = self.day_of_week_rv()
        n_timepoints = predicted.shape[0]
        daily_effect = tile_until_n(dow_effect, n_timepoints, offset=first_day_dow)
        self._deterministic("day_of_week_effect", daily_effect)
        if predicted.ndim == 2:
            daily_effect = daily_effect[:, None]
        return predicted * daily_effect


class Counts(CountBase):
    """
    Aggregated count observation.

    Maps aggregate infections to counts through ascertainment x delay
    convolution with composable noise model.

    Parameters
    ----------
    name : str
        Unique name for this observation process. Used to prefix all
        numpyro sample and deterministic site names (e.g., "hospital"
        produces sites "hospital_obs", "hospital_predicted").
    ascertainment_rate_rv : RandomVariable
        Ascertainment rate in [0, 1] (e.g., IHR, IER).
    delay_distribution_rv : RandomVariable
        Delay distribution PMF (must sum to ~1.0).
    noise : CountNoise
        Noise model (PoissonNoise, NegativeBinomialNoise, etc.).
    """

    def infection_resolution(self) -> str:
        """
        Return "aggregate" for aggregated observations.

        Returns
        -------
        str
            The string "aggregate".
        """
        return "aggregate"

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Counts(name={self.name!r}, "
            f"ascertainment_rate_rv={self.ascertainment_rate_rv!r}, "
            f"delay_distribution_rv={self.temporal_pmf_rv!r}, "
            f"noise={self.noise!r}, "
            f"right_truncation_rv={self.right_truncation_rv!r}, "
            f"day_of_week_rv={self.day_of_week_rv!r})"
        )

    def validate_data(
        self,
        n_total: int,
        n_subpops: int,
        obs: ArrayLike | None = None,
        **kwargs,
    ) -> None:
        """
        Validate aggregated count observation data.

        Parameters
        ----------
        n_total : int
            Total number of time steps (n_init + n_days_post_init).
        n_subpops : int
            Number of subpopulations (unused for aggregate observations).
        obs : ArrayLike | None
            Observed counts on shared time axis. Shape: (n_total,).
        **kwargs
            Additional keyword arguments (ignored).

        Raises
        ------
        ValueError
            If obs length doesn't match n_total.
        """
        if obs is not None:
            self._validate_obs_dense(obs, n_total)

    def sample(
        self,
        infections: ArrayLike,
        obs: ArrayLike | None = None,
        right_truncation_offset: int | None = None,
        first_day_dow: int | None = None,
    ) -> ObservationSample:
        """
        Sample aggregated counts.

        Both infections and obs use a shared time axis [0, n_total) where
        n_total = n_init + n_days. NaN in obs marks unobserved timepoints
        (initialization period or missing data).

        Parameters
        ----------
        infections : ArrayLike
            Aggregate infections from the infection process.
            Shape: (n_total,) where n_total = n_init + n_days.
        obs : ArrayLike | None
            Observed counts on shared time axis. Shape: (n_total,).
            Use NaN for initialization period and any missing observations.
            None for prior predictive sampling.
        right_truncation_offset : int | None
            If provided (and ``right_truncation_rv`` was set at construction),
            apply right-truncation adjustment to predicted counts.
        first_day_dow : int | None
            Day of the week for the first timepoint on the shared time
            axis (0=Monday, 6=Sunday, ISO convention). Required when
            ``day_of_week_rv`` was set at construction. Use
            ``model.compute_first_day_dow(obs_start_dow)`` to convert
            from the day of the week of the first observation.

        Returns
        -------
        ObservationSample
            Named tuple with `observed` (sampled/conditioned counts) and
            `predicted` (predicted counts before noise, shape: n_total).
        """
        predicted_counts = self._predicted_obs(infections)
        if self.day_of_week_rv is not None and first_day_dow is not None:
            predicted_counts = self._apply_day_of_week(predicted_counts, first_day_dow)
        if self.right_truncation_rv is not None and right_truncation_offset is not None:
            predicted_counts = self._apply_right_truncation(
                predicted_counts, right_truncation_offset
            )
        self._deterministic("predicted", predicted_counts)

        # Compute mask: True where observation contributes to likelihood.
        # NaN in predictions (initialization period) or obs (missing data)
        # are excluded via mask.
        valid_pred = ~jnp.isnan(predicted_counts)
        if obs is not None:
            valid_obs = ~jnp.isnan(obs)
            mask = valid_pred & valid_obs
        else:
            mask = valid_pred

        # JAX evaluates log_prob for all array elements even when mask
        # excludes them from the likelihood sum. Replace NaN with safe values
        # to avoid NaN propagation in JAX's computation graph. These values
        # do not affect inference since mask=False excludes them.
        safe_predicted = jnp.where(jnp.isnan(predicted_counts), 1.0, predicted_counts)
        safe_obs = None
        if obs is not None:
            safe_obs = jnp.where(jnp.isnan(obs), safe_predicted, obs)

        observed = self.noise.sample(
            name=self._sample_site_name("obs"),
            predicted=safe_predicted,
            obs=safe_obs,
            mask=mask,
        )

        return ObservationSample(observed=observed, predicted=predicted_counts)


class CountsBySubpop(CountBase):
    """
    Subpopulation-level count observation.

    Maps subpopulation-level infections to counts through
    ascertainment x delay convolution with composable noise model.

    Parameters
    ----------
    name : str
        Unique name for this observation process. Used to prefix all
        numpyro sample and deterministic site names.
    ascertainment_rate_rv : RandomVariable
        Ascertainment rate in [0, 1].
    delay_distribution_rv : RandomVariable
        Delay distribution PMF (must sum to ~1.0).
    noise : CountNoise
        Noise model (PoissonNoise, NegativeBinomialNoise, etc.).
    """

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CountsBySubpop(name={self.name!r}, "
            f"ascertainment_rate_rv={self.ascertainment_rate_rv!r}, "
            f"delay_distribution_rv={self.temporal_pmf_rv!r}, "
            f"noise={self.noise!r}, "
            f"right_truncation_rv={self.right_truncation_rv!r}, "
            f"day_of_week_rv={self.day_of_week_rv!r})"
        )

    def infection_resolution(self) -> str:
        """
        Return "subpop" for subpopulation-level observations.

        Returns
        -------
        str
            The string "subpop".
        """
        return "subpop"

    def validate_data(
        self,
        n_total: int,
        n_subpops: int,
        times: ArrayLike | None = None,
        subpop_indices: ArrayLike | None = None,
        obs: ArrayLike | None = None,
        **kwargs,
    ) -> None:
        """
        Validate subpopulation-level count observation data.

        Parameters
        ----------
        n_total : int
            Total number of time steps (n_init + n_days_post_init).
        n_subpops : int
            Number of subpopulations.
        times : ArrayLike | None
            Day index for each observation on the shared time axis.
        subpop_indices : ArrayLike | None
            Subpopulation index for each observation (0-indexed).
        obs : ArrayLike | None
            Observed counts (n_obs,).
        **kwargs
            Additional keyword arguments (ignored).

        Raises
        ------
        ValueError
            If times or subpop_indices are out of bounds, or if
            obs and times have mismatched lengths.
        """
        if times is not None:
            self._validate_times(times, n_total)
            if obs is not None:
                self._validate_obs_times_shape(obs, times)
        if subpop_indices is not None:
            self._validate_subpop_indices(subpop_indices, n_subpops)

    def sample(
        self,
        infections: ArrayLike,
        times: ArrayLike,
        subpop_indices: ArrayLike,
        obs: ArrayLike | None = None,
        right_truncation_offset: int | None = None,
        first_day_dow: int | None = None,
    ) -> ObservationSample:
        """
        Sample subpopulation-level counts.

        Times are on the shared time axis [0, n_total) where
        n_total = n_init + n_days. This method performs direct indexing
        without any offset adjustment.

        Parameters
        ----------
        infections : ArrayLike
            Subpopulation-level infections from the infection process.
            Shape: (n_total, n_subpops)
        times : ArrayLike
            Day index for each observation on the shared time axis.
            Must be in range [0, n_total). Shape: (n_obs,)
        subpop_indices : ArrayLike
            Subpopulation index for each observation (0-indexed).
            Shape: (n_obs,)
        obs : ArrayLike | None
            Observed counts (n_obs,), or None for prior sampling.
        right_truncation_offset : int | None
            If provided (and ``right_truncation_rv`` was set at construction),
            apply right-truncation adjustment to predicted counts.
        first_day_dow : int | None
            Day of the week for the first timepoint on the shared time
            axis (0=Monday, 6=Sunday, ISO convention). Required when
            ``day_of_week_rv`` was set at construction. Use
            ``model.compute_first_day_dow(obs_start_dow)`` to convert
            from the day of the week of the first observation.

        Returns
        -------
        ObservationSample
            Named tuple with `observed` (sampled/conditioned counts) and
            `predicted` (predicted counts before noise, shape: n_total x n_subpops).
        """
        predicted_counts = self._predicted_obs(infections)
        if self.day_of_week_rv is not None and first_day_dow is not None:
            predicted_counts = self._apply_day_of_week(predicted_counts, first_day_dow)
        if self.right_truncation_rv is not None and right_truncation_offset is not None:
            predicted_counts = self._apply_right_truncation(
                predicted_counts, right_truncation_offset
            )
        self._deterministic("predicted", predicted_counts)

        # Direct indexing on shared time axis - no offset needed
        predicted_obs = predicted_counts[times, subpop_indices]

        observed = self.noise.sample(
            name=self._sample_site_name("obs"),
            predicted=predicted_obs,
            obs=obs,
        )

        return ObservationSample(observed=observed, predicted=predicted_counts)
