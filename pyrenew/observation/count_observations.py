# numpydoc ignore=GL08
"""
Count observations with composable noise models.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from pyrenew.convolve import compute_prop_already_reported
from pyrenew.metaclass import RandomVariable
from pyrenew.observation.base import BaseObservationProcess
from pyrenew.observation.noise import CountNoise
from pyrenew.observation.types import ObservationSample
from pyrenew.time import daily_to_weekly, get_sequential_day_of_week_indices


class CountObservation(BaseObservationProcess):
    """
    Abstract Base class for count observation processes.

    Subclasses map infections to counts through ascertainment x delay convolution
    with composable noise model. Count observations always receive predictions
    on the model's daily time axis and then, if requested, aggregate those
    daily predictions to the observation reporting grid before evaluating the
    likelihood.
    """

    _SUPPORTED_SCHEDULES = ("regular", "irregular")

    def __init__(
        self,
        name: str,
        ascertainment_rate_rv: RandomVariable,
        delay_distribution_rv: RandomVariable,
        noise: CountNoise,
        right_truncation_rv: RandomVariable | None = None,
        day_of_week_rv: RandomVariable | None = None,
        aggregation_period: int = 1,
        reporting_schedule: str = "regular",
        period_end_dow: int | None = None,
    ) -> None:
        """
        Initialize count observation base.

        Parameters
        ----------
        name
            Unique name for this observation process. Used to prefix all
            numpyro sample and deterministic site names.
        ascertainment_rate_rv
            Ascertainment rate in [0, 1] (e.g., IHR, IER).
        delay_distribution_rv
            Delay distribution PMF (must sum to ~1.0).
        noise
            Noise model for count observations (Poisson, NegBin, etc.).
        right_truncation_rv
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
        aggregation_period
            Width of the observation reporting period in days. Must be in
            ``{1, 7}``. ``1`` means no aggregation (daily observations).
            This controls only the scale on which the count likelihood is
            evaluated. It does not control how often the latent Rt temporal
            process samples new parameters.
        reporting_schedule
            Either ``"regular"`` (dense observation array, one entry
            per period, NaN for unobserved periods) or ``"irregular"``
            (sparse observation array with user-supplied period-end
            time indices).
        period_end_dow
            Day-of-week index of each weekly period's final day (e.g., 5
            for MMWR Sunday-Saturday epiweeks; 6 for ISO Monday-Sunday
            weeks). Required when ``aggregation_period == 7``; ignored
            otherwise. Anchors the weekly likelihood: daily predictions
            are bucketed into weeks ending on this day, then summed before
            scoring. (0=Monday, 6=Sunday, ISO convention.)

        Raises
        ------
        ValueError
            If aggregation/reporting parameters are invalid, or if a
            day-of-week effect is combined with ``aggregation_period > 1``
            (within-period structure is destroyed by aggregation).
        """
        super().__init__(name=name, temporal_pmf_rv=delay_distribution_rv)
        self.ascertainment_rate_rv = ascertainment_rate_rv
        self.noise = noise
        self.right_truncation_rv = right_truncation_rv
        self.day_of_week_rv = day_of_week_rv
        self._validate_aggregation_params(aggregation_period, period_end_dow)
        if reporting_schedule not in self._SUPPORTED_SCHEDULES:
            raise ValueError(
                f"reporting_schedule must be one of {self._SUPPORTED_SCHEDULES}, "
                f"got {reporting_schedule!r}"
            )
        if aggregation_period > 1 and day_of_week_rv is not None:
            raise ValueError(
                "day_of_week_rv cannot be combined with aggregation_period > 1; "
                "aggregation destroys within-period structure."
            )
        self.aggregation_period = aggregation_period
        self.reporting_schedule = reporting_schedule
        self.period_end_dow = period_end_dow

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

    def infection_resolution(self) -> str:
        """
        Return required infection resolution.

        Returns
        -------
        str
            "aggregate" or "subpop".
        """
        raise NotImplementedError("Subclasses must implement infection_resolution()")

    def _predicted_obs(
        self,
        infections: ArrayLike,
    ) -> ArrayLike:
        """
        Compute predicted counts via ascertainment x delay convolution.

        Parameters
        ----------
        infections
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
        predicted
            Predicted counts. Shape: (n_timepoints,) or
            (n_timepoints, n_subpops).
        right_truncation_offset
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
        dow_effect = self.day_of_week_rv()
        self._deterministic("day_of_week_effect", dow_effect)
        n_timepoints = predicted.shape[0]
        daily_effect = dow_effect[
            get_sequential_day_of_week_indices(first_day_dow, n_timepoints)
        ]
        if predicted.ndim == 2:
            daily_effect = daily_effect[:, None]
        return predicted * daily_effect

    def _aggregate(
        self,
        predicted_daily: ArrayLike,
        first_day_dow: int | None,
    ) -> ArrayLike:
        """
        Aggregate daily predicted counts to the observation reporting grid.

        When ``aggregation_period == 1`` returns the input unchanged.
        Otherwise sums daily values over non-overlapping fixed-width
        periods anchored by ``period_end_dow``, via
        ``pyrenew.time.daily_to_weekly``. Works on both 1D
        ``(n_total,)`` and 2D ``(n_total, n_subpops)`` inputs.
        This aggregation is part of the observation likelihood path and is
        independent of the parameter cadence used by the latent Rt process.

        Parameters
        ----------
        predicted_daily
            Predicted counts on the daily time axis.
        first_day_dow
            Day-of-week index of element 0 of the daily axis
            (0=Monday, 6=Sunday, ISO convention). Required when
            ``aggregation_period > 1``.

        Returns
        -------
        ArrayLike
            Aggregated counts on the period grid; same trailing
            dimensions as ``predicted_daily``. Returns
            ``predicted_daily`` unchanged when
            ``aggregation_period == 1``.

        Raises
        ------
        ValueError
            If ``aggregation_period > 1`` and ``first_day_dow`` is ``None``.
        """
        if self.aggregation_period == 1:
            return predicted_daily
        if first_day_dow is None:
            raise ValueError("first_day_dow is required when aggregation_period > 1")
        return daily_to_weekly(
            predicted_daily,
            input_data_first_dow=first_day_dow,
            week_start_dow=(self.period_end_dow + 1) % 7,
        )


class PopulationCounts(CountObservation):
    """
    Aggregated count observation.

    Maps aggregate infections to counts through ascertainment x delay
    convolution with composable noise model. Predictions are constructed on
    the daily model axis; ``aggregation_period`` controls whether those
    predictions are scored as daily counts or summed to weekly counts before
    the likelihood.

    Parameters
    ----------
    name
        Unique name for this observation process. Used to prefix all
        numpyro sample and deterministic site names (e.g., "hospital"
        produces sites "hospital_obs", "hospital_predicted").
    ascertainment_rate_rv
        Ascertainment rate in [0, 1] (e.g., IHR, IER).
    delay_distribution_rv
        Delay distribution PMF (must sum to ~1.0).
    noise
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
            f"PopulationCounts(name={self.name!r}, "
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
        period_end_times: ArrayLike | None = None,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> None:
        """
        Validate aggregated count observation data.

        Parameters
        ----------
        n_total
            Total number of daily time steps (``n_init + n_days_post_init``).
        n_subpops
            Number of subpopulations (unused for aggregate observations).
        obs
            Observed counts. Shape depends on ``reporting_schedule``:
            ``"regular"`` expects a dense array of length ``n_total // P``
            after front-trim, with NaN for unobserved periods;
            ``"irregular"`` expects an array matching ``period_end_times``.
        period_end_times
            Daily-axis indices of each observed period's final day. Required
            for ``reporting_schedule="irregular"``.
        first_day_dow
            Day-of-week index of element 0 of the shared time axis
            (0=Monday, 6=Sunday, ISO convention). Required when
            ``aggregation_period > 1`` so weekly observation periods can be
            aligned to the shared daily model axis.
        **kwargs
            Additional keyword arguments (ignored).

        Raises
        ------
        ValueError
            If obs length or period_end_times fail their respective
            checks, or if ``first_day_dow`` is missing when
            ``aggregation_period > 1``.
        """
        P = self.aggregation_period

        if self.reporting_schedule == "regular":
            if obs is None:
                return
            if P == 1:
                self._validate_obs_dense(obs, n_total)
                return
            if first_day_dow is None:
                raise ValueError(
                    f"Observation '{self.name}': first_day_dow is required "
                    f"when aggregation_period == {P}"
                )
            offset = self._compute_period_offset(first_day_dow, P, self.period_end_dow)
            n_periods = (n_total - offset) // P
            obs = jnp.asarray(obs)
            if obs.ndim != 1:
                raise ValueError(
                    f"Observation '{self.name}': obs must be 1D, got shape {obs.shape}"
                )
            if obs.shape[0] != n_periods:
                raise ValueError(
                    f"Observation '{self.name}': obs length {obs.shape[0]} "
                    f"must equal n_periods ({n_periods}). "
                    f"Pad with NaN for unobserved periods."
                )
            return

        if period_end_times is None:
            return
        if P > 1 and first_day_dow is None:
            raise ValueError(
                f"Observation '{self.name}': first_day_dow is required "
                f"when aggregation_period == {P}"
            )
        offset = self._compute_period_offset(first_day_dow, P, self.period_end_dow)
        self._validate_period_end_times(period_end_times, n_total, offset, P)
        if obs is not None:
            self._validate_shapes_match(
                obs, period_end_times, "obs", "period_end_times"
            )

    def sample(
        self,
        infections: ArrayLike,
        obs: ArrayLike | None = None,
        right_truncation_offset: int | None = None,
        first_day_dow: int | None = None,
        period_end_times: ArrayLike | None = None,
    ) -> ObservationSample:
        """
        Sample aggregated counts.

        Daily transforms (right-truncation, day-of-week) run on the
        daily axis. When ``aggregation_period > 1`` the daily
        predictions are summed onto the reporting-period grid before
        the noise model. Likelihood path depends on
        ``reporting_schedule``: ``"regular"`` uses a dense-with-NaN
        array plus a mask; ``"irregular"`` fancy-indexes the
        aggregated array at period indices derived from
        ``period_end_times``.

        ``aggregation_period`` describes the observation scale only. The
        latent infection process may use daily or coarser Rt parameter
        cadence, but by the time this method is called it supplies infections
        on the daily model axis.

        Parameters
        ----------
        infections
            Aggregate infections from the infection process.
            Shape ``(n_total,)``.
        obs
            Observed counts. Shape depends on ``reporting_schedule``:
            ``"regular"`` expects a dense array on the period grid
            with NaN for unobserved periods; ``"irregular"`` expects
            an array of the same length as ``period_end_times``.
            ``None`` for prior predictive sampling.
        right_truncation_offset
            If provided (and ``right_truncation_rv`` was set at
            construction), apply right-truncation adjustment to the
            daily predictions.
        first_day_dow
            Day-of-week index of the first timepoint on the shared
            time axis (0=Monday, 6=Sunday, ISO convention). Required
            when ``day_of_week_rv`` was set at construction or when
            ``aggregation_period > 1``. This aligns observation-level
            day-of-week effects or weekly aggregation to the shared daily
            model axis.
        period_end_times
            Daily-axis indices of each observed period's final day.
            Required when ``reporting_schedule == "irregular"``.

        Returns
        -------
        ObservationSample
            Named tuple with ``observed`` (sampled/conditioned counts)
            and ``predicted`` (predictions on the reporting-period
            grid; equal to daily predictions when
            ``aggregation_period == 1``).
        """
        predicted_daily = self._predicted_obs(infections)
        if self.day_of_week_rv is not None:
            if first_day_dow is None:
                raise ValueError(
                    "first_day_dow is required when day_of_week_rv is set."
                )
            predicted_daily = self._apply_day_of_week(predicted_daily, first_day_dow)
        if self.right_truncation_rv is not None and right_truncation_offset is not None:
            predicted_daily = self._apply_right_truncation(
                predicted_daily, right_truncation_offset
            )

        predicted = self._aggregate(predicted_daily, first_day_dow)
        if self.aggregation_period > 1:
            self._deterministic("predicted_daily", predicted_daily)
        self._deterministic("predicted", predicted)

        if self.reporting_schedule == "regular":
            valid_pred = ~jnp.isnan(predicted)
            if obs is not None:
                valid_obs = ~jnp.isnan(obs)
                mask = valid_pred & valid_obs
            else:
                mask = valid_pred

            # JAX evaluates log_prob for all elements even when mask excludes
            # them from the likelihood sum. Replace NaN with safe values to
            # avoid NaN propagation; mask=False ensures they do not affect
            # inference.
            safe_predicted = jnp.where(jnp.isnan(predicted), 1.0, predicted)
            safe_obs = None
            if obs is not None:
                safe_obs = jnp.where(jnp.isnan(obs), safe_predicted, obs)

            observed = self.noise.sample(
                name=self._sample_site_name("obs"),
                predicted=safe_predicted,
                obs=safe_obs,
                mask=mask,
            )
        else:
            if period_end_times is None:
                raise ValueError(
                    f"Observation '{self.name}': period_end_times is "
                    f"required when reporting_schedule == 'irregular'"
                )
            P = self.aggregation_period
            offset = self._compute_period_offset(first_day_dow, P, self.period_end_dow)
            period_idx = (jnp.asarray(period_end_times) - offset - (P - 1)) // P
            predicted_obs = predicted[period_idx]

            observed = self.noise.sample(
                name=self._sample_site_name("obs"),
                predicted=predicted_obs,
                obs=obs,
            )

        return ObservationSample(observed=observed, predicted=predicted)


class SubpopulationCounts(CountObservation):
    """
    Subpopulation-level count observation.

    Maps subpopulation-level infections to counts through
    ascertainment x delay convolution with composable noise model.
    Predictions are constructed on the daily model axis for each
    subpopulation; ``aggregation_period`` controls whether those predictions
    are scored as daily counts or summed to weekly counts before the
    likelihood.

    Parameters
    ----------
    name
        Unique name for this observation process. Used to prefix all
        numpyro sample and deterministic site names.
    ascertainment_rate_rv
        Ascertainment rate in [0, 1].
    delay_distribution_rv
        Delay distribution PMF (must sum to ~1.0).
    noise
        Noise model (PoissonNoise, NegativeBinomialNoise, etc.).
    """

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SubpopulationCounts(name={self.name!r}, "
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
        obs: ArrayLike | None = None,
        period_end_times: ArrayLike | None = None,
        first_day_dow: int | None = None,
        subpop_indices: ArrayLike | None = None,
        **kwargs: object,
    ) -> None:
        """
        Validate subpopulation-level count observation data.

        Parameters
        ----------
        n_total
            Total number of daily time steps (``n_init + n_days_post_init``).
        n_subpops
            Number of subpopulations.
        obs
            Observed counts. For ``reporting_schedule="regular"``
            has shape ``(n_periods, n_observed_subpops)`` with NaN
            for unobserved periods. For
            ``reporting_schedule="irregular"`` has shape ``(n_obs,)``
            matching ``period_end_times`` and ``subpop_indices``.
        period_end_times
            Daily-axis indices of each observed period's final day.
            Required for ``reporting_schedule="irregular"``.
        first_day_dow
            Day-of-week index of element 0 of the shared time axis
            (0=Monday, 6=Sunday, ISO convention). Required when
            ``aggregation_period > 1`` so weekly observation periods can be
            aligned to the shared daily model axis.
        subpop_indices
            Subpopulation indices (0-indexed). For
            ``reporting_schedule="regular"``: shape
            ``(n_observed_subpops,)`` selecting which subpopulation
            columns appear in ``obs``. For
            ``reporting_schedule="irregular"``: shape ``(n_obs,)``
            with one subpopulation per observation.
        **kwargs
            Additional keyword arguments (ignored).

        Raises
        ------
        ValueError
            If any index array is out of bounds, any shape check
            fails, or ``first_day_dow`` is missing when
            ``aggregation_period > 1``.
        """
        P = self.aggregation_period

        if subpop_indices is not None:
            self._validate_subpop_indices(subpop_indices, n_subpops)

        if self.reporting_schedule == "regular":
            if obs is None:
                return
            if P > 1 and first_day_dow is None:
                raise ValueError(
                    f"Observation '{self.name}': first_day_dow is required "
                    f"when aggregation_period == {P}"
                )
            offset = self._compute_period_offset(first_day_dow, P, self.period_end_dow)
            n_periods = n_total if P == 1 else (n_total - offset) // P
            obs = jnp.asarray(obs)
            if obs.ndim != 2:
                raise ValueError(
                    f"Observation '{self.name}': regular-schedule obs must "
                    f"be 2D (n_periods, n_observed_subpops); got shape {obs.shape}"
                )
            if obs.shape[0] != n_periods:
                raise ValueError(
                    f"Observation '{self.name}': obs dimension 0 length "
                    f"{obs.shape[0]} must equal n_periods ({n_periods}). "
                    f"Pad with NaN for unobserved periods."
                )
            if subpop_indices is not None:
                n_observed = jnp.asarray(subpop_indices).shape[0]
                if obs.shape[1] != n_observed:
                    raise ValueError(
                        f"Observation '{self.name}': obs dimension 1 length "
                        f"{obs.shape[1]} must equal len(subpop_indices) "
                        f"({n_observed})"
                    )
            return

        if period_end_times is None:
            return
        if P > 1 and first_day_dow is None:
            raise ValueError(
                f"Observation '{self.name}': first_day_dow is required "
                f"when aggregation_period == {P}"
            )
        offset = self._compute_period_offset(first_day_dow, P, self.period_end_dow)
        self._validate_period_end_times(period_end_times, n_total, offset, P)
        if obs is not None:
            self._validate_shapes_match(
                obs, period_end_times, "obs", "period_end_times"
            )
        if subpop_indices is not None:
            self._validate_shapes_match(
                subpop_indices,
                period_end_times,
                "subpop_indices",
                "period_end_times",
            )

    def sample(
        self,
        infections: ArrayLike,
        obs: ArrayLike | None = None,
        right_truncation_offset: int | None = None,
        first_day_dow: int | None = None,
        period_end_times: ArrayLike | None = None,
        subpop_indices: ArrayLike | None = None,
    ) -> ObservationSample:
        """
        Sample subpopulation-level counts.

        Daily transforms (right-truncation, day-of-week) run on the
        daily axis. When ``aggregation_period > 1`` the daily
        predictions are summed onto the reporting-period grid before
        the noise model. Likelihood path depends on
        ``reporting_schedule``: ``"regular"`` selects the observed
        subpopulation columns and uses a dense-with-NaN array plus
        a mask; ``"irregular"`` fancy-indexes the aggregated array
        at period indices derived from ``period_end_times``.

        ``aggregation_period`` describes the observation scale only. The
        latent infection process may use daily or coarser Rt parameter
        cadence, but by the time this method is called it supplies infections
        on the daily model axis.

        Parameters
        ----------
        infections
            Subpopulation-level infections from the infection process.
            Shape ``(n_total, n_subpops)``.
        obs
            Observed counts. For ``reporting_schedule="regular"``:
            shape ``(n_periods, n_observed_subpops)`` with NaN for
            unobserved periods. For
            ``reporting_schedule="irregular"``: shape ``(n_obs,)``
            matching ``period_end_times`` and ``subpop_indices``.
            ``None`` for prior predictive sampling.
        right_truncation_offset
            If provided (and ``right_truncation_rv`` was set at
            construction), apply right-truncation adjustment to the
            daily predictions.
        first_day_dow
            Day-of-week index of the first timepoint on the shared
            time axis (0=Monday, 6=Sunday, ISO convention). Required
            when ``day_of_week_rv`` was set at construction or when
            ``aggregation_period > 1``. This aligns observation-level
            day-of-week effects or weekly aggregation to the shared daily
            model axis.
        period_end_times
            Daily-axis indices of each observed period's final day.
            Required when ``reporting_schedule == "irregular"``.
        subpop_indices
            Subpopulation indices (0-indexed). Required. For
            ``reporting_schedule="regular"``: shape
            ``(n_observed_subpops,)`` selecting which subpopulation
            columns of the aggregated array enter the likelihood.
            For ``reporting_schedule="irregular"``: shape ``(n_obs,)``
            with one subpopulation per observation.

        Returns
        -------
        ObservationSample
            Named tuple with ``observed`` (sampled/conditioned counts)
            and ``predicted`` (predictions on the reporting-period
            grid, shape ``(n_periods, n_subpops)``; equal to daily
            predictions when ``aggregation_period == 1``).
        """
        if subpop_indices is None:
            raise ValueError(f"Observation '{self.name}': subpop_indices is required.")

        predicted_daily = self._predicted_obs(infections)
        if self.day_of_week_rv is not None:
            if first_day_dow is None:
                raise ValueError(
                    "first_day_dow is required when day_of_week_rv is set."
                )
            predicted_daily = self._apply_day_of_week(predicted_daily, first_day_dow)
        if self.right_truncation_rv is not None and right_truncation_offset is not None:
            predicted_daily = self._apply_right_truncation(
                predicted_daily, right_truncation_offset
            )

        predicted = self._aggregate(predicted_daily, first_day_dow)
        if self.aggregation_period > 1:
            self._deterministic("predicted_daily", predicted_daily)
        self._deterministic("predicted", predicted)

        if self.reporting_schedule == "regular":
            predicted_selected = predicted[:, subpop_indices]

            valid_pred = ~jnp.isnan(predicted_selected)
            if obs is not None:
                valid_obs = ~jnp.isnan(obs)
                mask = valid_pred & valid_obs
            else:
                mask = valid_pred

            # JAX evaluates log_prob for all elements even when mask excludes
            # them from the likelihood sum. Replace NaN with safe values to
            # avoid NaN propagation; mask=False ensures they do not affect
            # inference.
            safe_predicted = jnp.where(
                jnp.isnan(predicted_selected), 1.0, predicted_selected
            )
            safe_obs = None
            if obs is not None:
                safe_obs = jnp.where(jnp.isnan(obs), safe_predicted, obs)

            observed = self.noise.sample(
                name=self._sample_site_name("obs"),
                predicted=safe_predicted,
                obs=safe_obs,
                mask=mask,
            )
        else:
            if period_end_times is None:
                raise ValueError(
                    f"Observation '{self.name}': period_end_times is "
                    f"required when reporting_schedule == 'irregular'"
                )
            P = self.aggregation_period
            offset = self._compute_period_offset(first_day_dow, P, self.period_end_dow)
            period_idx = (jnp.asarray(period_end_times) - offset - (P - 1)) // P
            predicted_obs = predicted[period_idx, subpop_indices]

            observed = self.noise.sample(
                name=self._sample_site_name("obs"),
                predicted=predicted_obs,
                obs=obs,
            )

        return ObservationSample(observed=observed, predicted=predicted)
