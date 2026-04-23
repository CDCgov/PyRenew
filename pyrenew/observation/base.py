# numpydoc ignore=GL08
"""
Abstract base class for observation processes.

Provides common functionality for observation processes that use convolution
with temporal distributions to connect infections to observed data.
"""

from __future__ import annotations

from abc import abstractmethod

import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike

from pyrenew.arrayutils import require_shape
from pyrenew.convolve import compute_delay_ascertained_incidence
from pyrenew.metaclass import RandomVariable
from pyrenew.time import WeekCycle


class BaseObservationProcess(RandomVariable):
    """
    Abstract base class for observation processes that use convolution
    with temporal distributions.

    This class provides common functionality for connecting infections
    to observed data (e.g., hospital admissions, wastewater concentrations)
    through temporal convolution operations.

    Key features provided:

    - PMF validation (sum to 1, non-negative)
    - Minimum observation day calculation
    - Convolution wrapper with timeline alignment
    - Deterministic quantity tracking

    Subclasses must implement:

    - ``lookback_days()``: Return PMF length for initialization
    - ``infection_resolution()``: Return ``"aggregate"`` or ``"subpop"``
    - ``_predicted_obs()``: Transform infections to predicted values
    - ``sample()``: Apply noise model to predicted observations

    Notes
    -----
    Computing predicted observations on day t requires infection history
    from previous days (determined by the temporal PMF length).
    The first ``len(pmf) - 1`` days have insufficient history and return NaN.

    See Also
    --------
    pyrenew.convolve.compute_delay_ascertained_incidence :
        Underlying convolution function
    pyrenew.metaclass.RandomVariable :
        Base class for all random variables
    """

    def __init__(self, name: str, temporal_pmf_rv: RandomVariable) -> None:
        """
        Initialize base observation process.

        Parameters
        ----------
        name
            Unique name for this observation process. Used to prefix all
            numpyro sample and deterministic site names, enabling multiple
            observations of the same type in a single model.
        temporal_pmf_rv
            The temporal distribution PMF (e.g., delay or shedding distribution).
            Must sample to a 1D array that sums to ~1.0 with non-negative values.
            Subclasses may have additional parameters.

        Notes
        -----
        Subclasses should call ``super().__init__(name, temporal_pmf_rv)``
        in their constructors and may add additional parameters.
        """
        super().__init__(name=name)
        self.temporal_pmf_rv = temporal_pmf_rv

    @abstractmethod
    def lookback_days(self) -> int:
        """
        Return the number of days this observation process needs to look back.

        This determines the minimum n_initialization_points required by the
        latent process when this observation is included in a multi-signal model.

        Returns
        -------
        int
            Number of days of infection history required.

        Notes
        -----
        Delay/shedding PMFs are 0-indexed (effect can occur on day 0), so a
        PMF of length L covers lags 0 to L-1, requiring L-1 initialization
        points. Implementations should return ``len(pmf) - 1``.

        This is used by model builders to automatically compute
        n_initialization_points as ``max(gen_int_length, max(all lookbacks))``.
        """
        pass  # pragma: no cover

    @abstractmethod
    def infection_resolution(self) -> str:
        """
        Return whether this observation uses aggregate or subpop infections.

        Returns one of:

        - ``"aggregate"``: Uses a single aggregated infection trajectory.
          Shape: ``(n_days,)``
        - ``"subpop"``: Uses subpopulation-level infection trajectories.
          Shape: ``(n_days, n_subpops)``, indexed via ``subpop_indices``.

        Returns
        -------
        str
            Either ``"aggregate"`` or ``"subpop"``

        Notes
        -----
        This is used by multi-signal models to route the correct infection
        output to each observation process.
        """
        pass  # pragma: no cover

    def _validate_pmf(
        self,
        pmf: ArrayLike,
        param_name: str,
        atol: float = 1e-6,
    ) -> None:
        """
        Validate that an array is a valid probability mass function.

        Checks:

        - Non-empty array
        - Sums to 1.0 (within tolerance)
        - All non-negative values

        Parameters
        ----------
        pmf
            The PMF array to validate
        param_name
            Name of the parameter (for error messages)
        atol
            Absolute tolerance for sum-to-one check

        Raises
        ------
        ValueError
            If PMF is empty, doesn't sum to 1.0 (within tolerance),
            or contains negative values.
        """
        if pmf.size == 0:
            raise ValueError(f"{param_name} must return non-empty array")

        pmf_sum = jnp.sum(pmf)
        if not jnp.isclose(pmf_sum, 1.0, atol=atol):
            raise ValueError(
                f"{param_name} must sum to 1.0 (±{atol}), got {float(pmf_sum):.6f}"
            )

        if jnp.any(pmf < 0):
            raise ValueError(f"{param_name} must have non-negative values")

    def _validate_dow_effect(
        self,
        dow_effect: ArrayLike,
        param_name: str,
    ) -> None:
        """
        Validate a day-of-week effect vector.

        Checks that the vector has exactly 7 non-negative elements
        (one per day, 0=Monday through 6=Sunday, ISO convention).

        Parameters
        ----------
        dow_effect : ArrayLike
            Day-of-week multiplicative effects to validate.
        param_name : str
            Name of the parameter (for error messages).

        Raises
        ------
        ValueError
            If shape is not (7,) or any values are negative.
        """
        require_shape(dow_effect, (7,), param_name)
        if jnp.any(dow_effect < 0):
            raise ValueError(f"{param_name} must have non-negative values")

    def _validate_week(
        self,
        aggregation: str,
        week: WeekCycle | None,
    ) -> None:
        """
        Validate the ``(aggregation, week)`` pair.

        ``aggregation="weekly"`` requires a :class:`WeekCycle`;
        ``aggregation="daily"`` ignores ``week``.

        Parameters
        ----------
        aggregation
            Observation reporting cadence; one of ``"daily"`` or
            ``"weekly"``.
        week
            Calendar-week anchor; required iff
            ``aggregation == "weekly"``.

        Raises
        ------
        ValueError
            If ``aggregation`` is unrecognized, or if
            ``aggregation == "weekly"`` and ``week`` is ``None``.
        """
        if aggregation not in ("daily", "weekly"):
            raise ValueError(
                f"aggregation must be one of {{'daily', 'weekly'}}, got {aggregation!r}"
            )
        if aggregation == "weekly" and week is None:
            raise ValueError("week is required when aggregation == 'weekly'")

    def _compute_period_offset(
        self,
        first_day_dow: int | None,
        week: WeekCycle | None,
    ) -> int:
        """
        Compute the number of leading daily timepoints to trim so
        that the daily axis aligns to whole weekly periods.

        Parameters
        ----------
        first_day_dow
            Day-of-week index of element 0 of the daily axis
            (0=Monday, 6=Sunday, ISO convention). Required when
            ``week`` is not ``None``.
        week
            Calendar-week anchor. ``None`` indicates daily
            (non-aggregated) observations; in that case the
            offset is ``0``.

        Returns
        -------
        int
            Trim offset in ``[0, 7)``. Returns ``0`` when ``week`` is ``None``.

        Raises
        ------
        ValueError
            If ``week`` is provided but ``first_day_dow`` is ``None``.
        """
        if week is None:
            return 0
        if first_day_dow is None:
            raise ValueError("first_day_dow is required when week is not None")
        return (week.end_dow + 1 - first_day_dow) % 7

    def _convolve_with_alignment(
        self,
        latent_incidence: ArrayLike,
        pmf: ArrayLike,
        p_observed: float = 1.0,
    ) -> tuple[ArrayLike, int]:
        """
        Convolve latent incidence with PMF while maintaining timeline alignment.

        This is a wrapper around ``compute_delay_ascertained_incidence`` that
        always uses ``pad=True`` to ensure day t in the output corresponds to
        day t in the input. The first ``len(pmf) - 1`` days will be NaN.

        Parameters
        ----------
        latent_incidence
            Latent incidence time series (infections, prevalence, etc.).
            Shape: (n_days,)
        pmf
            Delay or shedding PMF. Shape: (n_pmf,)
        p_observed
            Observation probability multiplier. Scales the convolution result.

        Returns
        -------
        tuple[ArrayLike, int]
            - convolved_array : ArrayLike
                Convolved time series with same length as input.
                First ``len(pmf) - 1`` days are NaN.
                Shape: (n_days,)
            - offset : int
                Always 0 when pad=True (maintained for API compatibility)

        Notes
        -----
        For t < len(pmf)-1, there is insufficient history, so output[t] = NaN.

        See Also
        --------
        pyrenew.convolve.compute_delay_ascertained_incidence :
            Underlying function
        """
        return compute_delay_ascertained_incidence(
            latent_incidence=latent_incidence,
            delay_incidence_to_observation_pmf=pmf,
            p_observed_given_incident=p_observed,
            pad=True,  # Maintains timeline alignment
        )

    def _deterministic(self, suffix: str, value: ArrayLike) -> None:
        """
        Track a deterministic quantity in the numpyro execution trace.

        This is a convenience wrapper around ``numpyro.deterministic`` for
        tracking intermediate quantities (e.g., latent admissions, predicted
        concentrations) that are useful for diagnostics and model checking.
        These quantities are stored in MCMC samples and can be used for
        model diagnostics and posterior predictive checks.

        The site name is prefixed with ``self.name`` to ensure uniqueness
        when multiple observations of the same type are used.

        Parameters
        ----------
        suffix
            Suffix for the site name. Will be prefixed with ``self.name``.
            For example, suffix="predicted" becomes "{name}_predicted".
        value
            Value to track. Can be any shape.
        """
        numpyro.deterministic(self._sample_site_name(suffix), value)

    def _sample_site_name(self, suffix: str) -> str:
        """
        Generate a prefixed sample site name.

        Parameters
        ----------
        suffix
            Suffix for the site name (e.g., "obs").

        Returns
        -------
        str
            Full site name with ``self.name`` prefix.
            For example, suffix="obs" returns "{name}_obs".
        """
        return f"{self.name}_{suffix}"

    @abstractmethod
    def _predicted_obs(
        self,
        infections: ArrayLike,
    ) -> ArrayLike:
        """
        Transform infections to predicted observation values.

        This is the core transformation that each observation process must
        implement. It converts infections (from the infection process)
        to predicted values for the observation model.

        Parameters
        ----------
        infections
            Infections from the infection process.
            Shape: (n_days,) for aggregate observations
            Shape: (n_days, n_subpops) for subpop-level observations

        Returns
        -------
        ArrayLike
            Predicted observation values (counts, log-concentrations, etc.).
            Same shape as input, with first len(pmf)-1 days as NaN.

        Notes
        -----
        The transformation is observation-specific:

        - Count observations: ascertainment x delay convolution -> predicted counts
        - Wastewater: shedding convolution -> genome scaling -> dilution -> log

        See Also
        --------
        sample
        """
        pass  # pragma: no cover

    @abstractmethod
    def validate_data(
        self,
        n_total: int,
        n_subpops: int,
        **obs_data: dict[str, object],
    ) -> None:
        """
        Validate observation data before running inference.

        Each observation process validates its own data requirements.
        Called by the model's ``validate_data()`` method with concrete
        (non-traced) values before JAX tracing begins.

        Parameters
        ----------
        n_total
            Total number of time steps (n_init + n_days_post_init).
        n_subpops
            Number of subpopulations.
        **obs_data
            Observation-specific data kwargs (same as passed to ``sample()``,
            minus ``infections`` which comes from the latent process).

        Raises
        ------
        ValueError
            If any data fails validation.
        """
        pass  # pragma: no cover

    def _validate_index_array(
        self, indices: ArrayLike, upper_bound: int, param_name: str
    ) -> None:
        """
        Validate an index array is 1D with non-negative values within bounds.

        Checks that the array is 1D and that all values are non-negative
        integers in ``[0, upper_bound)``. An empty 1D array is a no-op
        and passes the bounds check.

        Parameters
        ----------
        indices
            Index array to validate.
        upper_bound
            Exclusive upper bound for valid indices.
        param_name
            Name of the parameter (for error messages).

        Raises
        ------
        ValueError
            If indices is not 1D, contains negative values, or values
            >= upper_bound.
        """
        indices = jnp.asarray(indices)
        if indices.ndim != 1:
            raise ValueError(
                f"Observation '{self.name}': {param_name} must be 1D, "
                f"got shape {indices.shape}"
            )
        if indices.size == 0:
            return
        if jnp.any(indices < 0):
            raise ValueError(
                f"Observation '{self.name}': {param_name} cannot be negative"
            )
        max_val = jnp.max(indices)
        if max_val >= upper_bound:
            raise ValueError(
                f"Observation '{self.name}': {param_name} contains "
                f"{int(max_val)} >= {upper_bound} ({param_name} upper bound)"
            )

    def _validate_times(self, times: ArrayLike, n_total: int) -> None:
        """
        Validate a times index array.

        Checks that all values are non-negative and within ``[0, n_total)``.

        Parameters
        ----------
        times
            Time indices on the shared time axis.
        n_total
            Total number of time steps.

        Raises
        ------
        ValueError
            If times contains negative values or values >= n_total.
        """
        self._validate_index_array(times, n_total, "times")

    def _validate_subpop_indices(
        self, subpop_indices: ArrayLike, n_subpops: int
    ) -> None:
        """
        Validate a subpopulation index array.

        Checks that all values are non-negative and within ``[0, n_subpops)``.

        Parameters
        ----------
        subpop_indices
            Subpopulation indices (0-indexed).
        n_subpops
            Number of subpopulations.

        Raises
        ------
        ValueError
            If subpop_indices contains negative values or values >= n_subpops.
        """
        self._validate_index_array(subpop_indices, n_subpops, "subpop_indices")

    def _validate_shapes_match(
        self,
        first: ArrayLike,
        second: ArrayLike,
        first_name: str,
        second_name: str,
    ) -> None:
        """
        Validate that two arrays have matching shapes.

        Parameters
        ----------
        first
            First array.
        second
            Second array.
        first_name
            Name of the first parameter (for error messages).
        second_name
            Name of the second parameter (for error messages).

        Raises
        ------
        ValueError
            If the two arrays have different shapes.
        """
        first = jnp.asarray(first)
        second = jnp.asarray(second)
        if first.shape != second.shape:
            raise ValueError(
                f"Observation '{self.name}': {first_name} shape {first.shape} "
                f"must match {second_name} shape {second.shape}"
            )

    def _validate_obs_dense(self, obs: ArrayLike, n_total: int) -> None:
        """
        Validate that obs covers the full shared time axis.

        For dense observations on the shared time axis ``[0, n_total)``,
        obs must be 1D with length equal to ``n_total``. Use NaN to
        mark unobserved timepoints (initialization period or missing
        data).

        Parameters
        ----------
        obs
            Observed data array on the shared time axis.
        n_total
            Total number of time steps (n_init + n_days_post_init).

        Raises
        ------
        ValueError
            If obs is not 1D or its length doesn't equal n_total.
        """
        obs = jnp.asarray(obs)
        if obs.ndim != 1:
            raise ValueError(
                f"Observation '{self.name}': obs must be 1D, got shape {obs.shape}"
            )
        if obs.shape[0] != n_total:
            raise ValueError(
                f"Observation '{self.name}': obs length {obs.shape[0]} "
                f"must equal n_total ({n_total}). "
                f"Pad with NaN for initialization period."
            )

    def _validate_period_end_times(
        self,
        period_end_times: ArrayLike,
        n_total: int,
        offset: int,
        aggregation_period: int,
    ) -> None:
        """
        Validate a period-end-time index array.

        Checks that all values are non-negative, within
        ``[0, n_total)``, and lie on aggregation-period boundaries,
        i.e., ``(t - offset) % aggregation_period ==
        aggregation_period - 1`` for every entry ``t``. When
        ``aggregation_period == 1`` the alignment condition holds
        trivially and only the bounds check runs.

        Parameters
        ----------
        period_end_times
            Daily-axis indices of each observed period's final day.
        n_total
            Total number of time steps.
        offset
            Front-trim offset in daily units, as returned by
            ``_compute_period_offset``. Must be in
            ``[0, aggregation_period)``.
        aggregation_period
            Width of the reporting period in fundamental time units.

        Raises
        ------
        ValueError
            If ``period_end_times`` contains negative values, values
            ``>= n_total``, or entries that fail the alignment check.
        """
        self._validate_index_array(period_end_times, n_total, "period_end_times")
        if aggregation_period == 1:
            return
        period_end_times = jnp.asarray(period_end_times)
        misaligned = (period_end_times - offset) % aggregation_period != (
            aggregation_period - 1
        )
        if jnp.any(misaligned):
            raise ValueError(
                f"Observation '{self.name}': period_end_times must lie on "
                f"aggregation-period boundaries "
                f"(offset={offset}, aggregation_period={aggregation_period}); "
                f"each entry t must satisfy "
                f"(t - offset) % {aggregation_period} == {aggregation_period - 1}."
            )

    @abstractmethod
    def sample(
        self,
        obs: ArrayLike | None = None,
        **kwargs: object,
    ) -> ArrayLike:
        """
        Sample from the observation process.

        Subclasses must implement this method to define the specific
        observation model. Typically calls ``_predicted_obs`` first,
        then applies the noise model.

        Parameters
        ----------
        obs
            Observed data for conditioning, or None for prior predictive sampling.
        **kwargs
            Subclass-specific parameters (e.g., infections from the infection process).

        Returns
        -------
        ArrayLike
            Observed or sampled values from the observation process.
        """
        pass  # pragma: no cover
