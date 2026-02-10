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

from pyrenew.convolve import compute_delay_ascertained_incidence
from pyrenew.metaclass import RandomVariable


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

    - ``validate()``: Validate parameters (call ``_validate_pmf()`` for PMFs)
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
        name : str
            Unique name for this observation process. Used to prefix all
            numpyro sample and deterministic site names, enabling multiple
            observations of the same type in a single model.
        temporal_pmf_rv : RandomVariable
            The temporal distribution PMF (e.g., delay or shedding distribution).
            Must sample to a 1D array that sums to ~1.0 with non-negative values.
            Subclasses may have additional parameters.

        Notes
        -----
        Subclasses should call ``super().__init__(name, temporal_pmf_rv)``
        in their constructors and may add additional parameters.
        """
        self.name = name
        self.temporal_pmf_rv = temporal_pmf_rv

    @abstractmethod
    def validate(self) -> None:
        """
        Validate observation process parameters.

        Subclasses must implement this method to validate all parameters.
        Typically this involves calling ``_validate_pmf()`` for the PMF
        and adding any additional parameter-specific validation.

        Raises
        ------
        ValueError
            If any parameters fail validation.
        """
        pass  # pragma: no cover

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
        pmf : ArrayLike
            The PMF array to validate
        param_name : str
            Name of the parameter (for error messages)
        atol : float, default 1e-6
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
        latent_incidence : ArrayLike
            Latent incidence time series (infections, prevalence, etc.).
            Shape: (n_days,)
        pmf : ArrayLike
            Delay or shedding PMF. Shape: (n_pmf,)
        p_observed : float, default 1.0
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
        suffix : str
            Suffix for the site name. Will be prefixed with ``self.name``.
            For example, suffix="predicted" becomes "{name}_predicted".
        value : ArrayLike
            Value to track. Can be any shape.
        """
        numpyro.deterministic(self._sample_site_name(suffix), value)

    def _sample_site_name(self, suffix: str) -> str:
        """
        Generate a prefixed sample site name.

        Parameters
        ----------
        suffix : str
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
        infections : ArrayLike
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
        sample : Uses this method then applies noise model
        """
        pass  # pragma: no cover

    @abstractmethod
    def validate_data(
        self,
        n_total: int,
        n_subpops: int,
        **obs_data,
    ) -> None:
        """
        Validate observation data before running inference.

        Each observation process validates its own data requirements.
        Called by the model's ``validate_data()`` method with concrete
        (non-traced) values before JAX tracing begins.

        Parameters
        ----------
        n_total : int
            Total number of time steps (n_init + n_days_post_init).
        n_subpops : int
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

    def _validate_times(self, times: ArrayLike, n_total: int) -> None:
        """
        Validate a times index array.

        Checks that all values are non-negative and within ``[0, n_total)``.

        Parameters
        ----------
        times : ArrayLike
            Time indices on the shared time axis.
        n_total : int
            Total number of time steps.

        Raises
        ------
        ValueError
            If times contains negative values or values >= n_total.
        """
        times = jnp.asarray(times)
        if jnp.any(times < 0):
            raise ValueError(f"Observation '{self.name}': times cannot be negative")
        max_time = jnp.max(times)
        if max_time >= n_total:
            raise ValueError(
                f"Observation '{self.name}': times index {int(max_time)} "
                f">= n_total ({n_total}). "
                f"Times must be on shared axis [0, {n_total})."
            )

    def _validate_subpop_indices(
        self, subpop_indices: ArrayLike, n_subpops: int
    ) -> None:
        """
        Validate a subpopulation index array.

        Checks that all values are non-negative and within ``[0, n_subpops)``.

        Parameters
        ----------
        subpop_indices : ArrayLike
            Subpopulation indices (0-indexed).
        n_subpops : int
            Number of subpopulations.

        Raises
        ------
        ValueError
            If subpop_indices contains negative values or values >= n_subpops.
        """
        subpop_indices = jnp.asarray(subpop_indices)
        if jnp.any(subpop_indices < 0):
            raise ValueError(
                f"Observation '{self.name}': subpop_indices cannot be negative"
            )
        max_idx = jnp.max(subpop_indices)
        if max_idx >= n_subpops:
            raise ValueError(
                f"Observation '{self.name}': subpop_indices contains "
                f"{int(max_idx)} >= {n_subpops} (n_subpops)"
            )

    def _validate_obs_times_length(self, obs: ArrayLike, times: ArrayLike) -> None:
        """
        Validate that obs and times arrays have matching lengths.

        Parameters
        ----------
        obs : ArrayLike
            Observed data array.
        times : ArrayLike
            Times index array.

        Raises
        ------
        ValueError
            If obs and times have different lengths.
        """
        if len(obs) != len(times):
            raise ValueError(
                f"Observation '{self.name}': obs length {len(obs)} "
                f"must match times length {len(times)}"
            )

    @abstractmethod
    def sample(
        self,
        obs: ArrayLike | None = None,
        **kwargs,
    ) -> ArrayLike:
        """
        Sample from the observation process.

        Subclasses must implement this method to define the specific
        observation model. Typically calls ``_predicted_obs`` first,
        then applies the noise model.

        Parameters
        ----------
        obs : ArrayLike | None
            Observed data for conditioning, or None for prior predictive sampling.
        **kwargs
            Subclass-specific parameters (e.g., infections from the infection process).

        Returns
        -------
        ArrayLike
            Observed or sampled values from the observation process.
        """
        pass  # pragma: no cover
