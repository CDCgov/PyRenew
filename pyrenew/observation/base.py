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
    - ``get_required_lookback()``: Return PMF length for initialization
    - ``infection_resolution()``: Return ``"jurisdiction"`` or ``"site"``
    - ``_expected_signal()``: Transform infections to expected values
    - ``sample()``: Apply noise model to expected signal

    Notes
    -----
    Computing expected observations on day t requires infection history
    from previous days (determined by the temporal PMF length).
    The first ``len(pmf) - 1`` days have insufficient history and return NaN.

    See Also
    --------
    pyrenew.convolve.compute_delay_ascertained_incidence :
        Underlying convolution function
    pyrenew.metaclass.RandomVariable :
        Base class for all random variables
    """

    def __init__(self, temporal_pmf_rv: RandomVariable) -> None:
        """
        Initialize base observation process.

        Parameters
        ----------
        temporal_pmf_rv : RandomVariable
            The temporal distribution PMF (e.g., delay or shedding distribution).
            Must sample to a 1D array that sums to ~1.0 with non-negative values.
            Subclasses may have additional parameters.

        Notes
        -----
        Subclasses should call ``super().__init__(temporal_pmf_rv)``
        in their constructors and may add additional parameters.
        """
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
    def get_required_lookback(self) -> int:
        """
        Return the number of days this observation process needs to look back.

        This determines the minimum n_initialization_points required by the
        latent process when this observation is included in a multi-signal model.

        Returns
        -------
        int
            Number of days of infection history required.
            Typically the length of the delay or shedding PMF.

        Notes
        -----
        This is used by model builders to automatically compute
        n_initialization_points as:
        ``max(gen_int_length, max(all lookbacks)) - 1``
        """
        pass  # pragma: no cover

    @abstractmethod
    def infection_resolution(self) -> str:
        """
        Return the resolution of infections this observation uses.

        Returns one of:

        - ``"jurisdiction"``: Uses jurisdiction-level aggregated infections
        - ``"site"``: Uses site-level disaggregated infections

        Returns
        -------
        str
            Either ``"jurisdiction"`` or ``"site"``

        Examples
        --------
        >>> # Aggregated count observations use jurisdiction-level
        >>> hosp_obs.infection_resolution()  # Returns "jurisdiction"
        >>>
        >>> # Wastewater uses site-level
        >>> ww_obs.infection_resolution()  # Returns "site"

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

    def get_minimum_observation_day(self) -> int:
        """
        Get the first day with valid (non-NaN) convolution results.

        Due to the convolution operation requiring a history window,
        the first ``len(pmf) - 1`` days will have NaN values in the
        output. This method returns the index of the first valid day.

        Returns
        -------
        int
            Day index (0-based) of first valid observation.
            Equal to ``len(pmf) - 1``.
        """
        pmf = self.temporal_pmf_rv()
        return int(len(pmf) - 1)

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

    def _deterministic(self, name: str, value: ArrayLike) -> None:
        """
        Track a deterministic quantity in the numpyro execution trace.

        This is a convenience wrapper around ``numpyro.deterministic`` for
        tracking intermediate quantities (e.g., latent admissions, expected
        concentrations) that are useful for diagnostics and model checking.
        These quantities are stored in MCMC samples and can be used for
        model diagnostics and posterior predictive checks.

        Parameters
        ----------
        name : str
            Name for the tracked quantity. Will appear in MCMC samples.
        value : ArrayLike
            Value to track. Can be any shape.
        """
        numpyro.deterministic(name, value)

    @abstractmethod
    def _expected_signal(
        self,
        infections: ArrayLike,
    ) -> ArrayLike:
        """
        Transform infections to expected observation values.

        This is the core transformation that each observation process must
        implement. It converts infections (from the infection process)
        to expected values for the observation model.

        Parameters
        ----------
        infections : ArrayLike
            Infections from the infection process.
            Shape: (n_days,) for jurisdiction-level observations
            Shape: (n_days, n_sites) for site-level observations

        Returns
        -------
        ArrayLike
            Expected observation values (counts, log-concentrations, etc.).
            Same shape as input, with first len(pmf)-1 days as NaN.

        Notes
        -----
        The transformation is observation-specific:

        - Count observations: ascertainment x delay convolution -> expected counts
        - Wastewater: shedding convolution -> genome scaling -> dilution -> log

        See Also
        --------
        sample : Uses this method then applies noise model
        """
        pass  # pragma: no cover

    @abstractmethod
    def sample(self, **kwargs) -> ArrayLike:
        """
        Sample from the observation process.

        Subclasses must implement this method to define the specific
        observation model. Typically calls ``_expected_signal`` first,
        then applies the noise model.

        Parameters
        ----------
        **kwargs
            Subclass-specific parameters. At minimum, should include:

            - infections from the infection process
            - Observed data (or None for prior predictive sampling)

        Returns
        -------
        ArrayLike
            Observed or sampled values from the observation process.
        """
        pass  # pragma: no cover
