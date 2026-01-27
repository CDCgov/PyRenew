# numpydoc ignore=GL08
"""
Count observations with composable noise models.

Ascertainment x delay convolution with pluggable noise (Poisson, Negative Binomial, etc.).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from pyrenew.metaclass import RandomVariable
from pyrenew.observation.base import BaseObservationProcess
from pyrenew.observation.noise import CountNoise
from pyrenew.observation.types import ObservationSample


class _CountBase(BaseObservationProcess):
    """
    Internal base for count observation processes.

    Implements ascertainment x delay convolution with pluggable noise model.
    """

    def __init__(
        self,
        name: str,
        ascertainment_rate_rv: RandomVariable,
        delay_distribution_rv: RandomVariable,
        noise: CountNoise,
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
        """
        super().__init__(name=name, temporal_pmf_rv=delay_distribution_rv)
        self.ascertainment_rate_rv = ascertainment_rate_rv
        self.noise = noise

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

    def lookback_days(self) -> int:
        """
        Return delay PMF length.

        Returns
        -------
        int
            Length of delay distribution PMF.
        """
        return len(self.temporal_pmf_rv())

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

        is_1d = infections.ndim == 1
        if is_1d:
            infections = infections[:, jnp.newaxis]

        def convolve_col(col):  # numpydoc ignore=GL08
            return self._convolve_with_alignment(col, delay_pmf, ascertainment_rate)[0]

        predicted_counts = jax.vmap(convolve_col, in_axes=1, out_axes=1)(infections)

        return predicted_counts[:, 0] if is_1d else predicted_counts


class Counts(_CountBase):
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

    Notes
    -----
    Output preserves input timeline. First len(delay_pmf)-1 days return
    -1 or ~0 (depending on noise model) due to NaN padding.
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
            f"noise={self.noise!r})"
        )

    def sample(
        self,
        infections: ArrayLike,
        obs: ArrayLike | None = None,
        times: ArrayLike | None = None,
    ) -> ObservationSample:
        """
        Sample aggregated counts with dense or sparse observations.

        Validation is performed before JAX tracing at runtime,
        prior to calling this method.

        Parameters
        ----------
        infections : ArrayLike
            Aggregate infections from the infection process.
            Shape: (n_days,)
        obs : ArrayLike | None
            Observed counts. Dense: (n_days,), Sparse: (n_obs,), None: prior.
        times : ArrayLike | None
            Day indices for sparse observations. None for dense observations.

        Returns
        -------
        ObservationSample
            Named tuple with `observed` (sampled/conditioned counts) and
            `predicted` (predicted counts before noise).
        """
        predicted_counts = self._predicted_obs(infections)
        self._deterministic("predicted", predicted_counts)

        # Only use sparse indexing when conditioning on observations
        if times is not None and obs is not None:
            predicted_obs = predicted_counts[times]
        else:
            predicted_obs = predicted_counts

        observed = self.noise.sample(
            name=self._sample_site_name("obs"),
            predicted=predicted_obs,
            obs=obs,
        )

        return ObservationSample(observed=observed, predicted=predicted_counts)


class CountsBySubpop(_CountBase):
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

    Notes
    -----
    Output preserves input timeline. First len(delay_pmf)-1 days are NaN.
    """

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CountsBySubpop(name={self.name!r}, "
            f"ascertainment_rate_rv={self.ascertainment_rate_rv!r}, "
            f"delay_distribution_rv={self.temporal_pmf_rv!r}, "
            f"noise={self.noise!r})"
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

    def sample(
        self,
        infections: ArrayLike,
        subpop_indices: ArrayLike,
        times: ArrayLike,
        obs: ArrayLike | None = None,
    ) -> ObservationSample:
        """
        Sample subpopulation-level counts with flexible indexing.

        Validation is performed before JAX tracing at runtime,
        prior to calling this method.

        Parameters
        ----------
        infections : ArrayLike
            Subpopulation-level infections from the infection process.
            Shape: (n_days, n_subpops)
        subpop_indices : ArrayLike
            Subpopulation index for each observation (0-indexed).
            Shape: (n_obs,)
        times : ArrayLike
            Day index for each observation (0-indexed).
            Shape: (n_obs,)
        obs : ArrayLike | None
            Observed counts (n_obs,), or None for prior sampling.

        Returns
        -------
        ObservationSample
            Named tuple with `observed` (sampled/conditioned counts) and
            `predicted` (predicted counts before noise, shape: n_days x n_subpops).
        """
        # Compute predicted counts for all subpops
        predicted_counts = self._predicted_obs(infections)

        self._deterministic("predicted", predicted_counts)
        predicted_obs = predicted_counts[times, subpop_indices]

        observed = self.noise.sample(
            name=self._sample_site_name("obs"),
            predicted=predicted_obs,
            obs=obs,
        )

        return ObservationSample(observed=observed, predicted=predicted_counts)
