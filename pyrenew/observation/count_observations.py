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


class _CountBase(BaseObservationProcess):
    """
    Internal base for count observation processes.

    Implements ascertainment x delay convolution with pluggable noise model.
    """

    def __init__(
        self,
        ascertainment_rate_rv: RandomVariable,
        delay_distribution_rv: RandomVariable,
        noise: CountNoise,
    ) -> None:
        """
        Initialize count observation base.

        Parameters
        ----------
        ascertainment_rate_rv : RandomVariable
            Ascertainment rate in [0, 1] (e.g., IHR, IER).
        delay_distribution_rv : RandomVariable
            Delay distribution PMF (must sum to ~1.0).
        noise : CountNoise
            Noise model for count observations (Poisson, NegBin, etc.).
        """
        super().__init__(temporal_pmf_rv=delay_distribution_rv)
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

    def get_required_lookback(self) -> int:
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
            "jurisdiction" for aggregated, "site" for disaggregated.
        """
        raise NotImplementedError("Subclasses must implement infection_resolution()")

    def _expected_signal(
        self,
        infections: ArrayLike,
    ) -> ArrayLike:
        """
        Compute expected counts via ascertainment x delay convolution.

        Parameters
        ----------
        infections : ArrayLike
            Infections from the infection process.
            Shape: (n_days,) for jurisdiction-level
            Shape: (n_days, n_sites) for site-level

        Returns
        -------
        ArrayLike
            Expected counts with timeline alignment.
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

        expected_counts = jax.vmap(convolve_col, in_axes=1, out_axes=1)(infections)

        return expected_counts[:, 0] if is_1d else expected_counts


class Counts(_CountBase):
    """
    Aggregated count observation for jurisdiction-level data.

    Maps jurisdiction-level infections to aggregated counts through
    ascertainment x delay convolution with composable noise model.

    Parameters
    ----------
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

    Examples
    --------
    >>> from pyrenew.deterministic import DeterministicVariable, DeterministicPMF
    >>> from pyrenew.observation import Counts, NegativeBinomialNoise
    >>> import jax.numpy as jnp
    >>> import numpyro
    >>>
    >>> delay_pmf = jnp.array([0.2, 0.5, 0.3])
    >>> counts_obs = Counts(
    ...     ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
    ...     delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
    ...     noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
    ... )
    >>>
    >>> with numpyro.handlers.seed(rng_seed=42):
    ...     infections = jnp.ones(30) * 1000
    ...     sampled_counts = counts_obs.sample(infections=infections, counts=None)
    """

    def infection_resolution(self) -> str:
        """
        Return "jurisdiction" for aggregated observations.

        Returns
        -------
        str
            The string "jurisdiction".
        """
        return "jurisdiction"

    def sample(
        self,
        infections: ArrayLike,
        counts: ArrayLike | None = None,
        times: ArrayLike | None = None,
    ) -> ArrayLike:
        """
        Sample aggregated counts with dense or sparse observations.

        Validation is performed before JAX tracing at runtime,
        prior to calling this method.

        Parameters
        ----------
        infections : ArrayLike
            Jurisdiction-level infections from the infection process.
            Shape: (n_days,)
        counts : ArrayLike | None
            Observed counts. Dense: (n_days,), Sparse: (n_obs,), None: prior.
        times : ArrayLike | None
            Day indices for sparse observations. None for dense observations.

        Returns
        -------
        ArrayLike
            Observed or sampled counts.
            Dense: (n_days,), Sparse: (n_obs,)
        """
        expected_counts = self._expected_signal(infections)
        self._deterministic("expected_counts", expected_counts)
        expected_counts_safe = jnp.nan_to_num(expected_counts, nan=0.0)

        # Only use sparse indexing when conditioning on observations
        if times is not None and counts is not None:
            expected_obs = expected_counts_safe[times]
        else:
            expected_obs = expected_counts_safe

        return self.noise.sample(
            name="counts",
            expected=expected_obs,
            obs=counts,
        )


class CountsBySite(_CountBase):
    """
    Disaggregated count observation for site-specific data.

    Maps site-level infections to site-specific counts through
    ascertainment x delay convolution with composable noise model.

    Parameters
    ----------
    ascertainment_rate_rv : RandomVariable
        Ascertainment rate in [0, 1].
    delay_distribution_rv : RandomVariable
        Delay distribution PMF (must sum to ~1.0).
    noise : CountNoise
        Noise model (PoissonNoise, NegativeBinomialNoise, etc.).

    Notes
    -----
    Output preserves input timeline. First len(delay_pmf)-1 days are NaN.

    Examples
    --------
    >>> from pyrenew.deterministic import DeterministicVariable, DeterministicPMF
    >>> from pyrenew.observation import CountsBySite, PoissonNoise
    >>> import jax.numpy as jnp
    >>> import numpyro
    >>>
    >>> delay_pmf = jnp.array([0.3, 0.4, 0.3])
    >>> counts_obs = CountsBySite(
    ...     ascertainment_rate_rv=DeterministicVariable("ihr", 0.02),
    ...     delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
    ...     noise=PoissonNoise(),
    ... )
    >>>
    >>> with numpyro.handlers.seed(rng_seed=42):
    ...     infections = jnp.ones((30, 3)) * 500  # 30 days, 3 sites
    ...     times = jnp.array([10, 15, 10, 15])
    ...     subpop_indices = jnp.array([0, 0, 1, 1])
    ...     sampled = counts_obs.sample(
    ...         infections=infections,
    ...         subpop_indices=subpop_indices,
    ...         times=times,
    ...         counts=None,
    ...     )
    """

    def infection_resolution(self) -> str:
        """
        Return "site" for disaggregated observations.

        Returns
        -------
        str
            The string "site".
        """
        return "site"

    def sample(
        self,
        infections: ArrayLike,
        subpop_indices: ArrayLike,
        times: ArrayLike,
        counts: ArrayLike | None = None,
    ) -> ArrayLike:
        """
        Sample disaggregated counts with flexible indexing.

        Validation is performed before JAX tracing at runtime,
        prior to calling this method.

        Parameters
        ----------
        infections : ArrayLike
            Site-level infections from the infection process.
            Shape: (n_days, n_sites)
        subpop_indices : ArrayLike
            Subpopulation index for each observation (0-indexed).
            Shape: (n_obs,)
        times : ArrayLike
            Day index for each observation (0-indexed).
            Shape: (n_obs,)
        counts : ArrayLike | None
            Observed counts (n_obs,), or None for prior sampling.

        Returns
        -------
        ArrayLike
            Observed or sampled counts.
            Shape: (n_obs,)
        """
        # Compute expected counts for all sites
        expected_counts_all = self._expected_signal(infections)

        self._deterministic("expected_counts_by_site", expected_counts_all)

        # Replace NaN padding with 0 for distribution creation
        expected_counts_safe = jnp.nan_to_num(expected_counts_all, nan=0.0)
        expected_obs = expected_counts_safe[times, subpop_indices]

        return self.noise.sample(
            name="counts_by_site",
            expected=expected_obs,
            obs=counts,
        )
