# numpydoc ignore=GL08
"""
Noise models for observation processes.

Provides composable noise strategies for count and measurement observations,
separating the noise distribution from the observation structure.

Count Noise
-----------
- ``PoissonNoise``: Equidispersed counts (variance = mean). No parameters.
- ``NegativeBinomialNoise``: Overdispersed counts relative to Poisson (variance > mean).
  Takes ``concentration_rv`` (higher concentration = less overdispersed, more Poisson-like).

Measurement Noise
-----------------
- ``HierarchicalNormalNoise``: Normal noise with hierarchical sensor effects.
  Takes ``sensor_mode_prior_rv`` and ``sensor_sd_prior_rv`` for sensor-level
  bias and variability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from pyrenew.metaclass import RandomVariable

_EPSILON = 1e-10


class CountNoise(ABC):
    """
    Abstract base for count observation noise models.

    Defines how discrete count observations are distributed around expected values.
    """

    @abstractmethod
    def sample(
        self,
        name: str,
        expected: ArrayLike,
        obs: ArrayLike | None = None,
    ) -> ArrayLike:
        """
        Sample count observations given expected counts.

        Parameters
        ----------
        name : str
            Numpyro sample site name.
        expected : ArrayLike
            Expected count values (non-negative).
        obs : ArrayLike | None
            Observed counts for conditioning, or None for prior sampling.

        Returns
        -------
        ArrayLike
            Sampled or conditioned counts, same shape as expected.
        """
        pass  # pragma: no cover

    @abstractmethod
    def validate(self) -> None:
        """
        Validate noise model parameters.

        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        pass  # pragma: no cover


class PoissonNoise(CountNoise):
    """
    Poisson noise for equidispersed counts (variance = mean).
    """

    def __init__(self) -> None:
        """Initialize Poisson noise (no parameters)."""
        pass

    def __repr__(self) -> str:
        """Return string representation."""
        return "PoissonNoise()"

    def validate(self) -> None:
        """Validate Poisson noise (always valid)."""
        pass

    def sample(
        self,
        name: str,
        expected: ArrayLike,
        obs: ArrayLike | None = None,
    ) -> ArrayLike:
        """
        Sample from Poisson distribution.

        Parameters
        ----------
        name : str
            Numpyro sample site name.
        expected : ArrayLike
            Expected count values.
        obs : ArrayLike | None
            Observed counts for conditioning.

        Returns
        -------
        ArrayLike
            Poisson-distributed counts.
        """
        return numpyro.sample(
            name,
            dist.Poisson(rate=expected + _EPSILON),
            obs=obs,
        )


class NegativeBinomialNoise(CountNoise):
    """
    Negative Binomial noise for overdispersed counts (variance > mean).

    Uses NB2 parameterization. Higher concentration reduces overdispersion.

    Parameters
    ----------
    concentration_rv : RandomVariable
        Concentration parameter (must be > 0).
        Higher values reduce overdispersion.

    Notes
    -----
    The NB2 parameterization has variance = mean + mean^2 / concentration.
    As concentration -> infinity, this approaches Poisson.
    """

    def __init__(self, concentration_rv: RandomVariable) -> None:
        """
        Initialize Negative Binomial noise.

        Parameters
        ----------
        concentration_rv : RandomVariable
            Concentration parameter (must be > 0).
            Higher values reduce overdispersion.
        """
        self.concentration_rv = concentration_rv

    def __repr__(self) -> str:
        """Return string representation."""
        return f"NegativeBinomialNoise(concentration_rv={self.concentration_rv!r})"

    def validate(self) -> None:
        """
        Validate concentration is positive.

        Raises
        ------
        ValueError
            If concentration <= 0.
        """
        concentration = self.concentration_rv()
        if jnp.any(concentration <= 0):
            raise ValueError(
                f"NegativeBinomialNoise: concentration must be positive, "
                f"got {float(concentration)}"
            )

    def sample(
        self,
        name: str,
        expected: ArrayLike,
        obs: ArrayLike | None = None,
    ) -> ArrayLike:
        """
        Sample from Negative Binomial distribution.

        Parameters
        ----------
        name : str
            Numpyro sample site name.
        expected : ArrayLike
            Expected count values.
        obs : ArrayLike | None
            Observed counts for conditioning.

        Returns
        -------
        ArrayLike
            Negative Binomial-distributed counts.
        """
        concentration = self.concentration_rv()
        return numpyro.sample(
            name,
            dist.NegativeBinomial2(
                mean=expected + _EPSILON,
                concentration=concentration,
            ),
            obs=obs,
        )


class MeasurementNoise(ABC):
    """
    Abstract base for continuous measurement noise models.

    Defines how continuous observations are distributed around expected values.
    """

    @abstractmethod
    def sample(
        self,
        name: str,
        expected: ArrayLike,
        obs: ArrayLike | None = None,
        **kwargs,
    ) -> ArrayLike:
        """
        Sample continuous observations given expected values.

        Parameters
        ----------
        name : str
            Numpyro sample site name.
        expected : ArrayLike
            Expected measurement values.
        obs : ArrayLike | None
            Observed measurements for conditioning, or None for prior sampling.
        **kwargs
            Additional context (e.g., sensor indices).

        Returns
        -------
        ArrayLike
            Sampled or conditioned measurements, same shape as expected.
        """
        pass  # pragma: no cover

    @abstractmethod
    def validate(self) -> None:
        """
        Validate noise model parameters.

        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        pass  # pragma: no cover


class HierarchicalNormalNoise(MeasurementNoise):
    """
    Normal noise with hierarchical sensor-level effects.

    Observation model: ``obs ~ Normal(expected + sensor_mode, sensor_sd)``
    where sensor_mode and sensor_sd are hierarchically modeled.

    Parameters
    ----------
    sensor_mode_prior_rv : RandomVariable
        Hierarchical prior for sensor-level modes (log-scale biases).
        Must support ``sample(n_groups=...)`` interface.
    sensor_sd_prior_rv : RandomVariable
        Hierarchical prior for sensor-level SDs (must be > 0).
        Must support ``sample(n_groups=...)`` interface.

    Notes
    -----
    Expects data already on log scale for wastewater applications.

    See Also
    --------
    pyrenew.randomvariable.HierarchicalNormalPrior :
        Suitable prior for sensor_mode_prior_rv
    pyrenew.randomvariable.GammaGroupSdPrior :
        Suitable prior for sensor_sd_prior_rv
    """

    def __init__(
        self,
        sensor_mode_prior_rv: RandomVariable,
        sensor_sd_prior_rv: RandomVariable,
    ) -> None:
        """
        Initialize hierarchical Normal noise.

        Parameters
        ----------
        sensor_mode_prior_rv : RandomVariable
            Hierarchical prior for sensor-level modes (log-scale biases).
            Must support ``sample(n_groups=...)`` interface.
        sensor_sd_prior_rv : RandomVariable
            Hierarchical prior for sensor-level SDs (must be > 0).
            Must support ``sample(n_groups=...)`` interface.
        """
        self.sensor_mode_prior_rv = sensor_mode_prior_rv
        self.sensor_sd_prior_rv = sensor_sd_prior_rv

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"HierarchicalNormalNoise("
            f"sensor_mode_prior_rv={self.sensor_mode_prior_rv!r}, "
            f"sensor_sd_prior_rv={self.sensor_sd_prior_rv!r})"
        )

    def validate(self) -> None:
        """
        Validate noise parameters.

        Notes
        -----
        Full validation requires n_groups, which is only available during sample().
        """
        pass

    def sample(
        self,
        name: str,
        expected: ArrayLike,
        obs: ArrayLike | None = None,
        *,
        sensor_indices: ArrayLike,
        n_sensors: int,
    ) -> ArrayLike:
        """
        Sample from Normal distribution with sensor-level hierarchical effects.

        Parameters
        ----------
        name : str
            Numpyro sample site name.
        expected : ArrayLike
            Expected log-scale measurement values.
            Shape: (n_obs,)
        obs : ArrayLike | None
            Observed log-scale measurements for conditioning.
            Shape: (n_obs,)
        sensor_indices : ArrayLike
            Sensor index for each observation (0-indexed).
            Shape: (n_obs,)
        n_sensors : int
            Total number of sensors.

        Returns
        -------
        ArrayLike
            Normal distributed measurements with hierarchical sensor effects.
            Shape: (n_obs,)

        Raises
        ------
        ValueError
            If sensor_sd samples non-positive values.
        """
        sensor_mode = self.sensor_mode_prior_rv.sample(n_groups=n_sensors)
        sensor_sd = self.sensor_sd_prior_rv.sample(n_groups=n_sensors)

        loc = expected + sensor_mode[sensor_indices]
        scale = sensor_sd[sensor_indices]

        return numpyro.sample(name, dist.Normal(loc=loc, scale=scale), obs=obs)
