"""
Shared test helper classes and functions.

This module provides reusable classes and functions that are not pytest fixtures.
For pytest fixtures, see conftest.py.
"""

import jax
import jax.numpy as jnp

from pyrenew.observation import Measurements


class ConcreteMeasurements(Measurements):
    """Concrete implementation of Measurements for testing."""

    def __init__(self, name, temporal_pmf_rv, noise, log10_scale=9.0):
        """Initialize the concrete measurements for testing."""
        super().__init__(name=name, temporal_pmf_rv=temporal_pmf_rv, noise=noise)
        self.log10_scale = log10_scale

    def validate(self) -> None:
        """Validate parameters."""
        pmf = self.temporal_pmf_rv()
        self._validate_pmf(pmf, "temporal_pmf_rv")

    def _predicted_obs(self, infections):
        """
        Simple predicted signal: log(convolution * scale).

        Returns
        -------
        jnp.ndarray
            Log-transformed predicted signal.
        """
        pmf = self.temporal_pmf_rv()

        if infections.ndim == 1:
            infections = infections[:, jnp.newaxis]

        def convolve_col(col):  # numpydoc ignore=GL08
            return self._convolve_with_alignment(col, pmf, 1.0)[0]

        predicted = jax.vmap(convolve_col, in_axes=1, out_axes=1)(infections)

        log_predicted = jnp.log(predicted + 1e-10) + self.log10_scale * jnp.log(10)

        return log_predicted


def create_mock_infections(
    n_days: int,
    peak_day: int = 10,
    peak_value: float = 1000.0,
    shape: str = "spike",
) -> jnp.ndarray:
    """
    Create mock infection time series for testing.

    Parameters
    ----------
    n_days : int
        Number of days
    peak_day : int
        Day of peak infections
    peak_value : float
        Peak infection value
    shape : str
        Shape of the curve: "spike", "constant", or "decay"

    Returns
    -------
    jnp.ndarray
        Array of infections of shape (n_days,)
    """
    if shape == "spike":
        infections = jnp.zeros(n_days)
        infections = infections.at[peak_day].set(peak_value)
    elif shape == "constant":
        infections = jnp.ones(n_days) * peak_value
    elif shape == "decay":
        infections = peak_value * jnp.exp(-jnp.arange(n_days) / 20.0)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    return infections
