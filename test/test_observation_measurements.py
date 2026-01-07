"""
Unit tests for Measurements (continuous measurement observations).

These tests validate the measurement observation process base class implementation.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

from pyrenew.deterministic import DeterministicPMF
from pyrenew.observation import HierarchicalNormalNoise, Measurements
from pyrenew.observation.base import BaseObservationProcess
from pyrenew.randomvariable import DistributionalVariable


class ConcreteMeasurements(Measurements):
    """Concrete implementation of Measurements for testing."""

    def __init__(self, temporal_pmf_rv, noise, log10_scale=9.0):
        """Initialize the concrete measurements for testing."""
        super().__init__(temporal_pmf_rv=temporal_pmf_rv, noise=noise)
        self.log10_scale = log10_scale

    def validate(self) -> None:
        """Validate parameters."""
        pmf = self.temporal_pmf_rv()
        self._validate_pmf(pmf, "temporal_pmf_rv")

    def lookback_days(self) -> int:
        """
        Return temporal PMF length.

        Returns
        -------
        int
            Length of the temporal PMF.
        """
        return len(self.temporal_pmf_rv())

    def _predicted_obs(self, infections):
        """
        Simple predicted signal: log(convolution * scale).

        Returns
        -------
        jnp.ndarray
            Log-transformed predicted signal.
        """
        pmf = self.temporal_pmf_rv()

        # Handle 2D infections (n_days, n_subpops)
        if infections.ndim == 1:
            infections = infections[:, jnp.newaxis]

        def convolve_col(col):  # numpydoc ignore=GL08
            return self._convolve_with_alignment(col, pmf, 1.0)[0]

        import jax

        predicted = jax.vmap(convolve_col, in_axes=1, out_axes=1)(infections)

        # Apply log10 scaling (simplified from wastewater model)
        log_predicted = jnp.log(predicted + 1e-10) + self.log10_scale * jnp.log(10)

        return log_predicted


class TestMeasurementsBase:
    """Test Measurements abstract base class."""

    def test_is_base_observation_process(self):
        """Test that Measurements inherits from BaseObservationProcess."""
        assert issubclass(Measurements, BaseObservationProcess)

    def test_infection_resolution_is_subpop(self):
        """Test that Measurements returns 'subpop' resolution."""
        shedding_pmf = jnp.array([0.3, 0.4, 0.3])
        sensor_mode_rv = DistributionalVariable("mode", dist.Normal(0, 0.5))
        sensor_sd_rv = DistributionalVariable(
            "sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)

        process = ConcreteMeasurements(
            temporal_pmf_rv=DeterministicPMF("shedding", shedding_pmf),
            noise=noise,
        )

        assert process.infection_resolution() == "subpop"


class TestHierarchicalNormalNoise:
    """Test HierarchicalNormalNoise model."""

    def test_validate(self):
        """Test HierarchicalNormalNoise validate method."""
        sensor_mode_rv = DistributionalVariable("mode", dist.Normal(0, 0.5))
        sensor_sd_rv = DistributionalVariable(
            "sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)
        # Should not raise - validation is deferred to sample time
        noise.validate()

    def test_sample_shape(self):
        """Test that HierarchicalNormalNoise produces correct shape."""
        sensor_mode_rv = DistributionalVariable("mode", dist.Normal(0, 0.5))
        sensor_sd_rv = DistributionalVariable(
            "sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)

        predicted = jnp.array([1.0, 2.0, 3.0, 4.0])
        sensor_indices = jnp.array([0, 0, 1, 1])

        with numpyro.handlers.seed(rng_seed=42):
            samples = noise.sample(
                name="test",
                predicted=predicted,
                obs=None,
                sensor_indices=sensor_indices,
                n_sensors=2,
            )

        assert samples.shape == predicted.shape

    def test_sample_with_observations(self):
        """Test that HierarchicalNormalNoise conditions on observations."""
        sensor_mode_rv = DistributionalVariable("mode", dist.Normal(0, 0.5))
        sensor_sd_rv = DistributionalVariable(
            "sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)

        predicted = jnp.array([1.0, 2.0, 3.0, 4.0])
        obs = jnp.array([1.1, 2.1, 3.1, 4.1])
        sensor_indices = jnp.array([0, 0, 1, 1])

        with numpyro.handlers.seed(rng_seed=42):
            samples = noise.sample(
                name="test",
                predicted=predicted,
                obs=obs,
                sensor_indices=sensor_indices,
                n_sensors=2,
            )

        # When obs is provided, samples should equal obs
        assert jnp.allclose(samples, obs)


class TestConcreteMeasurements:
    """Test concrete Measurements implementation."""

    def test_sample_shape(self):
        """Test that sample returns correct shape."""
        shedding_pmf = jnp.array([0.3, 0.4, 0.3])
        sensor_mode_rv = DistributionalVariable("mode", dist.Normal(0, 0.5))
        sensor_sd_rv = DistributionalVariable(
            "sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)

        process = ConcreteMeasurements(
            temporal_pmf_rv=DeterministicPMF("shedding", shedding_pmf),
            noise=noise,
        )

        # 30 days, 2 subpops
        infections = jnp.ones((30, 2)) * 1000
        subpop_indices = jnp.array([0, 0, 1, 1])
        sensor_indices = jnp.array([0, 0, 1, 1])
        times = jnp.array([10, 15, 10, 15])

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                subpop_indices=subpop_indices,
                sensor_indices=sensor_indices,
                times=times,
                obs=None,
                n_sensors=2,
            )

        assert result.observed.shape == times.shape
        assert result.predicted.shape == infections.shape

    def test_predicted_obs_stored(self):
        """Test that predicted_log_conc is stored as deterministic."""
        shedding_pmf = jnp.array([0.5, 0.5])
        sensor_mode_rv = DistributionalVariable("mode", dist.Normal(0, 0.01))
        sensor_sd_rv = DistributionalVariable(
            "sd", dist.TruncatedNormal(0.01, 0.005, low=0.001)
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)

        process = ConcreteMeasurements(
            temporal_pmf_rv=DeterministicPMF("shedding", shedding_pmf),
            noise=noise,
        )

        infections = jnp.ones((20, 2)) * 1000
        subpop_indices = jnp.array([0, 1])
        sensor_indices = jnp.array([0, 1])
        times = jnp.array([10, 10])

        with numpyro.handlers.seed(rng_seed=42):
            trace = numpyro.handlers.trace(
                lambda: process.sample(
                    infections=infections,
                    subpop_indices=subpop_indices,
                    sensor_indices=sensor_indices,
                    times=times,
                    obs=None,
                    n_sensors=2,
                )
            ).get_trace()

        assert "predicted_log_conc" in trace


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
