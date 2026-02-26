"""
Unit tests for Measurements (continuous measurement observations).

These tests validate the measurement observation process implementation
using ConcreteMeasurements from conftest.py.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

from pyrenew.deterministic import DeterministicPMF
from pyrenew.observation import (
    HierarchicalNormalNoise,
    VectorizedRV,
)
from pyrenew.randomvariable import DistributionalVariable
from test.test_helpers import ConcreteMeasurements


class TestVectorizedRV:
    """Test VectorizedRV wrapper class."""

    def test_init_and_sample(self):
        """Test VectorizedRV initialization and sampling."""
        rv = DistributionalVariable("test", dist.Normal(0, 1.0))
        vectorized = VectorizedRV(name="test_vectorized", rv=rv)

        with numpyro.handlers.seed(rng_seed=42):
            samples = vectorized.sample(n_groups=5)

        assert samples.shape == (5,)
        # Verify samples are actually different (not degenerate)
        assert jnp.std(samples) > 0


class TestHierarchicalNormalNoise:
    """Test HierarchicalNormalNoise model."""

    def test_sample_shape_and_sensor_variation(self, hierarchical_normal_noise):
        """Test shape and that different sensors produce different biases."""
        predicted = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        sensor_indices = jnp.array([0, 0, 1, 1, 2, 2])

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                samples = hierarchical_normal_noise.sample(
                    name="test",
                    predicted=predicted,
                    obs=None,
                    sensor_indices=sensor_indices,
                    n_sensors=3,
                )

        assert samples.shape == predicted.shape

        # Verify sensor modes are sampled (exist in trace)
        sensor_modes = trace["ww_sensor_mode"]["value"]
        assert sensor_modes.shape == (3,)

    def test_sample_with_observations(self, hierarchical_normal_noise):
        """Test that HierarchicalNormalNoise conditions on observations."""
        predicted = jnp.array([1.0, 2.0, 3.0, 4.0])
        obs = jnp.array([1.1, 2.1, 3.1, 4.1])
        sensor_indices = jnp.array([0, 0, 1, 1])

        with numpyro.handlers.seed(rng_seed=42):
            samples = hierarchical_normal_noise.sample(
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

    def test_lookback_days(self, hierarchical_normal_noise):
        """Test lookback_days returns len(pmf) - 1."""
        shedding_pmf = jnp.array([0.3, 0.4, 0.3])

        process = ConcreteMeasurements(
            name="test",
            temporal_pmf_rv=DeterministicPMF("shedding", shedding_pmf),
            noise=hierarchical_normal_noise,
        )

        assert process.lookback_days() == 2

    def test_sample_shape_and_log_scale(self, hierarchical_normal_noise):
        """Test that sample returns correct shape and log-scale output."""
        shedding_pmf = jnp.array([0.3, 0.4, 0.3])

        process = ConcreteMeasurements(
            name="test",
            temporal_pmf_rv=DeterministicPMF("shedding", shedding_pmf),
            noise=hierarchical_normal_noise,
        )

        infections = jnp.ones((30, 2)) * 1000
        subpop_indices = jnp.array([0, 0, 1, 1])
        sensor_indices = jnp.array([0, 0, 1, 1])
        times = jnp.array([10, 15, 10, 15])

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                times=times,
                subpop_indices=subpop_indices,
                sensor_indices=sensor_indices,
                n_sensors=2,
                obs=None,
            )

        assert result.observed.shape == times.shape
        assert result.predicted.shape == infections.shape
        # Output should be in log-scale (large positive values due to log10_scale=9)
        assert jnp.all(result.predicted[2:, :] > 0)

    def test_predicted_obs_stored(self, hierarchical_normal_noise_tight):
        """Test that predicted values are stored as deterministic."""
        shedding_pmf = jnp.array([0.5, 0.5])

        process = ConcreteMeasurements(
            name="test",
            temporal_pmf_rv=DeterministicPMF("shedding", shedding_pmf),
            noise=hierarchical_normal_noise_tight,
        )

        infections = jnp.ones((20, 2)) * 1000
        subpop_indices = jnp.array([0, 1])
        sensor_indices = jnp.array([0, 1])
        times = jnp.array([10, 10])

        with numpyro.handlers.seed(rng_seed=42):
            trace = numpyro.handlers.trace(
                lambda: process.sample(
                    infections=infections,
                    times=times,
                    subpop_indices=subpop_indices,
                    sensor_indices=sensor_indices,
                    n_sensors=2,
                    obs=None,
                )
            ).get_trace()

        assert "test_obs" in trace
        assert "test_predicted" in trace

    def test_non_contiguous_subpop_indices(self, hierarchical_normal_noise_tight):
        """Test that non-contiguous subpop_indices work correctly.

        This verifies that observation processes can observe any subset
        of subpopulations, not just contiguous indices starting from 0.
        """
        shedding_pmf = jnp.array([0.5, 0.5])

        process = ConcreteMeasurements(
            name="test",
            temporal_pmf_rv=DeterministicPMF("shedding", shedding_pmf),
            noise=hierarchical_normal_noise_tight,
        )

        # 5 subpopulations with distinct infection levels
        n_days = 20
        infections = jnp.zeros((n_days, 5))
        for k in range(5):
            infections = infections.at[:, k].set((k + 1) * 100.0)

        # Observe only subpops 0, 2, 4 (non-contiguous)
        times = jnp.array([10, 10, 10])
        subpop_indices = jnp.array([0, 2, 4])
        sensor_indices = jnp.array([0, 1, 2])

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                times=times,
                subpop_indices=subpop_indices,
                sensor_indices=sensor_indices,
                n_sensors=3,
                obs=None,
            )

        assert result.observed.shape == (3,)

        # Predicted values should be proportional to infection levels
        # In log space: differences should match log of ratios
        predicted_at_obs = result.predicted[10, subpop_indices]
        assert predicted_at_obs[0] < predicted_at_obs[1] < predicted_at_obs[2]
        assert jnp.isclose(
            predicted_at_obs[1] - predicted_at_obs[0], jnp.log(3.0), atol=0.01
        )
        assert jnp.isclose(
            predicted_at_obs[2] - predicted_at_obs[0], jnp.log(5.0), atol=0.01
        )

    def test_log_scale_correctness(self, hierarchical_normal_noise_tight):
        """Test that output is log-scale of convolved infections times scale."""
        # Use simple PMF [1.0] so convolution is identity
        process = ConcreteMeasurements(
            name="test",
            temporal_pmf_rv=DeterministicPMF("shedding", jnp.array([1.0])),
            noise=hierarchical_normal_noise_tight,
            log10_scale=0.0,  # No scaling, so output = log(infections)
        )

        infections = jnp.ones((20, 1)) * 500.0
        times = jnp.array([10])
        subpop_indices = jnp.array([0])
        sensor_indices = jnp.array([0])

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                times=times,
                subpop_indices=subpop_indices,
                sensor_indices=sensor_indices,
                n_sensors=1,
                obs=None,
            )

        # With PMF=[1.0] and log10_scale=0, predicted should be log(500)
        expected = jnp.log(500.0)
        assert jnp.isclose(result.predicted[10, 0], expected, atol=0.01)

    def test_sensor_bias_differences(self):
        """Test that hierarchical noise produces sensor-specific biases."""
        shedding_pmf = jnp.array([1.0])

        # Use wide priors to ensure sensors get distinguishable biases
        sensor_mode_rv = VectorizedRV(
            name="sensor_mode_rv",
            rv=DistributionalVariable("mode", dist.Normal(0, 2.0)),
        )
        sensor_sd_rv = VectorizedRV(
            name="sensor_sd_rv",
            rv=DistributionalVariable("sd", dist.TruncatedNormal(0.1, 0.05, low=0.01)),
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)

        process = ConcreteMeasurements(
            name="test",
            temporal_pmf_rv=DeterministicPMF("shedding", shedding_pmf),
            noise=noise,
        )

        infections = jnp.ones((30, 1)) * 1000.0
        # Same subpop, same time, 3 different sensors
        times = jnp.array([15, 15, 15])
        subpop_indices = jnp.array([0, 0, 0])
        sensor_indices = jnp.array([0, 1, 2])

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                process.sample(
                    infections=infections,
                    times=times,
                    subpop_indices=subpop_indices,
                    sensor_indices=sensor_indices,
                    n_sensors=3,
                    obs=None,
                )

        # Sensor modes should be different (with wide prior)
        sensor_modes = trace["mode"]["value"]
        assert sensor_modes.shape == (3,)
        # Not all modes should be identical
        assert jnp.std(sensor_modes) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
