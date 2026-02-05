"""
Unit tests for Measurements (continuous measurement observations).

These tests validate the measurement observation process base class implementation.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

from pyrenew.deterministic import DeterministicPMF
from pyrenew.observation import (
    HierarchicalNormalNoise,
    Measurements,
    VectorizedRV,
)
from pyrenew.observation.base import BaseObservationProcess
from pyrenew.randomvariable import DistributionalVariable


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

    def lookback_days(self) -> int:
        """
        Return required lookback days for this observation.

        Temporal PMFs are 0-indexed (effect can occur on day 0), so a PMF
        of length L covers lags 0 to L-1, requiring L-1 initialization points.

        Returns
        -------
        int
            Length of temporal PMF minus 1.
        """
        return len(self.temporal_pmf_rv()) - 1

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
        sensor_mode_rv = VectorizedRV(
            DistributionalVariable("mode", dist.Normal(0, 0.5)),
            plate_name="sensor_mode",
        )
        sensor_sd_rv = VectorizedRV(
            DistributionalVariable("sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)),
            plate_name="sensor_sd",
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)

        process = ConcreteMeasurements(
            name="test",
            temporal_pmf_rv=DeterministicPMF("shedding", shedding_pmf),
            noise=noise,
        )

        assert process.infection_resolution() == "subpop"


class TestVectorizedRV:
    """Test VectorizedRV wrapper class."""

    def test_init_and_sample(self):
        """Test VectorizedRV initialization and sampling."""
        rv = DistributionalVariable("test", dist.Normal(0, 1.0))
        vectorized = VectorizedRV(rv, plate_name="test_plate")

        with numpyro.handlers.seed(rng_seed=42):
            samples = vectorized.sample(n_groups=5)

        assert samples.shape == (5,)


class TestHierarchicalNormalNoise:
    """Test HierarchicalNormalNoise model."""

    def test_repr(self):
        """Test HierarchicalNormalNoise __repr__ method."""
        sensor_mode_rv = VectorizedRV(
            DistributionalVariable("mode", dist.Normal(0, 0.5)),
            plate_name="sensor_mode",
        )
        sensor_sd_rv = VectorizedRV(
            DistributionalVariable("sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)),
            plate_name="sensor_sd",
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)
        repr_str = repr(noise)
        assert "HierarchicalNormalNoise" in repr_str
        assert "sensor_mode_rv" in repr_str
        assert "sensor_sd_rv" in repr_str

    def test_validate(self):
        """Test HierarchicalNormalNoise validate method."""
        sensor_mode_rv = VectorizedRV(
            DistributionalVariable("mode", dist.Normal(0, 0.5)),
            plate_name="sensor_mode",
        )
        sensor_sd_rv = VectorizedRV(
            DistributionalVariable("sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)),
            plate_name="sensor_sd",
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)
        # Should not raise - validation is deferred to sample time
        noise.validate()

    def test_sample_shape(self):
        """Test that HierarchicalNormalNoise produces correct shape."""
        sensor_mode_rv = VectorizedRV(
            DistributionalVariable("mode", dist.Normal(0, 0.5)),
            plate_name="sensor_mode",
        )
        sensor_sd_rv = VectorizedRV(
            DistributionalVariable("sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)),
            plate_name="sensor_sd",
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
        sensor_mode_rv = VectorizedRV(
            DistributionalVariable("mode", dist.Normal(0, 0.5)),
            plate_name="sensor_mode",
        )
        sensor_sd_rv = VectorizedRV(
            DistributionalVariable("sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)),
            plate_name="sensor_sd",
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

    def test_lookback_days(self):
        """Test lookback_days returns len(pmf) - 1."""
        # PMF of length 3 should return 2 (covers lags 0, 1, 2)
        shedding_pmf = jnp.array([0.3, 0.4, 0.3])
        sensor_mode_rv = VectorizedRV(
            DistributionalVariable("mode", dist.Normal(0, 0.5)),
            plate_name="sensor_mode",
        )
        sensor_sd_rv = VectorizedRV(
            DistributionalVariable("sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)),
            plate_name="sensor_sd",
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)

        process = ConcreteMeasurements(
            name="test",
            temporal_pmf_rv=DeterministicPMF("shedding", shedding_pmf),
            noise=noise,
        )

        assert process.lookback_days() == 2  # len(3) - 1 = 2

    def test_repr(self):
        """Test Measurements __repr__ method."""
        shedding_pmf = jnp.array([0.3, 0.4, 0.3])
        sensor_mode_rv = VectorizedRV(
            DistributionalVariable("mode", dist.Normal(0, 0.5)),
            plate_name="sensor_mode",
        )
        sensor_sd_rv = VectorizedRV(
            DistributionalVariable("sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)),
            plate_name="sensor_sd",
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)

        process = ConcreteMeasurements(
            name="test",
            temporal_pmf_rv=DeterministicPMF("shedding", shedding_pmf),
            noise=noise,
        )

        repr_str = repr(process)
        assert "ConcreteMeasurements" in repr_str
        assert "temporal_pmf_rv" in repr_str
        assert "noise" in repr_str

    def test_sample_shape(self):
        """Test that sample returns correct shape."""
        shedding_pmf = jnp.array([0.3, 0.4, 0.3])
        sensor_mode_rv = VectorizedRV(
            DistributionalVariable("mode", dist.Normal(0, 0.5)),
            plate_name="sensor_mode",
        )
        sensor_sd_rv = VectorizedRV(
            DistributionalVariable("sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)),
            plate_name="sensor_sd",
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)

        process = ConcreteMeasurements(
            name="test",
            temporal_pmf_rv=DeterministicPMF("shedding", shedding_pmf),
            noise=noise,
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

    def test_predicted_obs_stored(self):
        """Test that predicted values are stored as deterministic."""
        shedding_pmf = jnp.array([0.5, 0.5])
        sensor_mode_rv = VectorizedRV(
            DistributionalVariable("mode", dist.Normal(0, 0.01)),
            plate_name="sensor_mode",
        )
        sensor_sd_rv = VectorizedRV(
            DistributionalVariable("sd", dist.TruncatedNormal(0.01, 0.005, low=0.001)),
            plate_name="sensor_sd",
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)

        process = ConcreteMeasurements(
            name="test",
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
                    times=times,
                    subpop_indices=subpop_indices,
                    sensor_indices=sensor_indices,
                    n_sensors=2,
                    obs=None,
                )
            ).get_trace()

        assert "test_obs" in trace
        assert "test_predicted" in trace

    def test_non_contiguous_subpop_indices(self):
        """Test that non-contiguous subpop_indices work correctly.

        This verifies that observation processes can observe any subset
        of subpopulations, not just contiguous indices starting from 0.
        For example, with K=5 subpopulations, observations might only
        cover indices {0, 2, 4} while indices {1, 3} are unobserved.
        """
        shedding_pmf = jnp.array([0.5, 0.5])
        sensor_mode_rv = VectorizedRV(
            DistributionalVariable("mode", dist.Normal(0, 0.01)),
            plate_name="sensor_mode",
        )
        sensor_sd_rv = VectorizedRV(
            DistributionalVariable("sd", dist.TruncatedNormal(0.01, 0.005, low=0.001)),
            plate_name="sensor_sd",
        )
        noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)

        process = ConcreteMeasurements(
            name="test",
            temporal_pmf_rv=DeterministicPMF("shedding", shedding_pmf),
            noise=noise,
        )

        # 5 subpopulations with distinct infection levels
        # Subpop 0: 100, Subpop 1: 200, Subpop 2: 300, Subpop 3: 400, Subpop 4: 500
        n_days = 20
        infections = jnp.zeros((n_days, 5))
        for k in range(5):
            infections = infections.at[:, k].set((k + 1) * 100.0)

        # Observe only subpops 0, 2, 4 (non-contiguous, skipping 1 and 3)
        # Each observation from a different sensor
        times = jnp.array([10, 10, 10])
        subpop_indices = jnp.array([0, 2, 4])  # Non-contiguous!
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

        # Verify correct shape
        assert result.observed.shape == (3,)

        # Verify the predicted values correspond to the correct subpopulations
        # predicted[t, k] should reflect infections from subpop k
        # At time 10, predicted values should be proportional to infection levels
        # Note: ConcreteMeasurements returns LOG-scale values, so linear ratios
        # become differences in log space
        predicted_at_obs = result.predicted[10, subpop_indices]

        # Subpop 0 has 100 infections, subpop 2 has 300, subpop 4 has 500
        # In log space: log(300) - log(100) = log(3), log(500) - log(100) = log(5)
        assert predicted_at_obs[0] < predicted_at_obs[1] < predicted_at_obs[2]

        # Verify the differences match log of the infection ratios
        # diff[1] - diff[0] should equal log(3) ≈ 1.099
        # diff[2] - diff[0] should equal log(5) ≈ 1.609
        assert jnp.isclose(
            predicted_at_obs[1] - predicted_at_obs[0], jnp.log(3.0), atol=0.01
        )
        assert jnp.isclose(
            predicted_at_obs[2] - predicted_at_obs[0], jnp.log(5.0), atol=0.01
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
