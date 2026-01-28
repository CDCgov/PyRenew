"""
Unit tests for Counts (aggregated count observations).

These tests validate the count observation process implementation.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.observation import (
    Counts,
    CountsBySubpop,
    NegativeBinomialNoise,
    PoissonNoise,
)
from pyrenew.observation.count_observations import _CountBase
from pyrenew.randomvariable import DistributionalVariable


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


class TestCountsBasics:
    """Test basic functionality of aggregated count observation process."""

    def test_sample_returns_correct_shape(self, counts_process):
        """Test that sample returns correct shape."""
        infections = jnp.ones(30) * 100

        with numpyro.handlers.seed(rng_seed=42):
            result = counts_process.sample(
                infections=infections,
                obs=None,
            )

        assert result.observed.shape[0] > 0
        assert result.observed.ndim == 1
        assert result.predicted.shape == infections.shape

    def test_delay_convolution(self, counts_factory, short_delay_pmf):
        """Test that delay is properly applied."""
        process = counts_factory.create(delay_pmf=short_delay_pmf)

        infections = jnp.zeros(30)
        infections = infections.at[10].set(1000)

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                obs=None,
            )

        # Timeline alignment: output length equals input length
        assert result.observed.shape[0] == len(infections)
        # First len(delay_pmf)-1 days are NaN (appear as -1 after NegativeBinomial sampling)
        assert jnp.all(result.observed[1:] >= 0)
        assert jnp.sum(result.observed[result.observed >= 0]) > 0

    def test_ascertainment_scaling(self, counts_factory, simple_delay_pmf):
        """Test that ascertainment rate properly scales counts."""
        infections = jnp.ones(20) * 100

        results = []
        for rate_value in [0.01, 0.02, 0.05]:
            process = counts_factory.create(
                delay_pmf=simple_delay_pmf,
                ascertainment_rate=rate_value,
            )

            with numpyro.handlers.seed(rng_seed=42):
                result = process.sample(
                    infections=infections,
                    obs=None,
                )
                results.append(jnp.mean(result.observed))

        # Higher ascertainment rate should lead to more counts
        assert results[1] > results[0]
        assert results[2] > results[1]

    def test_negative_binomial_observation(self, counts_factory, simple_delay_pmf):
        """Test that negative binomial observation is used."""
        process = counts_factory.create(
            delay_pmf=simple_delay_pmf,
            concentration=5.0,
        )

        infections = jnp.ones(20) * 100

        samples = []
        for seed in range(5):
            with numpyro.handlers.seed(rng_seed=seed):
                result = process.sample(
                    infections=infections,
                    obs=None,
                )
                samples.append(jnp.sum(result.observed))

        # Should have some variability due to negative binomial sampling
        assert jnp.std(jnp.array(samples)) > 0


class TestCountsWithPriors:
    """Test aggregated count observation with uncertain parameters."""

    def test_with_stochastic_ascertainment(self, short_shedding_pmf):
        """Test with uncertain ascertainment rate parameter."""
        delay = DeterministicPMF("delay", jnp.array([0.2, 0.5, 0.3]))
        ascertainment = DistributionalVariable("ihr", dist.Beta(2, 100))
        concentration = DeterministicVariable("conc", 10.0)

        process = Counts(
            name="test",
            ascertainment_rate_rv=ascertainment,
            delay_distribution_rv=delay,
            noise=NegativeBinomialNoise(concentration),
        )

        infections = jnp.ones(20) * 100

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                obs=None,
            )

        assert result.observed.shape[0] > 0
        # Skip NaN padding
        valid_counts = result.observed[2:]
        assert jnp.all(valid_counts >= 0)

    def test_with_stochastic_concentration(self, simple_delay_pmf):
        """Test with uncertain concentration parameter."""
        delay = DeterministicPMF("delay", simple_delay_pmf)
        ascertainment = DeterministicVariable("ihr", 0.01)
        concentration = DistributionalVariable("conc", dist.HalfNormal(10.0))

        process = Counts(
            name="test",
            ascertainment_rate_rv=ascertainment,
            delay_distribution_rv=delay,
            noise=NegativeBinomialNoise(concentration),
        )

        infections = jnp.ones(20) * 100

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                obs=None,
            )

        assert result.observed.shape[0] > 0
        assert jnp.all(result.observed >= 0)


class TestCountsEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_infections(self, counts_process):
        """Test with zero infections."""
        infections = jnp.zeros(20)

        with numpyro.handlers.seed(rng_seed=42):
            result = counts_process.sample(
                infections=infections,
                obs=None,
            )

        assert result.observed.shape[0] > 0

    def test_small_infections(self, counts_process):
        """Test with small infection values."""
        infections = jnp.ones(20) * 10

        with numpyro.handlers.seed(rng_seed=42):
            result = counts_process.sample(
                infections=infections,
                obs=None,
            )

        assert result.observed.shape[0] > 0
        assert jnp.all(result.observed >= 0)

    def test_long_delay_distribution(self, counts_factory, long_delay_pmf):
        """Test with longer delay distribution."""
        process = counts_factory.create(delay_pmf=long_delay_pmf)

        infections = create_mock_infections(40, peak_day=20, shape="spike")

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                obs=None,
            )

        # Timeline alignment maintained
        assert result.observed.shape[0] == infections.shape[0]
        # Skip NaN padding: 10-day delay -> first 9 days are NaN
        valid_counts = result.observed[9:]
        assert jnp.sum(valid_counts) > 0


class TestCountsSparseObservations:
    """Test sparse observation support."""

    def test_sparse_observations(self, counts_process):
        """Test with sparse (irregular) observations."""
        n_days = 30
        infections = jnp.ones(n_days) * 100

        # Sparse observations: only days 5, 10, 15, 20
        times = jnp.array([5, 10, 15, 20])
        counts_data = jnp.array([10, 12, 8, 15])

        with numpyro.handlers.seed(rng_seed=42):
            result = counts_process.sample(
                infections=infections,
                obs=counts_data,
                times=times,
            )

        assert result.observed.shape == times.shape
        assert jnp.allclose(result.observed, counts_data)

    def test_sparse_vs_dense_sampling(self, counts_process):
        """Test that sparse sampling gives different output shape than dense."""
        n_days = 30
        infections = jnp.ones(n_days) * 100

        # Dense: prior sampling (obs=None, no times)
        with numpyro.handlers.seed(rng_seed=42):
            dense_result = counts_process.sample(
                infections=infections,
                obs=None,
            )

        # Sparse with observed data: only some days
        times = jnp.array([5, 10, 15, 20])
        sparse_obs_data = jnp.array([10, 12, 8, 15])
        with numpyro.handlers.seed(rng_seed=42):
            sparse_result = counts_process.sample(
                infections=infections,
                obs=sparse_obs_data,
                times=times,
            )

        # Dense prior produces full length output
        assert dense_result.observed.shape == (n_days,)

        # Sparse observations produce output matching times shape
        assert sparse_result.observed.shape == times.shape
        assert jnp.allclose(sparse_result.observed, sparse_obs_data)

    def test_prior_sampling_ignores_times(self, counts_process):
        """Test that times parameter is ignored when obs=None (prior sampling)."""
        n_days = 30
        infections = jnp.ones(n_days) * 100
        times = jnp.array([5, 10, 15, 20])

        # When obs=None, times is ignored - output is dense
        with numpyro.handlers.seed(rng_seed=42):
            result_with_times = counts_process.sample(
                infections=infections,
                obs=None,
                times=times,
            )

        with numpyro.handlers.seed(rng_seed=42):
            result_without_times = counts_process.sample(
                infections=infections,
                obs=None,
            )

        # Both should produce dense output of shape (n_days,)
        assert result_with_times.observed.shape == (n_days,)
        assert result_without_times.observed.shape == (n_days,)
        # With same seed, outputs should be identical
        assert jnp.allclose(result_with_times.observed, result_without_times.observed)


class TestCountsBySubpop:
    """Test CountsBySubpop for subpopulation-level observations."""

    def test_sample_returns_correct_shape(self):
        """Test that CountsBySubpop sample returns correct shape."""
        delay_pmf = jnp.array([0.3, 0.4, 0.3])
        process = CountsBySubpop(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.02),
            delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
            noise=PoissonNoise(),
        )

        infections = jnp.ones((30, 3)) * 500  # 30 days, 3 subpops
        times = jnp.array([10, 15, 10, 15])
        subpop_indices = jnp.array([0, 0, 1, 1])

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                subpop_indices=subpop_indices,
                times=times,
                obs=None,
            )

        assert result.observed.shape == times.shape
        assert result.predicted.shape == infections.shape

    def test_infection_resolution(self):
        """Test that CountsBySubpop returns 'subpop' resolution."""
        delay_pmf = jnp.array([1.0])
        process = CountsBySubpop(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
            noise=PoissonNoise(),
        )

        assert process.infection_resolution() == "subpop"


class TestPoissonNoise:
    """Test PoissonNoise model."""

    def test_poisson_counts(self, simple_delay_pmf):
        """Test Counts with Poisson noise."""
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
        )

        infections = jnp.ones(20) * 1000

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                obs=None,
            )

        assert result.observed.shape[0] == 20
        assert jnp.all(result.observed >= 0)


class TestCountBaseInternalMethods:
    """Test internal _CountBase methods for coverage."""

    def test_count_base_infection_resolution_raises(self, simple_delay_pmf):
        """Test that _CountBase.infection_resolution() raises NotImplementedError."""

        # Create a subclass that doesn't override infection_resolution
        class IncompleteCountProcess(_CountBase):
            """Incomplete count process for testing."""

            def sample(self, **kwargs):
                """Sample method stub."""
                pass

        process = IncompleteCountProcess(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
        )
        with pytest.raises(
            NotImplementedError, match="Subclasses must implement infection_resolution"
        ):
            process.infection_resolution()


class TestValidationMethods:
    """Test validation methods for coverage."""

    def test_validate_calls_all_validations(self, simple_delay_pmf):
        """Test that validate() calls all necessary validations."""
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
        )
        # Should not raise
        process.validate()

    def test_validate_invalid_ascertainment_rate_negative(self, simple_delay_pmf):
        """Test that validate raises for negative ascertainment rate."""
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", -0.1),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
        )
        with pytest.raises(ValueError, match="ascertainment_rate_rv must be in"):
            process.validate()

    def test_validate_invalid_ascertainment_rate_greater_than_one(
        self, simple_delay_pmf
    ):
        """Test that validate raises for ascertainment rate > 1."""
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.5),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
        )
        with pytest.raises(ValueError, match="ascertainment_rate_rv must be in"):
            process.validate()

    def test_lookback_days(self, simple_delay_pmf, long_delay_pmf):
        """Test lookback_days returns PMF length."""
        process_short = Counts(
            name="test_short",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
        )
        assert process_short.lookback_days() == 1

        process_long = Counts(
            name="test_long",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", long_delay_pmf),
            noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
        )
        assert process_long.lookback_days() == 10

    def test_infection_resolution_counts(self, simple_delay_pmf):
        """Test that Counts returns 'aggregate' resolution."""
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
        )
        assert process.infection_resolution() == "aggregate"


class TestNoiseRepr:
    """Test noise model __repr__ methods."""

    def test_poisson_noise_repr(self):
        """Test PoissonNoise __repr__ method."""
        noise = PoissonNoise()
        assert repr(noise) == "PoissonNoise()"

    def test_negative_binomial_noise_repr(self):
        """Test NegativeBinomialNoise __repr__ method."""
        conc_rv = DeterministicVariable("conc", 10.0)
        noise = NegativeBinomialNoise(conc_rv)
        repr_str = repr(noise)
        assert "NegativeBinomialNoise" in repr_str
        assert "concentration_rv" in repr_str


class TestNoiseValidation:
    """Test noise model validation methods."""

    def test_poisson_noise_validate(self):
        """Test PoissonNoise validate method."""
        noise = PoissonNoise()
        # Should not raise - Poisson has no parameters to validate
        noise.validate()

    def test_negative_binomial_noise_validate_success(self):
        """Test NegativeBinomialNoise validate with valid concentration."""
        noise = NegativeBinomialNoise(DeterministicVariable("conc", 10.0))
        # Should not raise
        noise.validate()

    def test_negative_binomial_noise_validate_zero_concentration(self):
        """Test NegativeBinomialNoise validate with zero concentration."""
        noise = NegativeBinomialNoise(DeterministicVariable("conc", 0.0))
        with pytest.raises(ValueError, match="concentration must be positive"):
            noise.validate()

    def test_negative_binomial_noise_validate_negative_concentration(self):
        """Test NegativeBinomialNoise validate with negative concentration."""
        noise = NegativeBinomialNoise(DeterministicVariable("conc", -1.0))
        with pytest.raises(ValueError, match="concentration must be positive"):
            noise.validate()


class TestBaseObservationProcessValidation:
    """Test base observation process PMF validation."""

    def test_validate_pmf_empty_array(self, simple_delay_pmf):
        """Test that _validate_pmf raises for empty array."""
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
        )
        empty_pmf = jnp.array([])
        with pytest.raises(ValueError, match="must return non-empty array"):
            process._validate_pmf(empty_pmf, "test_pmf")

    def test_validate_pmf_sum_not_one(self, simple_delay_pmf):
        """Test that _validate_pmf raises for PMF not summing to 1."""
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
        )
        bad_pmf = jnp.array([0.3, 0.3, 0.3])  # sums to 0.9
        with pytest.raises(ValueError, match="must sum to 1.0"):
            process._validate_pmf(bad_pmf, "test_pmf")

    def test_validate_pmf_negative_values(self, simple_delay_pmf):
        """Test that _validate_pmf raises for negative values."""
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
        )
        bad_pmf = jnp.array([1.5, -0.5])  # sums to 1.0 but has negative
        with pytest.raises(ValueError, match="must have non-negative values"):
            process._validate_pmf(bad_pmf, "test_pmf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
