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
    CountsBySite,
    NegativeBinomialNoise,
    PoissonNoise,
)
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
            counts = counts_process.sample(
                infections=infections,
                counts=None,
            )

        assert counts.shape[0] > 0
        assert counts.ndim == 1

    def test_delay_convolution(self, counts_factory, short_delay_pmf):
        """Test that delay is properly applied."""
        process = counts_factory.create(delay_pmf=short_delay_pmf)

        infections = jnp.zeros(30)
        infections = infections.at[10].set(1000)

        with numpyro.handlers.seed(rng_seed=42):
            counts = process.sample(
                infections=infections,
                counts=None,
            )

        # Timeline alignment: output length equals input length
        assert counts.shape[0] == len(infections)
        # First len(delay_pmf)-1 days are NaN (appear as -1 after NegativeBinomial sampling)
        assert jnp.all(counts[1:] >= 0)
        assert jnp.sum(counts[counts >= 0]) > 0

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
                counts = process.sample(
                    infections=infections,
                    counts=None,
                )
                results.append(jnp.mean(counts))

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
                counts = process.sample(
                    infections=infections,
                    counts=None,
                )
                samples.append(jnp.sum(counts))

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
            ascertainment_rate_rv=ascertainment,
            delay_distribution_rv=delay,
            noise=NegativeBinomialNoise(concentration),
        )

        infections = jnp.ones(20) * 100

        with numpyro.handlers.seed(rng_seed=42):
            counts = process.sample(
                infections=infections,
                counts=None,
            )

        assert counts.shape[0] > 0
        # Skip NaN padding
        valid_counts = counts[2:]
        assert jnp.all(valid_counts >= 0)

    def test_with_stochastic_concentration(self, simple_delay_pmf):
        """Test with uncertain concentration parameter."""
        delay = DeterministicPMF("delay", simple_delay_pmf)
        ascertainment = DeterministicVariable("ihr", 0.01)
        concentration = DistributionalVariable("conc", dist.HalfNormal(10.0))

        process = Counts(
            ascertainment_rate_rv=ascertainment,
            delay_distribution_rv=delay,
            noise=NegativeBinomialNoise(concentration),
        )

        infections = jnp.ones(20) * 100

        with numpyro.handlers.seed(rng_seed=42):
            counts = process.sample(
                infections=infections,
                counts=None,
            )

        assert counts.shape[0] > 0
        assert jnp.all(counts >= 0)


class TestCountsEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_infections(self, counts_process):
        """Test with zero infections."""
        infections = jnp.zeros(20)

        with numpyro.handlers.seed(rng_seed=42):
            counts = counts_process.sample(
                infections=infections,
                counts=None,
            )

        assert counts.shape[0] > 0

    def test_small_infections(self, counts_process):
        """Test with small infection values."""
        infections = jnp.ones(20) * 10

        with numpyro.handlers.seed(rng_seed=42):
            counts = counts_process.sample(
                infections=infections,
                counts=None,
            )

        assert counts.shape[0] > 0
        assert jnp.all(counts >= 0)

    def test_long_delay_distribution(self, counts_factory, long_delay_pmf):
        """Test with longer delay distribution."""
        process = counts_factory.create(delay_pmf=long_delay_pmf)

        infections = create_mock_infections(40, peak_day=20, shape="spike")

        with numpyro.handlers.seed(rng_seed=42):
            counts = process.sample(
                infections=infections,
                counts=None,
            )

        # Timeline alignment maintained
        assert counts.shape[0] == infections.shape[0]
        # Skip NaN padding: 10-day delay -> first 9 days are NaN
        valid_counts = counts[9:]
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
            counts = counts_process.sample(
                infections=infections,
                counts=counts_data,
                times=times,
            )

        assert counts.shape == times.shape
        assert jnp.allclose(counts, counts_data)

    def test_sparse_vs_dense_sampling(self, counts_process):
        """Test that sparse sampling gives different output shape than dense."""
        n_days = 30
        infections = jnp.ones(n_days) * 100

        # Dense: prior sampling (counts=None, no times)
        with numpyro.handlers.seed(rng_seed=42):
            dense_counts = counts_process.sample(
                infections=infections,
                counts=None,
            )

        # Sparse with observed data: only some days
        times = jnp.array([5, 10, 15, 20])
        sparse_obs_data = jnp.array([10, 12, 8, 15])
        with numpyro.handlers.seed(rng_seed=42):
            sparse_counts = counts_process.sample(
                infections=infections,
                counts=sparse_obs_data,
                times=times,
            )

        # Dense prior produces full length output
        assert dense_counts.shape == (n_days,)

        # Sparse observations produce output matching times shape
        assert sparse_counts.shape == times.shape
        assert jnp.allclose(sparse_counts, sparse_obs_data)

    def test_prior_sampling_ignores_times(self, counts_process):
        """Test that times parameter is ignored when counts=None (prior sampling)."""
        n_days = 30
        infections = jnp.ones(n_days) * 100
        times = jnp.array([5, 10, 15, 20])

        # When counts=None, times is ignored - output is dense
        with numpyro.handlers.seed(rng_seed=42):
            prior_with_times = counts_process.sample(
                infections=infections,
                counts=None,
                times=times,
            )

        with numpyro.handlers.seed(rng_seed=42):
            prior_without_times = counts_process.sample(
                infections=infections,
                counts=None,
            )

        # Both should produce dense output of shape (n_days,)
        assert prior_with_times.shape == (n_days,)
        assert prior_without_times.shape == (n_days,)
        # With same seed, outputs should be identical
        assert jnp.allclose(prior_with_times, prior_without_times)


class TestCountsBySite:
    """Test CountsBySite for disaggregated observations."""

    def test_sample_returns_correct_shape(self):
        """Test that CountsBySite sample returns correct shape."""
        delay_pmf = jnp.array([0.3, 0.4, 0.3])
        process = CountsBySite(
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.02),
            delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
            noise=PoissonNoise(),
        )

        infections = jnp.ones((30, 3)) * 500  # 30 days, 3 sites
        times = jnp.array([10, 15, 10, 15])
        subpop_indices = jnp.array([0, 0, 1, 1])

        with numpyro.handlers.seed(rng_seed=42):
            counts = process.sample(
                infections=infections,
                subpop_indices=subpop_indices,
                times=times,
                counts=None,
            )

        assert counts.shape == times.shape

    def test_infection_resolution(self):
        """Test that CountsBySite returns 'site' resolution."""
        delay_pmf = jnp.array([1.0])
        process = CountsBySite(
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
            noise=PoissonNoise(),
        )

        assert process.infection_resolution() == "site"


class TestPoissonNoise:
    """Test PoissonNoise model."""

    def test_poisson_counts(self, simple_delay_pmf):
        """Test Counts with Poisson noise."""
        process = Counts(
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
        )

        infections = jnp.ones(20) * 1000

        with numpyro.handlers.seed(rng_seed=42):
            counts = process.sample(
                infections=infections,
                counts=None,
            )

        assert counts.shape[0] == 20
        assert jnp.all(counts >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
