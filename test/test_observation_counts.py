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
from pyrenew.randomvariable import DistributionalVariable
from test.test_helpers import create_mock_infections


class TestCountsBasics:
    """Test basic functionality of aggregated count observation process."""

    def test_sample_returns_correct_shape_with_value_checks(self, counts_process):
        """Test that sample returns correct shape with non-negative predicted counts."""
        infections = jnp.ones(30) * 100

        with numpyro.handlers.seed(rng_seed=42):
            result = counts_process.sample(
                infections=infections,
                obs=None,
            )

        assert result.observed.shape[0] > 0
        assert result.observed.ndim == 1
        assert result.predicted.shape == infections.shape
        # Predicted counts must be non-negative
        assert jnp.all(result.predicted >= 0)
        # Observed counts must be non-negative (count data)
        assert jnp.all(result.observed >= 0)

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

        assert result.predicted.shape[0] == len(infections)
        assert result.observed.shape[0] == len(infections)
        # First len(pmf) - 1 entries in predicted are NaN (initialization period)
        assert jnp.all(jnp.isnan(result.predicted[:1]))
        assert jnp.all(~jnp.isnan(result.predicted[1:]))
        assert jnp.all(result.observed >= 0)

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

        assert results[1] > results[0]
        assert results[2] > results[1]

    def test_negative_binomial_observation(self, counts_factory, simple_delay_pmf):
        """Test that negative binomial observation produces variability."""
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

        assert jnp.std(jnp.array(samples)) > 0

    def test_convolution_hand_computable(self):
        """Test convolution with hand-computable spike input.

        1000 infections on day 10, delay PMF [0.5, 0.5], ascertainment 1.0.
        Expected predicted: 500 on day 10, 500 on day 11.
        """
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", jnp.array([0.5, 0.5])),
            noise=PoissonNoise(),
        )

        infections = jnp.zeros(30)
        infections = infections.at[10].set(1000.0)

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(infections=infections, obs=None)

        # Day 10: 1000 * 0.5 * 1.0 = 500
        # Day 11: 1000 * 0.5 * 1.0 = 500
        # Day 0 is NaN (initialization from 2-element PMF)
        assert jnp.isclose(result.predicted[10], 500.0, atol=1.0)
        assert jnp.isclose(result.predicted[11], 500.0, atol=1.0)
        # Other days (post-init, not 10 or 11) should be near zero
        assert jnp.isclose(result.predicted[5], 0.0, atol=1.0)

    def test_observation_passthrough(self, counts_process):
        """Test that providing obs returns those exact values."""
        infections = jnp.ones(30) * 100
        known_obs = jnp.arange(30, dtype=jnp.float32)

        with numpyro.handlers.seed(rng_seed=42):
            result = counts_process.sample(
                infections=infections,
                obs=known_obs,
            )

        # When obs is provided, observed values should equal obs
        assert jnp.allclose(result.observed, known_obs)


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
        assert jnp.all(~jnp.isnan(result.observed))
        assert jnp.all(result.observed >= 0)

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
        """Test with zero infections produces zero predicted counts."""
        infections = jnp.zeros(20)

        with numpyro.handlers.seed(rng_seed=42):
            result = counts_process.sample(
                infections=infections,
                obs=None,
            )

        assert result.observed.shape[0] > 0
        assert jnp.all(result.observed >= 0)
        # Zero infections should produce zero predicted counts
        assert jnp.allclose(result.predicted, 0.0)

    def test_small_infections(self, counts_process):
        """Test with small infection values produces plausible counts."""
        infections = jnp.ones(20) * 10

        with numpyro.handlers.seed(rng_seed=42):
            result = counts_process.sample(
                infections=infections,
                obs=None,
            )

        assert result.observed.shape[0] > 0
        assert jnp.all(result.observed >= 0)
        # With ascertainment rate 0.01 and 10 infections,
        # predicted should be ~0.1 per day
        assert jnp.all(result.predicted <= 10)

    def test_long_delay_distribution(self, counts_factory, long_delay_pmf):
        """Test with longer delay distribution."""
        process = counts_factory.create(delay_pmf=long_delay_pmf)

        infections = create_mock_infections(40, peak_day=20, shape="spike")

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                obs=None,
            )

        assert jnp.all(~jnp.isnan(result.observed))
        assert jnp.all(result.observed >= 0)


class TestCountsDenseObservations:
    """Test dense observation support with NaN padding."""

    def test_dense_observations_with_nan_padding(self, counts_process):
        """Test with dense observations including NaN padding."""
        n_days = 30
        infections = jnp.ones(n_days) * 100

        obs = jnp.ones(n_days) * 10.0
        obs = obs.at[:5].set(jnp.nan)

        with numpyro.handlers.seed(rng_seed=42):
            result = counts_process.sample(
                infections=infections,
                obs=obs,
            )

        assert result.observed.shape[0] == n_days
        assert result.predicted.shape[0] == n_days
        # Predicted values should be non-NaN (predictions exist for all days)
        assert jnp.all(~jnp.isnan(result.predicted))


class TestCountsBySubpop:
    """Test CountsBySubpop for subpopulation-level observations."""

    def test_non_contiguous_subpop_indices(self):
        """Test that non-contiguous subpop_indices work correctly.

        Verifies observation processes can observe any subset
        of subpopulations with correct value proportionality.
        """
        delay_pmf = jnp.array([0.3, 0.4, 0.3])
        process = CountsBySubpop(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
            noise=PoissonNoise(),
        )

        # 5 subpopulations with distinct infection levels
        n_days = 20
        infections = jnp.zeros((n_days, 5))
        for k in range(5):
            infections = infections.at[:, k].set((k + 1) * 100.0)

        times = jnp.array([10, 10, 10])
        subpop_indices = jnp.array([0, 2, 4])

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                times=times,
                subpop_indices=subpop_indices,
                obs=None,
            )

        assert result.observed.shape == (3,)

        # Predicted values should be proportional to infection levels
        predicted_at_obs = result.predicted[10, subpop_indices]
        assert predicted_at_obs[0] < predicted_at_obs[1] < predicted_at_obs[2]
        assert jnp.isclose(predicted_at_obs[1] / predicted_at_obs[0], 3.0, atol=0.01)
        assert jnp.isclose(predicted_at_obs[2] / predicted_at_obs[0], 5.0, atol=0.01)


class TestPoissonNoise:
    """Test PoissonNoise model."""

    def test_poisson_mean_approximation(self, simple_delay_pmf):
        """Test that Poisson samples have mean close to predicted rate."""
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
        )

        infections = jnp.ones(20) * 10000  # Large enough for stable mean

        samples = []
        for seed in range(50):
            with numpyro.handlers.seed(rng_seed=seed):
                result = process.sample(infections=infections, obs=None)
                samples.append(result.observed[10])

        sample_mean = jnp.mean(jnp.array(samples))
        expected_rate = 10000 * 0.01  # 100
        # Mean should be within 20% of expected rate
        assert jnp.abs(sample_mean - expected_rate) / expected_rate < 0.2


class TestCountsValidation:
    """Test validation methods."""

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


class TestNoiseValidation:
    """Test noise model validation methods."""

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
        bad_pmf = jnp.array([0.3, 0.3, 0.3])
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
        bad_pmf = jnp.array([1.5, -0.5])
        with pytest.raises(ValueError, match="must have non-negative values"):
            process._validate_pmf(bad_pmf, "test_pmf")


class TestRightTruncation:
    """Test right-truncation adjustment in count observation processes."""

    def test_no_truncation_rv_unchanged(self, simple_delay_pmf):
        """Test that right_truncation_rv=None produces unchanged behavior."""
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
        )
        infections = jnp.ones(20) * 1000

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections, obs=None, right_truncation_offset=3
            )

        assert result.predicted.shape == infections.shape
        assert jnp.allclose(result.predicted, 10.0)

    def test_truncation_rv_without_offset_unchanged(self, simple_delay_pmf):
        """Test that right_truncation_offset=None skips adjustment."""
        rt_pmf = jnp.array([0.2, 0.3, 0.5])
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            right_truncation_rv=DeterministicPMF("rt_delay", rt_pmf),
        )
        infections = jnp.ones(20) * 1000

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections, obs=None, right_truncation_offset=None
            )

        assert jnp.allclose(result.predicted, 10.0)

    def test_truncation_reduces_recent_counts(self, simple_delay_pmf):
        """Test that right-truncation reduces predicted counts for recent timepoints."""
        rt_pmf = jnp.array([0.2, 0.3, 0.5])
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            right_truncation_rv=DeterministicPMF("rt_delay", rt_pmf),
        )
        infections = jnp.ones(10) * 100

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections, obs=None, right_truncation_offset=0
            )

        assert jnp.isclose(result.predicted[0], 100.0)
        assert jnp.isclose(result.predicted[-2], 50.0)
        assert jnp.isclose(result.predicted[-1], 20.0)

    def test_deterministic_site_recorded(self, simple_delay_pmf):
        """Test that prop_already_reported deterministic site is recorded."""
        rt_pmf = jnp.array([0.2, 0.3, 0.5])
        process = Counts(
            name="hosp",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            right_truncation_rv=DeterministicPMF("rt_delay", rt_pmf),
        )
        infections = jnp.ones(10) * 100

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                process.sample(
                    infections=infections, obs=None, right_truncation_offset=0
                )

        assert "hosp_prop_already_reported" in trace
        prop = trace["hosp_prop_already_reported"]["value"]
        assert prop.shape == (10,)
        assert jnp.all(prop <= 1.0)
        assert jnp.all(prop > 0.0)

    def test_validate_catches_invalid_rt_pmf(self, simple_delay_pmf):
        """Test that validate() rejects invalid right-truncation PMFs."""
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            right_truncation_rv=DeterministicVariable(
                "rt_delay", jnp.array([0.3, 0.3])
            ),
        )
        with pytest.raises(ValueError, match="must sum to 1.0"):
            process.validate()

    def test_short_observation_window_raises(self, simple_delay_pmf):
        """Test that observation window shorter than delay support raises."""
        rt_pmf = jnp.array([0.2, 0.3, 0.5])
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            right_truncation_rv=DeterministicPMF("rt_delay", rt_pmf),
        )
        infections = jnp.ones(2) * 100

        with numpyro.handlers.seed(rng_seed=42):
            with pytest.raises(ValueError, match="Observation window length"):
                process.sample(
                    infections=infections,
                    obs=None,
                    right_truncation_offset=0,
                )

    def test_counts_by_subpop_2d_broadcasting(self):
        """Test right-truncation with CountsBySubpop 2D infections."""
        rt_pmf = jnp.array([0.2, 0.3, 0.5])
        delay_pmf = jnp.array([1.0])
        process = CountsBySubpop(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
            noise=PoissonNoise(),
            right_truncation_rv=DeterministicPMF("rt_delay", rt_pmf),
        )

        n_days = 10
        n_subpops = 3
        infections = jnp.ones((n_days, n_subpops)) * 100
        times = jnp.array([0, 8, 9])
        subpop_indices = jnp.array([0, 1, 2])

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                times=times,
                subpop_indices=subpop_indices,
                obs=None,
                right_truncation_offset=0,
            )

        assert result.predicted.shape == (n_days, n_subpops)
        assert jnp.isclose(result.predicted[0, 0], 100.0)
        assert jnp.isclose(result.predicted[-2, 0], 50.0)
        assert jnp.isclose(result.predicted[-1, 0], 20.0)
        assert jnp.allclose(result.predicted[:, 0], result.predicted[:, 1])


class TestDayOfWeek:
    """Test day-of-week multiplicative adjustment in count observations."""

    def test_no_dow_rv_unchanged(self, simple_delay_pmf):
        """Test that day_of_week_rv=None ignores first_day_dow."""
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
        )
        infections = jnp.ones(20) * 1000

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(infections=infections, obs=None, first_day_dow=3)

        assert jnp.allclose(result.predicted, 10.0)

    def test_dow_rv_without_offset_raises(self, simple_delay_pmf):
        """Test that first_day_dow=None raises when day_of_week_rv is set."""
        dow_effect = jnp.array([2.0, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5])
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            day_of_week_rv=DeterministicVariable("dow", dow_effect),
        )
        infections = jnp.ones(20) * 1000

        with numpyro.handlers.seed(rng_seed=42):
            with pytest.raises(ValueError, match="first_day_dow is required"):
                process.sample(infections=infections, obs=None, first_day_dow=None)

    def test_uniform_dow_effect_unchanged(self, simple_delay_pmf):
        """Test that uniform effect [1,1,...,1] leaves predictions unchanged."""
        dow_effect = jnp.ones(7)
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            day_of_week_rv=DeterministicVariable("dow", dow_effect),
        )
        infections = jnp.ones(14) * 100

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(infections=infections, obs=None, first_day_dow=0)

        assert jnp.allclose(result.predicted, 100.0)

    def test_dow_effect_scales_predictions(self, simple_delay_pmf):
        """Test that known day-of-week effects produce correct per-day scaling.

        With constant infections of 100, ascertainment 1.0, no delay,
        and first_day_dow=0 (Monday), element i of predicted should
        equal 100 * dow_effect[i % 7].
        """
        dow_effect = jnp.array([2.0, 1.5, 1.0, 1.0, 0.5, 0.5, 0.5])
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            day_of_week_rv=DeterministicVariable("dow", dow_effect),
        )
        infections = jnp.ones(14) * 100

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(infections=infections, obs=None, first_day_dow=0)

        assert jnp.isclose(result.predicted[0], 200.0)
        assert jnp.isclose(result.predicted[1], 150.0)
        assert jnp.isclose(result.predicted[4], 50.0)
        assert jnp.isclose(result.predicted[7], 200.0)

    def test_dow_effect_with_multiday_delay(self, short_delay_pmf):
        """Test that DOW ratios are correct with a multi-day delay PMF.

        With a 2-day delay, the first element is NaN (init period).
        Post-init predicted values should satisfy:
        predicted_with_dow[t] / predicted_no_dow[t] == dow_effect[t % 7].
        """
        dow_effect = jnp.array([2.0, 1.5, 1.0, 1.0, 0.5, 0.5, 0.5])
        process_no_dow = Counts(
            name="base",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", short_delay_pmf),
            noise=PoissonNoise(),
        )
        process_with_dow = Counts(
            name="dow",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", short_delay_pmf),
            noise=PoissonNoise(),
            day_of_week_rv=DeterministicVariable("dow", dow_effect),
        )
        infections = jnp.ones(21) * 100

        with numpyro.handlers.seed(rng_seed=42):
            result_no = process_no_dow.sample(infections=infections, obs=None)
        with numpyro.handlers.seed(rng_seed=42):
            result_yes = process_with_dow.sample(
                infections=infections, obs=None, first_day_dow=0
            )

        day_one = 1
        for t in range(day_one, 14):
            expected_ratio = float(dow_effect[t % 7])
            actual_ratio = float(result_yes.predicted[t] / result_no.predicted[t])
            assert jnp.isclose(actual_ratio, expected_ratio, atol=1e-5)

    def test_dow_offset_shifts_pattern(self, simple_delay_pmf):
        """Test that first_day_dow offsets the weekly pattern correctly.

        Starting on Wednesday (dow=2) means element 0 gets
        dow_effect[2], element 1 gets dow_effect[3], etc.
        """
        dow_effect = jnp.array([2.0, 1.5, 1.0, 0.8, 0.7, 0.5, 0.5])
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            day_of_week_rv=DeterministicVariable("dow", dow_effect),
        )
        infections = jnp.ones(7) * 100

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(infections=infections, obs=None, first_day_dow=2)

        assert jnp.isclose(result.predicted[0], 100.0)
        assert jnp.isclose(result.predicted[1], 80.0)
        assert jnp.isclose(result.predicted[5], 200.0)

    def test_deterministic_site_recorded(self, simple_delay_pmf):
        """Test that day_of_week_effect deterministic site is recorded."""
        dow_effect = jnp.ones(7)
        process = Counts(
            name="ed",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            day_of_week_rv=DeterministicVariable("dow", dow_effect),
        )
        infections = jnp.ones(10) * 100

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                process.sample(infections=infections, obs=None, first_day_dow=0)

        assert "ed_day_of_week_effect" in trace
        effect = trace["ed_day_of_week_effect"]["value"]
        assert effect.shape == (7,)

    def test_counts_by_subpop_2d_broadcasting(self):
        """Test day-of-week with CountsBySubpop 2D infections."""
        dow_effect = jnp.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        delay_pmf = jnp.array([1.0])
        process = CountsBySubpop(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
            noise=PoissonNoise(),
            day_of_week_rv=DeterministicVariable("dow", dow_effect),
        )

        n_days = 14
        n_subpops = 3
        infections = jnp.ones((n_days, n_subpops)) * 100
        times = jnp.array([0, 1, 7])
        subpop_indices = jnp.array([0, 1, 2])

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                times=times,
                subpop_indices=subpop_indices,
                obs=None,
                first_day_dow=0,
            )

        assert result.predicted.shape == (n_days, n_subpops)
        assert jnp.isclose(result.predicted[0, 0], 200.0)
        assert jnp.isclose(result.predicted[1, 0], 100.0)
        assert jnp.isclose(result.predicted[7, 0], 200.0)
        assert jnp.allclose(result.predicted[:, 0], result.predicted[:, 1])

    def test_dow_with_right_truncation(self, simple_delay_pmf):
        """Test that day-of-week and right-truncation compose correctly.

        Day-of-week is applied first, then right-truncation scales
        the adjusted predictions.
        """
        dow_effect = jnp.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        rt_pmf = jnp.array([0.2, 0.3, 0.5])
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            right_truncation_rv=DeterministicPMF("rt_delay", rt_pmf),
            day_of_week_rv=DeterministicVariable("dow", dow_effect),
        )
        infections = jnp.ones(10) * 100

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                obs=None,
                right_truncation_offset=0,
                first_day_dow=0,
            )

        assert jnp.isclose(result.predicted[0], 200.0)
        assert jnp.isclose(result.predicted[1], 100.0)
        assert result.predicted[-1] < result.predicted[0]

    def test_validate_catches_wrong_shape(self, simple_delay_pmf):
        """Test that validate() rejects non-length-7 effect vectors."""
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            day_of_week_rv=DeterministicVariable("dow", jnp.ones(5)),
        )
        with pytest.raises(ValueError, match="must return shape \\(7,\\)"):
            process.validate()

    def test_validate_catches_negative_values(self, simple_delay_pmf):
        """Test that validate() rejects negative effect values."""
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            day_of_week_rv=DeterministicVariable(
                "dow", jnp.array([1.0, 1.0, 1.0, -0.5, 1.0, 1.0, 1.0])
            ),
        )
        with pytest.raises(ValueError, match="must have non-negative values"):
            process.validate()

    def test_invalid_first_day_dow_raises(self, simple_delay_pmf):
        """Test that out-of-range first_day_dow raises ValueError."""
        dow_effect = jnp.ones(7)
        process = Counts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            day_of_week_rv=DeterministicVariable("dow", dow_effect),
        )
        infections = jnp.ones(14) * 100

        with numpyro.handlers.seed(rng_seed=42):
            with pytest.raises(ValueError, match="Day-of-week"):
                process.sample(infections=infections, obs=None, first_day_dow=7)

    def test_counts_by_subpop_dow_without_offset_raises(self):
        """Test that first_day_dow=None raises for CountsBySubpop with day_of_week_rv."""
        dow_effect = jnp.ones(7)
        process = CountsBySubpop(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", jnp.array([1.0])),
            noise=PoissonNoise(),
            day_of_week_rv=DeterministicVariable("dow", dow_effect),
        )
        infections = jnp.ones((14, 2)) * 100
        times = jnp.array([0, 1])
        subpop_indices = jnp.array([0, 1])

        with numpyro.handlers.seed(rng_seed=42):
            with pytest.raises(ValueError, match="first_day_dow is required"):
                process.sample(
                    infections=infections,
                    times=times,
                    subpop_indices=subpop_indices,
                    obs=None,
                    first_day_dow=None,
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
