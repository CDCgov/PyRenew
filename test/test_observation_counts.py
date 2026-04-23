"""
Unit tests for PopulationCounts and SubpopulationCounts classes.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.observation import (
    NegativeBinomialNoise,
    PoissonNoise,
    PopulationCounts,
    SubpopulationCounts,
)
from pyrenew.randomvariable import DistributionalVariable
from pyrenew.time import MMWR_WEEK
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
        process = PopulationCounts(
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

        process = PopulationCounts(
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

        process = PopulationCounts(
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


class TestSubpopulationCounts:
    """Test SubpopulationCounts for subpopulation-level observations."""

    def test_non_contiguous_subpop_indices(self):
        """Test that non-contiguous subpop_indices work correctly.

        Verifies observation processes can observe any subset
        of subpopulations with correct value proportionality.
        """
        delay_pmf = jnp.array([0.3, 0.4, 0.3])
        process = SubpopulationCounts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
            noise=PoissonNoise(),
            reporting_schedule="irregular",
        )

        # 5 subpopulations with distinct infection levels
        n_days = 20
        infections = jnp.zeros((n_days, 5))
        for k in range(5):
            infections = infections.at[:, k].set((k + 1) * 100.0)

        period_end_times = jnp.array([10, 10, 10])
        subpop_indices = jnp.array([0, 2, 4])

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                period_end_times=period_end_times,
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
            noise.validate_concentration_rv()

    def test_negative_binomial_noise_validate_negative_concentration(self):
        """Test NegativeBinomialNoise validate with negative concentration."""
        noise = NegativeBinomialNoise(DeterministicVariable("conc", -1.0))
        with pytest.raises(ValueError, match="concentration must be positive"):
            noise.validate_concentration_rv()


class TestBaseObservationProcessValidation:
    """Test base observation process PMF validation."""

    def test_validate_pmf_empty_array(self, simple_delay_pmf):
        """Test that _validate_pmf raises for empty array."""
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        """Test right-truncation with SubpopulationCounts 2D infections."""
        rt_pmf = jnp.array([0.2, 0.3, 0.5])
        delay_pmf = jnp.array([1.0])
        process = SubpopulationCounts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
            noise=PoissonNoise(),
            right_truncation_rv=DeterministicPMF("rt_delay", rt_pmf),
            reporting_schedule="irregular",
        )

        n_days = 10
        n_subpops = 3
        infections = jnp.ones((n_days, n_subpops)) * 100
        period_end_times = jnp.array([0, 8, 9])
        subpop_indices = jnp.array([0, 1, 2])

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                period_end_times=period_end_times,
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        process_no_dow = PopulationCounts(
            name="base",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", short_delay_pmf),
            noise=PoissonNoise(),
        )
        process_with_dow = PopulationCounts(
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        """Test day-of-week with SubpopulationCounts 2D infections."""
        dow_effect = jnp.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        delay_pmf = jnp.array([1.0])
        process = SubpopulationCounts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
            noise=PoissonNoise(),
            day_of_week_rv=DeterministicVariable("dow", dow_effect),
            reporting_schedule="irregular",
        )

        n_days = 14
        n_subpops = 3
        infections = jnp.ones((n_days, n_subpops)) * 100
        period_end_times = jnp.array([0, 1, 7])
        subpop_indices = jnp.array([0, 1, 2])

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=infections,
                period_end_times=period_end_times,
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        process = PopulationCounts(
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
        """Test that first_day_dow=None raises for SubpopulationCounts with day_of_week_rv."""
        dow_effect = jnp.ones(7)
        process = SubpopulationCounts(
            name="test",
            ascertainment_rate_rv=DeterministicVariable("ihr", 1.0),
            delay_distribution_rv=DeterministicPMF("delay", jnp.array([1.0])),
            noise=PoissonNoise(),
            day_of_week_rv=DeterministicVariable("dow", dow_effect),
            reporting_schedule="irregular",
        )
        infections = jnp.ones((14, 2)) * 100
        period_end_times = jnp.array([0, 1])
        subpop_indices = jnp.array([0, 1])

        with numpyro.handlers.seed(rng_seed=42):
            with pytest.raises(ValueError, match="first_day_dow is required"):
                process.sample(
                    infections=infections,
                    period_end_times=period_end_times,
                    subpop_indices=subpop_indices,
                    obs=None,
                    first_day_dow=None,
                )


# ===================================================================
# PopulationCounts with aggregation: construction-time validation
# ===================================================================


class TestPopulationCountsAggregationConstruction:
    """Construction-time validation for the aggregation parameters."""

    def _make(self, simple_delay_pmf, **kwargs):
        """
        Build a PopulationCounts with a stub noise model and optional overrides.

        Returns
        -------
        PopulationCounts
            A PopulationCounts instance using the supplied delay PMF and
            any additional constructor overrides passed via ``**kwargs``.
        """
        return PopulationCounts(
            name="hosp",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
            **kwargs,
        )

    def test_default_construction_is_daily_regular(self, simple_delay_pmf):
        """Default constructor yields aggregation='daily', reporting_schedule='regular'."""
        process = self._make(simple_delay_pmf)
        assert process.aggregation == "daily"
        assert process.reporting_schedule == "regular"
        assert process.week is None

    def test_weekly_requires_week(self, simple_delay_pmf):
        """aggregation='weekly' without week must raise."""
        with pytest.raises(ValueError, match="week is required"):
            self._make(simple_delay_pmf, aggregation="weekly")

    def test_weekly_with_mmwr_anchor_constructs(self, simple_delay_pmf):
        """aggregation='weekly' with MMWR_WEEK is valid."""
        process = self._make(simple_delay_pmf, aggregation="weekly", week=MMWR_WEEK)
        assert process.aggregation == "weekly"
        assert process.week == MMWR_WEEK

    def test_dow_effect_with_weekly_aggregation_raises(self, simple_delay_pmf):
        """day_of_week_rv cannot be combined with aggregation='weekly'."""
        with pytest.raises(ValueError, match="day_of_week_rv cannot be combined"):
            self._make(
                simple_delay_pmf,
                aggregation="weekly",
                week=MMWR_WEEK,
                day_of_week_rv=DeterministicVariable("dow", jnp.ones(7)),
            )

    def test_dow_effect_with_daily_aggregation_allowed(self, simple_delay_pmf):
        """day_of_week_rv remains valid for aggregation='daily'."""
        process = self._make(
            simple_delay_pmf,
            day_of_week_rv=DeterministicVariable("dow", jnp.ones(7)),
        )
        assert process.day_of_week_rv is not None

    def test_unknown_aggregation_raises(self, simple_delay_pmf):
        """aggregation must be 'daily' or 'weekly'."""
        with pytest.raises(ValueError, match="aggregation must be one of"):
            self._make(simple_delay_pmf, aggregation="monthly")

    def test_unknown_reporting_schedule_raises(self, simple_delay_pmf):
        """reporting_schedule must be 'regular' or 'irregular'."""
        with pytest.raises(ValueError, match="reporting_schedule must be one of"):
            self._make(simple_delay_pmf, reporting_schedule="sporadic")


# ===================================================================
# PopulationCounts with aggregation: validate_data
# ===================================================================


class TestPopulationCountsAggregationValidateData:
    """validate_data branches for each (schedule, aggregation_period) combination."""

    def test_weekly_regular_correct_length_passes(self, weekly_regular_counts):
        """Weekly-regular obs of length n_periods passes when first_day_dow aligns."""
        obs = jnp.ones(4) * 10.0
        weekly_regular_counts.validate_data(
            n_total=28, n_subpops=1, obs=obs, first_day_dow=6
        )

    def test_weekly_regular_wrong_length_raises(self, weekly_regular_counts):
        """Weekly-regular obs with wrong length raises."""
        obs = jnp.ones(28) * 10.0
        with pytest.raises(ValueError, match="must equal n_periods"):
            weekly_regular_counts.validate_data(
                n_total=28, n_subpops=1, obs=obs, first_day_dow=6
            )

    def test_weekly_regular_missing_first_day_dow_raises(self, weekly_regular_counts):
        """Weekly-regular with first_day_dow=None raises."""
        obs = jnp.ones(4) * 10.0
        with pytest.raises(ValueError, match="first_day_dow is required"):
            weekly_regular_counts.validate_data(n_total=28, n_subpops=1, obs=obs)

    def test_weekly_regular_obs_none_passes(self, weekly_regular_counts):
        """Weekly-regular with obs=None skips checks."""
        weekly_regular_counts.validate_data(n_total=28, n_subpops=1)

    def test_weekly_regular_honors_offset(self, weekly_regular_counts):
        """Weekly-regular n_periods reflects front-trim offset."""
        obs = jnp.ones(3) * 10.0
        weekly_regular_counts.validate_data(
            n_total=28, n_subpops=1, obs=obs, first_day_dow=0
        )

    def test_weekly_irregular_aligned_times_pass(self, weekly_irregular_counts):
        """Weekly-irregular with Saturdays at offset 0 passes."""
        period_end_times = jnp.array([6, 13, 20])
        obs = jnp.ones(3) * 10.0
        weekly_irregular_counts.validate_data(
            n_total=28,
            n_subpops=1,
            obs=obs,
            period_end_times=period_end_times,
            first_day_dow=6,
        )

    def test_weekly_irregular_misaligned_times_raise(self, weekly_irregular_counts):
        """Weekly-irregular with non-Saturday period_end_times raises."""
        period_end_times = jnp.array([5, 13, 20])
        with pytest.raises(ValueError, match="period_end_times must lie on"):
            weekly_irregular_counts.validate_data(
                n_total=28,
                n_subpops=1,
                period_end_times=period_end_times,
                first_day_dow=6,
            )

    def test_weekly_irregular_missing_first_day_dow_raises(
        self, weekly_irregular_counts
    ):
        """Weekly-irregular with first_day_dow=None raises."""
        period_end_times = jnp.array([6, 13, 20])
        with pytest.raises(ValueError, match="first_day_dow is required"):
            weekly_irregular_counts.validate_data(
                n_total=28, n_subpops=1, period_end_times=period_end_times
            )

    def test_weekly_irregular_obs_shape_mismatch_raises(self, weekly_irregular_counts):
        """Weekly-irregular obs of wrong length raises."""
        period_end_times = jnp.array([6, 13, 20])
        obs = jnp.ones(2) * 10.0
        with pytest.raises(ValueError, match="must match"):
            weekly_irregular_counts.validate_data(
                n_total=28,
                n_subpops=1,
                obs=obs,
                period_end_times=period_end_times,
                first_day_dow=6,
            )

    def test_daily_irregular_passes(self, daily_irregular_counts):
        """Daily-irregular validates via bounds only; alignment is trivial."""
        period_end_times = jnp.array([0, 5, 19])
        obs = jnp.ones(3) * 10.0
        daily_irregular_counts.validate_data(
            n_total=20,
            n_subpops=1,
            obs=obs,
            period_end_times=period_end_times,
        )

    def test_daily_irregular_out_of_bounds_raises(self, daily_irregular_counts):
        """Daily-irregular out-of-bounds index raises."""
        period_end_times = jnp.array([0, 5, 25])
        with pytest.raises(ValueError, match="upper bound"):
            daily_irregular_counts.validate_data(
                n_total=20, n_subpops=1, period_end_times=period_end_times
            )

    def test_irregular_no_period_end_times_is_noop(self, weekly_irregular_counts):
        """Irregular schedule with period_end_times=None returns without error."""
        weekly_irregular_counts.validate_data(
            n_total=28, n_subpops=1, obs=None, period_end_times=None
        )


# ===================================================================
# PopulationCounts with aggregation: sample
# ===================================================================


class TestPopulationCountsAggregationSample:
    """Sample-time behavior for the new aggregation paths."""

    def test_weekly_regular_predicted_shape(self, weekly_regular_counts):
        """Weekly-regular predicted has shape (n_periods,) equal to weekly sums."""
        infections = jnp.ones(28) * 100.0
        with numpyro.handlers.seed(rng_seed=42):
            result = weekly_regular_counts.sample(
                infections=infections, first_day_dow=6
            )
        assert result.predicted.shape == (4,)
        assert jnp.allclose(result.predicted, 7.0, rtol=1e-5)

    def test_weekly_regular_emits_predicted_daily_site(self, weekly_regular_counts):
        """When aggregation_period > 1, the 'predicted_daily' deterministic site exists."""
        infections = jnp.ones(28) * 100.0
        with numpyro.handlers.trace() as trace:
            with numpyro.handlers.seed(rng_seed=42):
                weekly_regular_counts.sample(infections=infections, first_day_dow=6)
        assert "hosp_predicted_daily" in trace
        assert "hosp_predicted" in trace
        assert trace["hosp_predicted_daily"]["value"].shape == (28,)
        assert trace["hosp_predicted"]["value"].shape == (4,)

    def test_daily_backward_compat_no_predicted_daily_site(self, simple_delay_pmf):
        """When aggregation_period == 1, 'predicted_daily' is not emitted."""
        process = PopulationCounts(
            name="hosp",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
        )
        infections = jnp.ones(28) * 100.0
        with numpyro.handlers.trace() as trace:
            with numpyro.handlers.seed(rng_seed=42):
                process.sample(infections=infections)
        assert "hosp_predicted_daily" not in trace
        assert "hosp_predicted" in trace
        assert trace["hosp_predicted"]["value"].shape == (28,)

    def test_weekly_regular_with_obs_runs(self, weekly_regular_counts):
        """Weekly-regular sample accepts dense-with-NaN obs on the period grid."""
        infections = jnp.ones(28) * 100.0
        obs = jnp.array([7.0, jnp.nan, 7.0, 7.0])
        with numpyro.handlers.seed(rng_seed=42):
            result = weekly_regular_counts.sample(
                infections=infections, obs=obs, first_day_dow=6
            )
        assert result.observed.shape == (4,)
        assert result.predicted.shape == (4,)

    def test_weekly_irregular_period_indexing(self, weekly_irregular_counts):
        """Weekly-irregular fancy-indexes the aggregated array at correct periods."""
        infections = jnp.ones(28) * 100.0
        period_end_times = jnp.array([6, 20])
        with numpyro.handlers.seed(rng_seed=42):
            result = weekly_irregular_counts.sample(
                infections=infections,
                period_end_times=period_end_times,
                first_day_dow=6,
            )
        assert result.predicted.shape == (4,)
        assert result.observed.shape == (2,)
        assert jnp.allclose(result.predicted, 7.0, rtol=1e-5)

    def test_weekly_irregular_missing_period_end_times_raises(
        self, weekly_irregular_counts
    ):
        """Irregular schedule requires period_end_times at sample time."""
        infections = jnp.ones(28) * 100.0
        with pytest.raises(ValueError, match="period_end_times is required"):
            with numpyro.handlers.seed(rng_seed=42):
                weekly_irregular_counts.sample(infections=infections, first_day_dow=6)

    def test_daily_irregular_period_indexing(self, daily_irregular_counts):
        """Daily-irregular fancy-indexes at the supplied daily indices directly."""
        infections = jnp.ones(30) * 100.0
        period_end_times = jnp.array([5, 10, 20])
        with numpyro.handlers.seed(rng_seed=42):
            result = daily_irregular_counts.sample(
                infections=infections, period_end_times=period_end_times
            )
        assert result.predicted.shape == (30,)
        assert result.observed.shape == (3,)

    def test_aggregate_helper_missing_first_day_dow_raises(self, weekly_regular_counts):
        """_aggregate raises when aggregation == 'weekly' and first_day_dow is None."""
        predicted_daily = jnp.ones(28)
        with pytest.raises(
            ValueError, match="first_day_dow is required when aggregation == 'weekly'"
        ):
            weekly_regular_counts._aggregate(predicted_daily, first_day_dow=None)


# ===================================================================
# SubpopulationCounts with aggregation: validate_data
# ===================================================================


class TestSubpopulationCountsAggregationValidateData:
    """validate_data branches for each (schedule, aggregation_period) combination."""

    def test_daily_regular_valid_passes(self, simple_delay_pmf):
        """Daily-regular dense 2D obs (n_total, n_observed_subpops) passes."""
        process = SubpopulationCounts(
            name="ed",
            ascertainment_rate_rv=DeterministicVariable("iedr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
        )
        obs = jnp.ones((30, 2)) * 5.0
        subpop_indices = jnp.array([0, 2])
        process.validate_data(
            n_total=30, n_subpops=3, obs=obs, subpop_indices=subpop_indices
        )

    def test_weekly_regular_valid_passes(self, weekly_regular_subpop_counts):
        """Weekly-regular dense 2D obs (n_periods, n_observed_subpops) passes."""
        obs = jnp.ones((4, 2)) * 5.0
        subpop_indices = jnp.array([0, 2])
        weekly_regular_subpop_counts.validate_data(
            n_total=28,
            n_subpops=3,
            obs=obs,
            first_day_dow=6,
            subpop_indices=subpop_indices,
        )

    def test_regular_no_obs_is_noop(self, weekly_regular_subpop_counts):
        """Regular schedule with obs=None returns without error."""
        weekly_regular_subpop_counts.validate_data(n_total=28, n_subpops=3, obs=None)

    def test_weekly_regular_wrong_n_periods_raises(self, weekly_regular_subpop_counts):
        """Weekly-regular obs with wrong dim-0 length raises."""
        obs = jnp.ones((28, 2)) * 5.0
        subpop_indices = jnp.array([0, 2])
        with pytest.raises(ValueError, match="must equal n_periods"):
            weekly_regular_subpop_counts.validate_data(
                n_total=28,
                n_subpops=3,
                obs=obs,
                first_day_dow=6,
                subpop_indices=subpop_indices,
            )

    def test_regular_wrong_n_subpops_raises(self, weekly_regular_subpop_counts):
        """Regular-schedule obs dim-1 must equal len(subpop_indices)."""
        obs = jnp.ones((4, 3)) * 5.0
        subpop_indices = jnp.array([0, 2])
        with pytest.raises(ValueError, match=r"must equal len\(subpop_indices\)"):
            weekly_regular_subpop_counts.validate_data(
                n_total=28,
                n_subpops=3,
                obs=obs,
                first_day_dow=6,
                subpop_indices=subpop_indices,
            )

    def test_weekly_regular_missing_first_day_dow_raises(
        self, weekly_regular_subpop_counts
    ):
        """Weekly-regular without first_day_dow raises."""
        obs = jnp.ones((4, 2)) * 5.0
        subpop_indices = jnp.array([0, 2])
        with pytest.raises(ValueError, match="first_day_dow is required"):
            weekly_regular_subpop_counts.validate_data(
                n_total=28,
                n_subpops=3,
                obs=obs,
                subpop_indices=subpop_indices,
            )

    def test_regular_1d_obs_raises(self, weekly_regular_subpop_counts):
        """Regular-schedule obs must be 2D."""
        obs = jnp.ones(4) * 5.0
        subpop_indices = jnp.array([0])
        with pytest.raises(ValueError, match="regular-schedule obs must be 2D"):
            weekly_regular_subpop_counts.validate_data(
                n_total=28,
                n_subpops=3,
                obs=obs,
                first_day_dow=6,
                subpop_indices=subpop_indices,
            )

    def test_regular_bad_subpop_indices_raises(self, weekly_regular_subpop_counts):
        """Regular-schedule out-of-bounds subpop_indices raises."""
        subpop_indices = jnp.array([0, 5])
        with pytest.raises(ValueError, match="upper bound"):
            weekly_regular_subpop_counts.validate_data(
                n_total=28, n_subpops=3, subpop_indices=subpop_indices
            )

    def test_daily_irregular_valid_passes(self, daily_irregular_subpop_counts):
        """Daily-irregular with valid period_end_times and subpop_indices passes."""
        period_end_times = jnp.array([5, 10, 15, 20])
        subpop_indices = jnp.array([0, 1, 2, 0])
        obs = jnp.array([10.0, 20.0, 30.0, 15.0])
        daily_irregular_subpop_counts.validate_data(
            n_total=30,
            n_subpops=3,
            obs=obs,
            period_end_times=period_end_times,
            subpop_indices=subpop_indices,
        )

    def test_weekly_irregular_valid_passes(
        self, weekly_irregular_subpop_counts, mmwr_saturday_indices_first_three
    ):
        """Weekly-irregular with Saturdays at offset 0 passes."""
        subpop_indices = jnp.array([0, 1, 2])
        obs = jnp.array([10.0, 20.0, 30.0])
        weekly_irregular_subpop_counts.validate_data(
            n_total=28,
            n_subpops=3,
            obs=obs,
            period_end_times=mmwr_saturday_indices_first_three,
            first_day_dow=6,
            subpop_indices=subpop_indices,
        )

    def test_weekly_irregular_misaligned_raises(self, weekly_irregular_subpop_counts):
        """Weekly-irregular with non-Saturday period_end_times raises."""
        period_end_times = jnp.array([5, 13, 20])
        subpop_indices = jnp.array([0, 1, 2])
        with pytest.raises(ValueError, match="period_end_times must lie on"):
            weekly_irregular_subpop_counts.validate_data(
                n_total=28,
                n_subpops=3,
                period_end_times=period_end_times,
                first_day_dow=6,
                subpop_indices=subpop_indices,
            )

    def test_weekly_irregular_missing_first_day_dow_raises(
        self, weekly_irregular_subpop_counts, mmwr_saturday_indices_first_three
    ):
        """Weekly-irregular without first_day_dow raises."""
        with pytest.raises(ValueError, match="first_day_dow is required"):
            weekly_irregular_subpop_counts.validate_data(
                n_total=28,
                n_subpops=3,
                period_end_times=mmwr_saturday_indices_first_three,
            )

    def test_irregular_obs_shape_mismatch_raises(
        self, weekly_irregular_subpop_counts, mmwr_saturday_indices_first_three
    ):
        """Irregular-schedule obs length must match period_end_times."""
        obs = jnp.array([10.0, 20.0])
        with pytest.raises(ValueError, match="must match"):
            weekly_irregular_subpop_counts.validate_data(
                n_total=28,
                n_subpops=3,
                obs=obs,
                period_end_times=mmwr_saturday_indices_first_three,
                first_day_dow=6,
            )

    def test_irregular_subpop_indices_shape_mismatch_raises(
        self, weekly_irregular_subpop_counts, mmwr_saturday_indices_first_three
    ):
        """Irregular-schedule subpop_indices length must match period_end_times."""
        subpop_indices = jnp.array([0, 1])
        with pytest.raises(ValueError, match="must match"):
            weekly_irregular_subpop_counts.validate_data(
                n_total=28,
                n_subpops=3,
                period_end_times=mmwr_saturday_indices_first_three,
                first_day_dow=6,
                subpop_indices=subpop_indices,
            )


# ===================================================================
# SubpopulationCounts with aggregation: sample
# ===================================================================


class TestSubpopulationCountsAggregationSample:
    """Sample-time behavior for the new aggregation paths."""

    def test_daily_regular_shape(self, simple_delay_pmf, subpop_infections_30d):
        """Daily-regular sample returns predicted of shape (n_total, n_subpops)."""
        process = SubpopulationCounts(
            name="ed",
            ascertainment_rate_rv=DeterministicVariable("iedr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
        )
        subpop_indices = jnp.array([0, 2])
        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                infections=subpop_infections_30d, subpop_indices=subpop_indices
            )
        assert result.predicted.shape == (30, 3)
        assert result.observed.shape == (30, 2)

    def test_weekly_regular_predicted_shape_and_values(
        self, weekly_regular_subpop_counts, subpop_infections_28d
    ):
        """Weekly-regular aggregates (n_total, n_subpops) to (n_periods, n_subpops)."""
        subpop_indices = jnp.array([0, 2])
        with numpyro.handlers.seed(rng_seed=42):
            result = weekly_regular_subpop_counts.sample(
                infections=subpop_infections_28d,
                first_day_dow=6,
                subpop_indices=subpop_indices,
            )
        assert result.predicted.shape == (4, 3)
        assert jnp.allclose(result.predicted, 7.0, rtol=1e-5)
        assert result.observed.shape == (4, 2)

    def test_weekly_regular_emits_predicted_daily_site(
        self, weekly_regular_subpop_counts, subpop_infections_28d
    ):
        """aggregation_period > 1 emits a 'predicted_daily' deterministic site."""
        subpop_indices = jnp.array([0, 2])
        with numpyro.handlers.trace() as trace:
            with numpyro.handlers.seed(rng_seed=42):
                weekly_regular_subpop_counts.sample(
                    infections=subpop_infections_28d,
                    first_day_dow=6,
                    subpop_indices=subpop_indices,
                )
        assert "ed_predicted_daily" in trace
        assert "ed_predicted" in trace
        assert trace["ed_predicted_daily"]["value"].shape == (28, 3)
        assert trace["ed_predicted"]["value"].shape == (4, 3)

    def test_daily_backward_compat_no_predicted_daily_site(
        self, simple_delay_pmf, subpop_infections_28d
    ):
        """aggregation_period == 1 emits only 'predicted', not 'predicted_daily'."""
        process = SubpopulationCounts(
            name="ed",
            ascertainment_rate_rv=DeterministicVariable("iedr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
            noise=PoissonNoise(),
        )
        subpop_indices = jnp.array([0, 1, 2])
        with numpyro.handlers.trace() as trace:
            with numpyro.handlers.seed(rng_seed=42):
                process.sample(
                    infections=subpop_infections_28d, subpop_indices=subpop_indices
                )
        assert "ed_predicted_daily" not in trace
        assert "ed_predicted" in trace
        assert trace["ed_predicted"]["value"].shape == (28, 3)

    def test_missing_subpop_indices_raises(
        self, weekly_regular_subpop_counts, subpop_infections_28d
    ):
        """subpop_indices is required in every sample call."""
        with pytest.raises(ValueError, match="subpop_indices is required"):
            with numpyro.handlers.seed(rng_seed=42):
                weekly_regular_subpop_counts.sample(
                    infections=subpop_infections_28d, first_day_dow=6
                )

    def test_weekly_irregular_period_indexing(
        self, weekly_irregular_subpop_counts, subpop_infections_28d
    ):
        """Weekly-irregular fancy-indexes the aggregated array at (period, subpop)."""
        period_end_times = jnp.array([6, 20])
        subpop_indices = jnp.array([0, 2])
        with numpyro.handlers.seed(rng_seed=42):
            result = weekly_irregular_subpop_counts.sample(
                infections=subpop_infections_28d,
                period_end_times=period_end_times,
                first_day_dow=6,
                subpop_indices=subpop_indices,
            )
        assert result.predicted.shape == (4, 3)
        assert result.observed.shape == (2,)
        assert jnp.allclose(result.predicted, 7.0, rtol=1e-5)

    def test_weekly_irregular_missing_period_end_times_raises(
        self, weekly_irregular_subpop_counts, subpop_infections_28d
    ):
        """Irregular schedule requires period_end_times at sample time."""
        subpop_indices = jnp.array([0, 1, 2])
        with pytest.raises(ValueError, match="period_end_times is required"):
            with numpyro.handlers.seed(rng_seed=42):
                weekly_irregular_subpop_counts.sample(
                    infections=subpop_infections_28d,
                    first_day_dow=6,
                    subpop_indices=subpop_indices,
                )

    def test_daily_irregular_fancy_indexing(
        self, daily_irregular_subpop_counts, subpop_infections_30d
    ):
        """Daily-irregular indexes predicted at (period_end_times, subpop_indices)."""
        period_end_times = jnp.array([5, 10, 20])
        subpop_indices = jnp.array([0, 1, 2])
        with numpyro.handlers.seed(rng_seed=42):
            result = daily_irregular_subpop_counts.sample(
                infections=subpop_infections_30d,
                period_end_times=period_end_times,
                subpop_indices=subpop_indices,
            )
        assert result.predicted.shape == (30, 3)
        assert result.observed.shape == (3,)

    def test_weekly_regular_with_obs_conditions(
        self, weekly_regular_subpop_counts, subpop_infections_28d
    ):
        """Weekly-regular sample conditions on 2D obs with NaN-padding for unobserved periods."""
        subpop_indices = jnp.array([0, 2])
        obs = jnp.array(
            [
                [7.0, 7.0],
                [jnp.nan, jnp.nan],
                [7.0, 7.0],
                [7.0, 7.0],
            ]
        )
        with numpyro.handlers.seed(rng_seed=42):
            result = weekly_regular_subpop_counts.sample(
                infections=subpop_infections_28d,
                obs=obs,
                first_day_dow=6,
                subpop_indices=subpop_indices,
            )
        assert result.predicted.shape == (4, 3)
        assert result.observed.shape == (4, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
