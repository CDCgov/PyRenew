"""
Unit tests for temporal processes.
"""

import jax.numpy as jnp
import numpyro
import pytest

from pyrenew.latent import (
    AR1,
    DifferencedAR1,
    RandomWalk,
    StepwiseTemporalProcess,
    WeeklyTemporalProcess,
)
from pyrenew.time import MMWR_WEEK

INNER_PROCESS_PARAMS = [
    (AR1, {"autoreg": 0.9, "innovation_sd": 0.05}),
    (DifferencedAR1, {"autoreg": 0.9, "innovation_sd": 0.05}),
    (RandomWalk, {"innovation_sd": 0.05}),
]


class TestTemporalProcessVectorizedSampling:
    """Test vectorized sampling across all temporal process types."""

    @pytest.mark.parametrize(
        "process_cls,kwargs",
        [
            (AR1, {"autoreg": 0.9, "innovation_sd": 0.05}),
            (DifferencedAR1, {"autoreg": 0.9, "innovation_sd": 0.05}),
            (RandomWalk, {"innovation_sd": 0.05}),
        ],
    )
    def test_vectorized_shape_and_initial_values_array(self, process_cls, kwargs):
        """Test shape and initial value handling with array initial values."""
        n_timepoints = 30
        n_processes = 4
        initial_values = jnp.array([0.0, 1.0, -1.0, 2.0])

        process = process_cls(**kwargs)

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = process.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=initial_values,
            )

        assert trajectories.shape == (n_timepoints, n_processes)

    @pytest.mark.parametrize(
        "process_cls,kwargs",
        [
            (AR1, {"autoreg": 0.9, "innovation_sd": 0.05}),
            (DifferencedAR1, {"autoreg": 0.9, "innovation_sd": 0.05}),
            (RandomWalk, {"innovation_sd": 0.05}),
        ],
    )
    def test_vectorized_shape_with_scalar_initial_value(self, process_cls, kwargs):
        """Test shape with scalar initial value broadcast."""
        n_timepoints = 30
        n_processes = 3

        process = process_cls(**kwargs)

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = process.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=1.0,
            )

        assert trajectories.shape == (n_timepoints, n_processes)


class TestRandomWalkInitialValues:
    """Test that RandomWalk preserves initial values."""

    def test_random_walk_vectorized_with_initial_values_array(self):
        """Test RandomWalk first row equals initial values."""
        n_timepoints = 30
        n_processes = 4
        initial_values = jnp.array([0.0, 1.0, -1.0, 2.0])

        rw = RandomWalk(innovation_sd=0.3)

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = rw.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=initial_values,
            )

        assert trajectories.shape == (n_timepoints, n_processes)
        assert jnp.allclose(trajectories[0, :], initial_values)

    def test_random_walk_vectorized_with_scalar_initial_value(self):
        """Test RandomWalk first row equals broadcast scalar."""
        n_timepoints = 30
        n_processes = 3

        rw = RandomWalk(innovation_sd=0.3)

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = rw.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=1.0,
            )

        assert trajectories.shape == (n_timepoints, n_processes)
        assert jnp.allclose(trajectories[0, :], 1.0)


class TestTemporalProcessInnovationSD:
    """Test that temporal processes correctly use innovation_sd parameter."""

    def test_random_walk_smaller_innovation_sd_produces_smoother_trajectory(
        self,
    ):
        """Verify that smaller innovation_sd produces less volatile trajectories."""
        n_timepoints = 100

        with numpyro.handlers.seed(rng_seed=42):
            rw_small = RandomWalk(innovation_sd=0.1)
            trajectory_small = rw_small.sample(n_timepoints=n_timepoints)

        with numpyro.handlers.seed(rng_seed=42):
            rw_large = RandomWalk(innovation_sd=1.0)
            trajectory_large = rw_large.sample(n_timepoints=n_timepoints)

        steps_small = jnp.abs(jnp.diff(trajectory_small[:, 0]))
        steps_large = jnp.abs(jnp.diff(trajectory_large[:, 0]))

        assert jnp.mean(steps_small) < jnp.mean(steps_large)
        assert jnp.max(steps_small) < jnp.max(steps_large)

    def test_ar1_smaller_innovation_sd_produces_lower_variance(self):
        """Verify AR1 with smaller innovation_sd produces lower variance trajectories."""
        n_timepoints = 100
        autoreg = 0.7

        with numpyro.handlers.seed(rng_seed=42):
            ar_small = AR1(autoreg=autoreg, innovation_sd=0.2)
            trajectory_small = ar_small.sample(n_timepoints=n_timepoints)

        with numpyro.handlers.seed(rng_seed=42):
            ar_large = AR1(autoreg=autoreg, innovation_sd=1.0)
            trajectory_large = ar_large.sample(n_timepoints=n_timepoints)

        burn_in = 20
        var_small = jnp.var(trajectory_small[burn_in:, 0])
        var_large = jnp.var(trajectory_large[burn_in:, 0])

        assert var_small < var_large

    def test_differenced_ar1_smaller_innovation_sd_produces_smoother_changes(
        self,
    ):
        """Verify DifferencedAR1 with smaller innovation_sd produces smoother changes."""
        n_timepoints = 100
        autoreg = 0.6

        with numpyro.handlers.seed(rng_seed=42):
            dar_small = DifferencedAR1(autoreg=autoreg, innovation_sd=0.15)
            trajectory_small = dar_small.sample(n_timepoints=n_timepoints)

        with numpyro.handlers.seed(rng_seed=42):
            dar_large = DifferencedAR1(autoreg=autoreg, innovation_sd=0.8)
            trajectory_large = dar_large.sample(n_timepoints=n_timepoints)

        diffs_small = jnp.diff(trajectory_small[:, 0])
        diffs_large = jnp.diff(trajectory_large[:, 0])

        assert jnp.std(diffs_small) < jnp.std(diffs_large)

    def test_vectorized_sampling_respects_innovation_sd(self):
        """Verify vectorized sampling uses the innovation_sd parameter."""
        n_processes = 5
        n_timepoints = 50

        with numpyro.handlers.seed(rng_seed=42):
            rw_small = RandomWalk(innovation_sd=0.2)
            trajs_small = rw_small.sample(
                n_timepoints=n_timepoints, n_processes=n_processes
            )

        with numpyro.handlers.seed(rng_seed=42):
            rw_large = RandomWalk(innovation_sd=1.0)
            trajs_large = rw_large.sample(
                n_timepoints=n_timepoints, n_processes=n_processes
            )

        steps_small = jnp.abs(jnp.diff(trajs_small, axis=0))
        steps_large = jnp.abs(jnp.diff(trajs_large, axis=0))

        assert jnp.mean(steps_small) < jnp.mean(steps_large)

    def test_validation_rejects_non_positive_innovation_sd(self):
        """Verify that non-positive innovation_sd values are rejected."""
        with pytest.raises(ValueError, match="innovation_sd must be positive"):
            RandomWalk(innovation_sd=0.0)

        with pytest.raises(ValueError, match="innovation_sd must be positive"):
            RandomWalk(innovation_sd=-0.5)

        with pytest.raises(ValueError, match="innovation_sd must be positive"):
            AR1(autoreg=0.5, innovation_sd=-0.1)

        with pytest.raises(ValueError, match="innovation_sd must be positive"):
            DifferencedAR1(autoreg=0.5, innovation_sd=0.0)


class TestTemporalProcessBehavior:
    """Test behavioral properties of temporal processes."""

    def test_ar1_mean_reversion(self):
        """Test that AR1 reverts toward zero from a displaced initial value."""
        ar1 = AR1(autoreg=0.95, innovation_sd=0.05)

        with numpyro.handlers.seed(rng_seed=42):
            trajectory = ar1.sample(
                n_timepoints=100,
                initial_value=2.0,
            )

        # The trajectory mean should be closer to 0 than the initial value
        # due to mean-reverting dynamics
        trajectory_mean = jnp.mean(trajectory[:, 0])
        assert jnp.abs(trajectory_mean) < jnp.abs(2.0)

    def test_differenced_ar1_trend_persistence(self):
        """Test that DifferencedAR1 produces persistent trends."""
        dar1 = DifferencedAR1(autoreg=0.95, innovation_sd=0.01)

        with numpyro.handlers.seed(rng_seed=42):
            trajectory = dar1.sample(
                n_timepoints=50,
                initial_value=0.1,
            )

        # With positive initial rate and high autoreg, the differences
        # (growth rates) should remain predominantly positive,
        # producing a persistent upward trend
        diffs = jnp.diff(trajectory[:, 0])
        fraction_positive = jnp.mean(diffs > 0)
        assert fraction_positive > 0.5


class TestTemporalProcessStepSizeDefault:
    """Standard temporal processes expose step_size=1 as a class attribute."""

    @pytest.mark.parametrize(
        "process_cls",
        [AR1, DifferencedAR1, RandomWalk],
    )
    def test_class_attribute_step_size_is_one(self, process_cls):
        """Each concrete class has step_size=1 as a class attribute."""
        assert process_cls.step_size == 1


class TestStepwiseTemporalProcessConstruction:
    """Construction-time validation for StepwiseTemporalProcess."""

    def test_step_size_attribute(self):
        """step_size is exposed on the instance for builder inspection."""
        wrapper = StepwiseTemporalProcess(
            AR1(autoreg=0.9, innovation_sd=0.05), step_size=7
        )
        assert wrapper.step_size == 7

    def test_zero_step_size_raises(self):
        """step_size=0 raises."""
        with pytest.raises(ValueError, match="positive integer"):
            StepwiseTemporalProcess(AR1(autoreg=0.9, innovation_sd=0.05), step_size=0)

    def test_negative_step_size_raises(self):
        """Negative step_size raises."""
        with pytest.raises(ValueError, match="positive integer"):
            StepwiseTemporalProcess(AR1(autoreg=0.9, innovation_sd=0.05), step_size=-1)

    def test_float_step_size_raises(self):
        """Non-integer step_size raises."""
        with pytest.raises(ValueError, match="positive integer"):
            StepwiseTemporalProcess(AR1(autoreg=0.9, innovation_sd=0.05), step_size=7.0)

    def test_repr_includes_inner_and_step_size(self):
        """__repr__ shows the inner process and step_size."""
        inner = AR1(autoreg=0.9, innovation_sd=0.05)
        wrapper = StepwiseTemporalProcess(inner, step_size=7)
        rendered = repr(wrapper)
        assert rendered.startswith("StepwiseTemporalProcess(")
        assert f"inner={inner!r}" in rendered
        assert "step_size=7" in rendered


class TestStepwiseTemporalProcessSample:
    """Sample-time behavior of StepwiseTemporalProcess."""

    @pytest.mark.parametrize("inner_cls,inner_kwargs", INNER_PROCESS_PARAMS)
    def test_output_shape_divisible(self, inner_cls, inner_kwargs):
        """n_timepoints divisible by step_size yields (n_timepoints, n_processes)."""
        wrapper = StepwiseTemporalProcess(inner_cls(**inner_kwargs), step_size=7)
        with numpyro.handlers.seed(rng_seed=42):
            result = wrapper.sample(n_timepoints=28, n_processes=3)
        assert result.shape == (28, 3)

    @pytest.mark.parametrize("inner_cls,inner_kwargs", INNER_PROCESS_PARAMS)
    def test_output_shape_non_divisible(self, inner_cls, inner_kwargs):
        """n_timepoints not divisible by step_size still yields n_timepoints rows."""
        wrapper = StepwiseTemporalProcess(inner_cls(**inner_kwargs), step_size=7)
        with numpyro.handlers.seed(rng_seed=42):
            result = wrapper.sample(n_timepoints=30, n_processes=2)
        assert result.shape == (30, 2)

    @pytest.mark.parametrize("inner_cls,inner_kwargs", INNER_PROCESS_PARAMS)
    def test_broadcast_within_block(self, inner_cls, inner_kwargs):
        """Each block of step_size consecutive timepoints is constant."""
        wrapper = StepwiseTemporalProcess(inner_cls(**inner_kwargs), step_size=7)
        with numpyro.handlers.seed(rng_seed=42):
            result = wrapper.sample(n_timepoints=28, n_processes=1)
        for start in range(0, 28, 7):
            block = result[start : start + 7]
            assert jnp.allclose(block, block[0])

    @pytest.mark.parametrize("inner_cls,inner_kwargs", INNER_PROCESS_PARAMS)
    def test_broadcast_with_partial_final_block(self, inner_cls, inner_kwargs):
        """A partial final block is still constant, just shorter than step_size."""
        wrapper = StepwiseTemporalProcess(inner_cls(**inner_kwargs), step_size=7)
        with numpyro.handlers.seed(rng_seed=42):
            result = wrapper.sample(n_timepoints=30, n_processes=1)
        # The 4th block starts at row 28 and runs through row 29 (2 rows)
        final_block = result[28:30]
        assert jnp.allclose(final_block, final_block[0])

    def test_step_size_one_passthrough_shape(self):
        """step_size=1 yields (n_timepoints, n_processes), same as inner directly."""
        wrapper = StepwiseTemporalProcess(
            AR1(autoreg=0.9, innovation_sd=0.05), step_size=1
        )
        with numpyro.handlers.seed(rng_seed=42):
            result = wrapper.sample(n_timepoints=20, n_processes=2)
        assert result.shape == (20, 2)

    def test_coarse_trajectory_is_recorded(self):
        """StepwiseTemporalProcess records the coarse trajectory."""
        wrapper = StepwiseTemporalProcess(
            AR1(autoreg=0.9, innovation_sd=0.05), step_size=7
        )
        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(wrapper.sample, rng_seed=42)
        ).get_trace(n_timepoints=15, n_processes=1, name_prefix="rt")

        assert "rt_coarse" in traced
        assert traced["rt_coarse"]["type"] == "deterministic"
        assert traced["rt_coarse"]["value"].shape == (3, 1)


class TestWeeklyTemporalProcessConstruction:
    """Construction-time validation for WeeklyTemporalProcess."""

    def test_step_size_attribute(self):
        """step_size is exposed on the instance for builder inspection."""
        wrapper = WeeklyTemporalProcess(
            AR1(autoreg=0.9, innovation_sd=0.05), week=MMWR_WEEK
        )
        assert wrapper.step_size == 7

    def test_requires_calendar_anchor_attribute(self):
        """WeeklyTemporalProcess reports that it needs a calendar anchor."""
        wrapper = WeeklyTemporalProcess(
            AR1(autoreg=0.9, innovation_sd=0.05), week=MMWR_WEEK
        )
        assert wrapper.requires_calendar_anchor is True

    def test_requires_week(self):
        """WeeklyTemporalProcess requires a WeekCycle."""
        with pytest.raises(ValueError, match="week is required"):
            WeeklyTemporalProcess(AR1(autoreg=0.9, innovation_sd=0.05), week=None)

    def test_repr_includes_inner_and_week(self):
        """__repr__ shows the inner process and week."""
        inner = AR1(autoreg=0.9, innovation_sd=0.05)
        wrapper = WeeklyTemporalProcess(inner, week=MMWR_WEEK)
        rendered = repr(wrapper)
        assert rendered.startswith("WeeklyTemporalProcess(")
        assert f"inner={inner!r}" in rendered
        assert f"week={MMWR_WEEK!r}" in rendered


class TestWeeklyTemporalProcessSample:
    """Sample-time behavior of WeeklyTemporalProcess."""

    def test_requires_first_day_dow_at_sample_time(self):
        """WeeklyTemporalProcess needs the model-axis day of week."""
        wrapper = WeeklyTemporalProcess(
            AR1(autoreg=0.9, innovation_sd=0.05), week=MMWR_WEEK
        )
        with numpyro.handlers.seed(rng_seed=42):
            with pytest.raises(ValueError, match="first_day_dow"):
                wrapper.sample(n_timepoints=20, n_processes=1)

    def test_alignment_with_leading_partial_week(self):
        """WeeklyTemporalProcess starts full blocks on week.start_dow."""
        wrapper = WeeklyTemporalProcess(
            AR1(autoreg=0.9, innovation_sd=0.05), week=MMWR_WEEK
        )
        # first_day_dow=3 means day 0 is Thursday. With Sunday week starts,
        # days 0-2 are a leading partial week, then days 3-9 are the first
        # full Sunday-Saturday block on the model axis.
        with numpyro.handlers.seed(rng_seed=42):
            result = wrapper.sample(
                n_timepoints=17,
                n_processes=1,
                first_day_dow=3,
            )

        assert jnp.allclose(result[:3], result[0])
        assert jnp.allclose(result[3:10], result[3])
        assert jnp.allclose(result[10:17], result[10])
        assert not jnp.allclose(result[2], result[3])

    def test_alignment_without_leading_partial_week(self):
        """WeeklyTemporalProcess handles model axes starting on week.start_dow."""
        wrapper = WeeklyTemporalProcess(
            AR1(autoreg=0.9, innovation_sd=0.05), week=MMWR_WEEK
        )
        with numpyro.handlers.seed(rng_seed=42):
            result = wrapper.sample(
                n_timepoints=15,
                n_processes=1,
                first_day_dow=6,
            )

        assert jnp.allclose(result[:7], result[0])
        assert jnp.allclose(result[7:14], result[7])
        assert jnp.allclose(result[14:15], result[14])

    def test_weekly_trajectory_is_recorded(self):
        """WeeklyTemporalProcess records the weekly trajectory."""
        wrapper = WeeklyTemporalProcess(
            AR1(autoreg=0.9, innovation_sd=0.05), week=MMWR_WEEK
        )
        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(wrapper.sample, rng_seed=42)
        ).get_trace(
            n_timepoints=15,
            n_processes=1,
            name_prefix="rt",
            first_day_dow=6,
        )

        assert "rt_weekly" in traced
        assert traced["rt_weekly"]["type"] == "deterministic"
        assert traced["rt_weekly"]["value"].shape == (3, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
