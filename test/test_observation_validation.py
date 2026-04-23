"""
Unit tests for observation data validation functions.

Tests the refactored validation helpers on BaseObservationProcess
and the validate_data() methods on PopulationCounts, SubpopulationCounts, and
MeasurementObservation.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.observation import (
    HierarchicalNormalNoise,
    MeasurementObservation,
    PoissonNoise,
    PopulationCounts,
    SubpopulationCounts,
)
from pyrenew.randomvariable import DistributionalVariable, VectorizedVariable
from pyrenew.time import MMWR_WEEK, WeekCycle, daily_to_weekly

# ---------------------------------------------------------------------------
# Helpers – minimal concrete subclass of MeasurementObservation for testing
# ---------------------------------------------------------------------------


class StubMeasurementObservation(MeasurementObservation):
    """Minimal concrete MeasurementObservation for testing validate_data()."""

    def validate(self) -> None:
        """
        Validate parameters.

        Raises
        ------
        ValueError
            If PMF is invalid.
        """
        pmf = self.temporal_pmf_rv()
        self._validate_pmf(pmf, "temporal_pmf_rv")

    def _predicted_obs(self, infections):
        """
        Return infections unchanged (identity transform).

        Parameters
        ----------
        infections : ArrayLike
            Input infections.

        Returns
        -------
        ArrayLike
            Same as input.
        """
        return infections


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def counts_proc():
    """
    PopulationCounts process used to access base validation helpers.

    Returns
    -------
    PopulationCounts
        A PopulationCounts observation process.
    """
    return PopulationCounts(
        name="hosp",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", jnp.array([0.3, 0.5, 0.2])),
        noise=PoissonNoise(),
    )


@pytest.fixture()
def subpop_proc():
    """
    SubpopulationCounts process for validate_data() tests.

    Returns
    -------
    SubpopulationCounts
        A SubpopulationCounts observation process.
    """
    return SubpopulationCounts(
        name="subpop_hosp",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.02),
        delay_distribution_rv=DeterministicPMF("delay", jnp.array([0.4, 0.4, 0.2])),
        noise=PoissonNoise(),
        reporting_schedule="irregular",
    )


@pytest.fixture()
def measurements_proc():
    """
    MeasurementObservation process for validate_data() tests.

    Returns
    -------
    StubMeasurementObservation
        A StubMeasurementObservation observation process.
    """
    sensor_mode_rv = VectorizedVariable(
        name="sensor_mode_rv",
        rv=DistributionalVariable("mode", dist.Normal(0, 0.5)),
    )
    sensor_sd_rv = VectorizedVariable(
        name="sensor_sd_rv",
        rv=DistributionalVariable("sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)),
    )
    noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)
    return StubMeasurementObservation(
        name="ww",
        temporal_pmf_rv=DeterministicPMF("shedding", jnp.array([0.3, 0.4, 0.3])),
        noise=noise,
    )


# ===================================================================
# _validate_index_array
# ===================================================================


class TestValidateIndexArray:
    """Tests for BaseObservationProcess._validate_index_array."""

    def test_valid_indices(self, counts_proc):
        """Valid indices within bounds should not raise."""
        counts_proc._validate_index_array(
            jnp.array([0, 1, 2, 3]), upper_bound=5, param_name="test_idx"
        )

    def test_single_valid_index(self, counts_proc):
        """A single valid index should not raise."""
        counts_proc._validate_index_array(
            jnp.array([0]), upper_bound=1, param_name="test_idx"
        )

    def test_negative_index_raises(self, counts_proc):
        """Negative indices should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            counts_proc._validate_index_array(
                jnp.array([0, -1, 2]),
                upper_bound=5,
                param_name="my_param",
            )

    def test_index_at_upper_bound_raises(self, counts_proc):
        """Index exactly equal to upper_bound should raise."""
        with pytest.raises(ValueError, match="upper bound"):
            counts_proc._validate_index_array(
                jnp.array([0, 1, 5]),
                upper_bound=5,
                param_name="my_param",
            )

    def test_index_above_upper_bound_raises(self, counts_proc):
        """Index above upper_bound should raise."""
        with pytest.raises(ValueError, match="upper bound"):
            counts_proc._validate_index_array(
                jnp.array([10]),
                upper_bound=5,
                param_name="my_param",
            )

    def test_error_message_includes_name_and_value(self, counts_proc):
        """Error message should include the observation name and param name."""
        with pytest.raises(ValueError, match="hosp.*my_param"):
            counts_proc._validate_index_array(
                jnp.array([-1]),
                upper_bound=5,
                param_name="my_param",
            )

    def test_non_contiguous_valid_indices(self, counts_proc):
        """Non-contiguous but valid indices should not raise."""
        counts_proc._validate_index_array(
            jnp.array([0, 3, 7, 9]),
            upper_bound=10,
            param_name="test_idx",
        )

    def test_empty_array_passes(self, counts_proc):
        """Empty index array has no values to bounds-check; should not raise."""
        counts_proc._validate_index_array(
            jnp.array([], dtype=jnp.int32),
            upper_bound=10,
            param_name="test_idx",
        )

    def test_scalar_raises(self, counts_proc):
        """0-D scalar index array should raise ValueError."""
        with pytest.raises(ValueError, match="must be 1D"):
            counts_proc._validate_index_array(
                jnp.asarray(0),
                upper_bound=5,
                param_name="test_idx",
            )

    def test_2d_raises(self, counts_proc):
        """2D index array should raise ValueError."""
        with pytest.raises(ValueError, match="must be 1D"):
            counts_proc._validate_index_array(
                jnp.array([[0, 1], [2, 3]]),
                upper_bound=5,
                param_name="test_idx",
            )


# ===================================================================
# _validate_times
# ===================================================================


class TestValidateTimes:
    """Tests for BaseObservationProcess._validate_times."""

    def test_valid_times(self, counts_proc):
        """Valid time indices should not raise."""
        counts_proc._validate_times(jnp.array([0, 5, 19]), n_total=20)

    def test_negative_time_raises(self, counts_proc):
        """Negative time index should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            counts_proc._validate_times(jnp.array([0, -1]), n_total=20)

    def test_time_at_n_total_raises(self, counts_proc):
        """Time index equal to n_total should raise (0-indexed)."""
        with pytest.raises(ValueError, match="upper bound"):
            counts_proc._validate_times(jnp.array([0, 20]), n_total=20)

    def test_time_above_n_total_raises(self, counts_proc):
        """Time index above n_total should raise."""
        with pytest.raises(ValueError, match="upper bound"):
            counts_proc._validate_times(jnp.array([50]), n_total=20)


# ===================================================================
# _validate_subpop_indices
# ===================================================================


class TestValidateSubpopIndices:
    """Tests for BaseObservationProcess._validate_subpop_indices."""

    def test_valid_subpop_indices(self, counts_proc):
        """Valid subpop indices should not raise."""
        counts_proc._validate_subpop_indices(jnp.array([0, 1, 2]), n_subpops=3)

    def test_non_contiguous_subpop_indices(self, counts_proc):
        """Non-contiguous but valid subpop indices should not raise."""
        counts_proc._validate_subpop_indices(jnp.array([0, 2, 4]), n_subpops=5)

    def test_negative_subpop_index_raises(self, counts_proc):
        """Negative subpop index should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            counts_proc._validate_subpop_indices(jnp.array([0, -1]), n_subpops=3)

    def test_subpop_index_at_n_subpops_raises(self, counts_proc):
        """Subpop index equal to n_subpops should raise."""
        with pytest.raises(ValueError, match="upper bound"):
            counts_proc._validate_subpop_indices(jnp.array([0, 3]), n_subpops=3)

    def test_subpop_index_above_n_subpops_raises(self, counts_proc):
        """Subpop index well above n_subpops should raise."""
        with pytest.raises(ValueError, match="upper bound"):
            counts_proc._validate_subpop_indices(jnp.array([10]), n_subpops=3)


# ===================================================================
# _validate_shapes_match
# ===================================================================


class TestValidateShapesMatch:
    """Tests for BaseObservationProcess._validate_shapes_match."""

    def test_matching_1d_shapes(self, counts_proc):
        """Two arrays with matching 1D shapes should not raise."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([0, 5, 10])
        counts_proc._validate_shapes_match(a, b, "obs", "times")

    def test_mismatched_lengths_raises(self, counts_proc):
        """Two arrays with different lengths should raise ValueError."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([0, 5])
        with pytest.raises(ValueError, match="must match times shape"):
            counts_proc._validate_shapes_match(a, b, "obs", "times")

    def test_empty_arrays_match(self, counts_proc):
        """Two empty arrays should not raise."""
        a = jnp.array([])
        b = jnp.array([])
        counts_proc._validate_shapes_match(a, b, "obs", "times")

    def test_scalar_arrays_match(self, counts_proc):
        """Single-element arrays should not raise."""
        a = jnp.array([5.0])
        b = jnp.array([0])
        counts_proc._validate_shapes_match(a, b, "obs", "times")

    def test_error_includes_both_shapes(self, counts_proc):
        """Error message should report both shapes."""
        a = jnp.array([1.0, 2.0])
        b = jnp.array([0, 1, 2])
        with pytest.raises(ValueError, match=r"\(2,\).*\(3,\)"):
            counts_proc._validate_shapes_match(a, b, "obs", "times")

    def test_error_uses_provided_names(self, counts_proc):
        """Error message should use the caller-supplied parameter names."""
        a = jnp.array([1.0, 2.0])
        b = jnp.array([0, 1, 2])
        with pytest.raises(
            ValueError,
            match=r"subpop_indices shape .* must match period_end_times shape",
        ):
            counts_proc._validate_shapes_match(
                a, b, "subpop_indices", "period_end_times"
            )


# ===================================================================
# _validate_obs_dense
# ===================================================================


class TestValidateObsDense:
    """Tests for BaseObservationProcess._validate_obs_dense."""

    def test_correct_length(self, counts_proc):
        """Obs with length equal to n_total should not raise."""
        obs = jnp.ones(30)
        counts_proc._validate_obs_dense(obs, n_total=30)

    def test_obs_with_nan_correct_length(self, counts_proc):
        """Obs with NaN padding but correct length should not raise."""
        obs = jnp.ones(30).at[:5].set(jnp.nan)
        counts_proc._validate_obs_dense(obs, n_total=30)

    def test_obs_too_short_raises(self, counts_proc):
        """Obs shorter than n_total should raise ValueError."""
        obs = jnp.ones(20)
        with pytest.raises(ValueError, match="must equal n_total"):
            counts_proc._validate_obs_dense(obs, n_total=30)

    def test_obs_too_long_raises(self, counts_proc):
        """Obs longer than n_total should raise ValueError."""
        obs = jnp.ones(40)
        with pytest.raises(ValueError, match="must equal n_total"):
            counts_proc._validate_obs_dense(obs, n_total=30)

    def test_error_includes_lengths(self, counts_proc):
        """Error message should include actual and expected lengths."""
        obs = jnp.ones(15)
        with pytest.raises(ValueError, match="15.*30"):
            counts_proc._validate_obs_dense(obs, n_total=30)

    def test_2d_obs_raises(self, counts_proc):
        """2D obs (e.g., shape (n_total, 1)) should raise ValueError."""
        obs = jnp.ones((30, 1))
        with pytest.raises(ValueError, match="obs must be 1D"):
            counts_proc._validate_obs_dense(obs, n_total=30)


# ===================================================================
# PopulationCounts.validate_data()
# ===================================================================


class TestPopulationCountsValidateData:
    """Tests for PopulationCounts.validate_data()."""

    def test_none_obs_passes(self, counts_proc):
        """validate_data with obs=None should not raise."""
        counts_proc.validate_data(n_total=30, n_subpops=1, obs=None)

    def test_correct_obs_passes(self, counts_proc):
        """validate_data with correctly shaped obs should not raise."""
        obs = jnp.ones(30) * 5.0
        counts_proc.validate_data(n_total=30, n_subpops=1, obs=obs)

    def test_nan_padded_obs_passes(self, counts_proc):
        """validate_data with NaN-padded obs of correct length should not raise."""
        obs = jnp.ones(30).at[:3].set(jnp.nan)
        counts_proc.validate_data(n_total=30, n_subpops=1, obs=obs)

    def test_wrong_length_obs_raises(self, counts_proc):
        """validate_data with obs of wrong length should raise ValueError."""
        obs = jnp.ones(20)
        with pytest.raises(ValueError, match="must equal n_total"):
            counts_proc.validate_data(n_total=30, n_subpops=1, obs=obs)

    def test_2d_obs_daily_raises(self, counts_proc):
        """Daily regular-schedule obs must be 1D; 2D (n_total, 1) should raise."""
        obs = jnp.ones((30, 1))
        with pytest.raises(ValueError, match="obs must be 1D"):
            counts_proc.validate_data(n_total=30, n_subpops=1, obs=obs)

    def test_2d_obs_weekly_raises(self):
        """Weekly regular-schedule obs must be 1D; 2D (n_periods, 1) should raise."""
        proc = PopulationCounts(
            name="hosp",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF("delay", jnp.array([0.3, 0.5, 0.2])),
            noise=PoissonNoise(),
            aggregation="weekly",
            reporting_schedule="regular",
            week=MMWR_WEEK,
        )
        n_total = 35
        first_day_dow = 6
        offset = proc._compute_period_offset(first_day_dow, proc.week)
        n_periods = (n_total - offset) // proc.aggregation_period
        obs = jnp.ones((n_periods, 1))
        with pytest.raises(ValueError, match="obs must be 1D"):
            proc.validate_data(
                n_total=n_total,
                n_subpops=1,
                obs=obs,
                first_day_dow=first_day_dow,
            )

    def test_extra_kwargs_ignored(self, counts_proc):
        """validate_data should ignore extra keyword arguments."""
        counts_proc.validate_data(
            n_total=30, n_subpops=1, obs=None, extra_param="ignored"
        )


# ===================================================================
# SubpopulationCounts.validate_data()
# ===================================================================


class TestSubpopulationCountsValidateData:
    """Tests for SubpopulationCounts.validate_data()."""

    def test_all_none_passes(self, subpop_proc):
        """validate_data with all optional args None should not raise."""
        subpop_proc.validate_data(n_total=30, n_subpops=3)

    def test_valid_data_passes(self, subpop_proc):
        """validate_data with valid period_end_times, subpop_indices, obs should not raise."""
        period_end_times = jnp.array([5, 10, 15, 20])
        subpop_indices = jnp.array([0, 1, 2, 0])
        obs = jnp.array([10.0, 20.0, 30.0, 15.0])
        subpop_proc.validate_data(
            n_total=30,
            n_subpops=3,
            period_end_times=period_end_times,
            subpop_indices=subpop_indices,
            obs=obs,
        )

    def test_invalid_times_raises(self, subpop_proc):
        """validate_data with out-of-bounds period_end_times should raise."""
        period_end_times = jnp.array([5, 30])  # 30 == n_total, out of bounds
        with pytest.raises(ValueError, match="upper bound"):
            subpop_proc.validate_data(
                n_total=30, n_subpops=3, period_end_times=period_end_times
            )

    def test_negative_times_raises(self, subpop_proc):
        """validate_data with negative period_end_times should raise."""
        period_end_times = jnp.array([-1, 5])
        with pytest.raises(ValueError, match="cannot be negative"):
            subpop_proc.validate_data(
                n_total=30, n_subpops=3, period_end_times=period_end_times
            )

    def test_invalid_subpop_indices_raises(self, subpop_proc):
        """validate_data with out-of-bounds subpop_indices should raise."""
        subpop_indices = jnp.array([0, 3])  # 3 == n_subpops, out of bounds
        with pytest.raises(ValueError, match="upper bound"):
            subpop_proc.validate_data(
                n_total=30, n_subpops=3, subpop_indices=subpop_indices
            )

    def test_negative_subpop_indices_raises(self, subpop_proc):
        """validate_data with negative subpop_indices should raise."""
        subpop_indices = jnp.array([0, -1])
        with pytest.raises(ValueError, match="cannot be negative"):
            subpop_proc.validate_data(
                n_total=30, n_subpops=3, subpop_indices=subpop_indices
            )

    def test_mismatched_obs_times_raises(self, subpop_proc):
        """validate_data with obs/period_end_times shape mismatch should raise."""
        period_end_times = jnp.array([5, 10, 15])
        obs = jnp.array([1.0, 2.0])  # length 2 != length 3
        with pytest.raises(
            ValueError, match=r"obs shape .* must match period_end_times shape"
        ):
            subpop_proc.validate_data(
                n_total=30,
                n_subpops=3,
                period_end_times=period_end_times,
                obs=obs,
            )

    def test_obs_without_times_skips_shape_check(self, subpop_proc):
        """validate_data with obs but no period_end_times should not check shapes."""
        obs = jnp.array([1.0, 2.0])
        subpop_proc.validate_data(n_total=30, n_subpops=3, obs=obs)

    def test_times_without_obs_skips_shape_check(self, subpop_proc):
        """validate_data with period_end_times but no obs should validate indices only."""
        period_end_times = jnp.array([5, 10, 15])
        subpop_proc.validate_data(
            n_total=30, n_subpops=3, period_end_times=period_end_times
        )

    def test_non_contiguous_subpop_indices_valid(self, subpop_proc):
        """validate_data with non-contiguous but valid subpop_indices passes."""
        subpop_indices = jnp.array([0, 2])  # skip index 1
        subpop_proc.validate_data(
            n_total=30, n_subpops=3, subpop_indices=subpop_indices
        )

    def test_scalar_subpop_indices_raises(self, subpop_proc):
        """Scalar (0-D) subpop_indices should raise ValueError, not IndexError."""
        with pytest.raises(ValueError, match="must be 1D"):
            subpop_proc.validate_data(
                n_total=30, n_subpops=3, subpop_indices=jnp.asarray(0)
            )


# ===================================================================
# MeasurementObservation.validate_data()
# ===================================================================


class TestMeasurementObservationValidateData:
    """Tests for MeasurementObservation.validate_data()."""

    def test_all_none_passes(self, measurements_proc):
        """validate_data with all optional args None should not raise."""
        measurements_proc.validate_data(n_total=30, n_subpops=3)

    def test_valid_data_passes(self, measurements_proc):
        """validate_data with fully valid data should not raise."""
        times = jnp.array([5, 10, 15, 20])
        subpop_indices = jnp.array([0, 1, 2, 0])
        sensor_indices = jnp.array([0, 1, 0, 1])
        obs = jnp.array([1.1, 2.2, 3.3, 1.5])
        measurements_proc.validate_data(
            n_total=30,
            n_subpops=3,
            times=times,
            subpop_indices=subpop_indices,
            sensor_indices=sensor_indices,
            n_sensors=2,
            obs=obs,
        )

    def test_invalid_times_raises(self, measurements_proc):
        """validate_data with out-of-bounds times should raise."""
        times = jnp.array([5, 30])
        with pytest.raises(ValueError, match="upper bound"):
            measurements_proc.validate_data(n_total=30, n_subpops=3, times=times)

    def test_negative_times_raises(self, measurements_proc):
        """validate_data with negative times should raise."""
        times = jnp.array([-1, 5])
        with pytest.raises(ValueError, match="cannot be negative"):
            measurements_proc.validate_data(n_total=30, n_subpops=3, times=times)

    def test_invalid_subpop_indices_raises(self, measurements_proc):
        """validate_data with out-of-bounds subpop_indices should raise."""
        subpop_indices = jnp.array([0, 5])
        with pytest.raises(ValueError, match="upper bound"):
            measurements_proc.validate_data(
                n_total=30, n_subpops=3, subpop_indices=subpop_indices
            )

    def test_invalid_sensor_indices_raises(self, measurements_proc):
        """validate_data with out-of-bounds sensor_indices should raise."""
        sensor_indices = jnp.array([0, 4])
        with pytest.raises(ValueError, match="upper bound"):
            measurements_proc.validate_data(
                n_total=30,
                n_subpops=3,
                sensor_indices=sensor_indices,
                n_sensors=3,
            )

    def test_negative_sensor_indices_raises(self, measurements_proc):
        """validate_data with negative sensor_indices should raise."""
        sensor_indices = jnp.array([-1, 0])
        with pytest.raises(ValueError, match="cannot be negative"):
            measurements_proc.validate_data(
                n_total=30,
                n_subpops=3,
                sensor_indices=sensor_indices,
                n_sensors=3,
            )

    def test_sensor_indices_without_n_sensors_skips(self, measurements_proc):
        """validate_data with sensor_indices but no n_sensors skips check."""
        sensor_indices = jnp.array([0, 99])  # would be invalid with n_sensors
        # n_sensors is None so sensor_indices validation is skipped
        measurements_proc.validate_data(
            n_total=30,
            n_subpops=3,
            sensor_indices=sensor_indices,
            n_sensors=None,
        )

    def test_n_sensors_without_sensor_indices_skips(self, measurements_proc):
        """validate_data with n_sensors but no sensor_indices skips check."""
        measurements_proc.validate_data(
            n_total=30,
            n_subpops=3,
            sensor_indices=None,
            n_sensors=5,
        )

    def test_mismatched_obs_times_raises(self, measurements_proc):
        """validate_data with obs/times shape mismatch should raise."""
        times = jnp.array([5, 10, 15])
        obs = jnp.array([1.0, 2.0])
        with pytest.raises(ValueError, match=r"obs shape .* must match times shape"):
            measurements_proc.validate_data(
                n_total=30, n_subpops=3, times=times, obs=obs
            )

    def test_matching_obs_times_passes(self, measurements_proc):
        """validate_data with matching obs/times shapes should not raise."""
        times = jnp.array([5, 10])
        obs = jnp.array([1.0, 2.0])
        measurements_proc.validate_data(n_total=30, n_subpops=3, times=times, obs=obs)

    def test_non_contiguous_subpop_indices_valid(self, measurements_proc):
        """validate_data with non-contiguous but valid subpop_indices passes."""
        subpop_indices = jnp.array([0, 2])
        measurements_proc.validate_data(
            n_total=30, n_subpops=3, subpop_indices=subpop_indices
        )

    def test_extra_kwargs_ignored(self, measurements_proc):
        """validate_data should ignore extra keyword arguments."""
        measurements_proc.validate_data(n_total=30, n_subpops=3, foo="bar")


# ===================================================================
# _validate_week
# ===================================================================


class TestValidateWeek:
    """Tests for BaseObservationProcess._validate_week."""

    def test_daily_with_no_week_passes(self, counts_proc):
        """aggregation='daily' with week=None should not raise."""
        counts_proc._validate_week("daily", None)

    def test_daily_ignores_week(self, counts_proc):
        """aggregation='daily' ignores any supplied WeekCycle."""
        counts_proc._validate_week("daily", MMWR_WEEK)

    def test_weekly_with_week_passes(self, counts_proc):
        """aggregation='weekly' with any WeekCycle should not raise."""
        counts_proc._validate_week("weekly", MMWR_WEEK)
        counts_proc._validate_week("weekly", WeekCycle(start_dow=0))

    def test_weekly_missing_week_raises(self, counts_proc):
        """aggregation='weekly' with week=None should raise."""
        with pytest.raises(ValueError, match="week is required"):
            counts_proc._validate_week("weekly", None)

    @pytest.mark.parametrize("bad", ["monthly", "biweekly", "Weekly", ""])
    def test_unknown_aggregation_raises(self, counts_proc, bad):
        """Unrecognized aggregation strings should raise."""
        with pytest.raises(ValueError, match="aggregation must be one of"):
            counts_proc._validate_week(bad, MMWR_WEEK)


# ===================================================================
# _compute_period_offset
# ===================================================================


class TestComputePeriodOffset:
    """Tests for BaseObservationProcess._compute_period_offset."""

    def test_daily_returns_zero(self, counts_proc):
        """week=None always returns 0 regardless of first_day_dow."""
        assert counts_proc._compute_period_offset(None, None) == 0
        assert counts_proc._compute_period_offset(0, None) == 0
        assert counts_proc._compute_period_offset(6, None) == 0

    def test_mmwr_aligned_start(self, counts_proc):
        """Daily axis starting Sunday under MMWR_WEEK => offset 0."""
        assert counts_proc._compute_period_offset(6, MMWR_WEEK) == 0

    def test_monday_start_mmwr(self, counts_proc):
        """Daily axis starting Monday under MMWR_WEEK => offset 6."""
        assert counts_proc._compute_period_offset(0, MMWR_WEEK) == 6

    def test_saturday_start_mmwr(self, counts_proc):
        """Daily axis starting Saturday under MMWR_WEEK => offset 1."""
        assert counts_proc._compute_period_offset(5, MMWR_WEEK) == 1

    def test_iso_week_alignment(self, counts_proc):
        """Daily axis starting Thursday under ISO week (Mon start) => offset 4."""
        assert counts_proc._compute_period_offset(3, WeekCycle(start_dow=0)) == 4

    def test_offset_always_in_range(self, counts_proc):
        """Offset is always in [0, 7) for any valid (first_day_dow, week)."""
        for first in range(7):
            for start in range(7):
                offset = counts_proc._compute_period_offset(
                    first, WeekCycle(start_dow=start)
                )
                assert 0 <= offset < 7

    def test_missing_first_day_dow_raises(self, counts_proc):
        """week set with first_day_dow=None should raise."""
        with pytest.raises(ValueError, match="first_day_dow is required"):
            counts_proc._compute_period_offset(None, MMWR_WEEK)

    def test_offset_agrees_with_daily_to_weekly(self, counts_proc):
        """
        Offset from _compute_period_offset selects the same leading
        days that daily_to_weekly trims internally for the first
        complete period, for every (first_day_dow, week) pair.
        """
        daily = jnp.arange(21.0)
        for first in range(7):
            for start in range(7):
                week = WeekCycle(start_dow=start)
                offset = counts_proc._compute_period_offset(first, week)
                weekly = daily_to_weekly(
                    daily,
                    input_data_first_dow=first,
                    week_start_dow=start,
                )
                expected_first_week = float(jnp.sum(daily[offset : offset + 7]))
                assert float(weekly[0]) == expected_first_week


# ===================================================================
# _validate_period_end_times
# ===================================================================


class TestValidatePeriodEndTimes:
    """Tests for BaseObservationProcess._validate_period_end_times."""

    def test_p7_aligned_times_pass(self, counts_proc):
        """P=7 with Saturdays at offset 0 should not raise."""
        times = jnp.array([6, 13, 20])
        counts_proc._validate_period_end_times(
            times, n_total=21, offset=0, aggregation_period=7
        )

    def test_p7_aligned_nonzero_offset_pass(self, counts_proc):
        """P=7 with nonzero offset shifts the boundary days."""
        times = jnp.array([12, 19, 26])
        counts_proc._validate_period_end_times(
            times, n_total=30, offset=6, aggregation_period=7
        )

    def test_p1_any_in_bounds_passes(self, counts_proc):
        """P=1: alignment is trivial; any in-bounds index passes."""
        times = jnp.array([0, 3, 7, 19])
        counts_proc._validate_period_end_times(
            times, n_total=20, offset=0, aggregation_period=1
        )

    def test_misaligned_time_raises(self, counts_proc):
        """P=7 with a non-boundary time should raise."""
        times = jnp.array([5])
        with pytest.raises(ValueError, match="period_end_times must lie on"):
            counts_proc._validate_period_end_times(
                times, n_total=21, offset=0, aggregation_period=7
            )

    def test_partial_misalignment_raises(self, counts_proc):
        """Any single misaligned entry should raise."""
        times = jnp.array([6, 12, 20])
        with pytest.raises(ValueError, match="period_end_times must lie on"):
            counts_proc._validate_period_end_times(
                times, n_total=21, offset=0, aggregation_period=7
            )

    def test_negative_time_raises(self, counts_proc):
        """Negative period_end_times should raise via bounds check."""
        times = jnp.array([-1, 6])
        with pytest.raises(ValueError, match="cannot be negative"):
            counts_proc._validate_period_end_times(
                times, n_total=21, offset=0, aggregation_period=7
            )

    def test_time_at_n_total_raises(self, counts_proc):
        """period_end_times == n_total should raise via bounds check."""
        times = jnp.array([21])
        with pytest.raises(ValueError, match="upper bound"):
            counts_proc._validate_period_end_times(
                times, n_total=21, offset=0, aggregation_period=7
            )

    def test_error_reports_offset_and_period(self, counts_proc):
        """Alignment error message should include offset and aggregation_period."""
        times = jnp.array([5])
        with pytest.raises(ValueError, match=r"offset=0.*aggregation_period=7"):
            counts_proc._validate_period_end_times(
                times, n_total=21, offset=0, aggregation_period=7
            )
