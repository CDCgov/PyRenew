"""
Unit tests for observation data validation functions.

Tests the refactored validation helpers on BaseObservationProcess
and the validate_data() methods on Counts, CountsBySubpop, and
Measurements.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.observation import (
    Counts,
    CountsBySubpop,
    HierarchicalNormalNoise,
    PoissonNoise,
    VectorizedRV,
)
from pyrenew.observation.measurements import Measurements
from pyrenew.randomvariable import DistributionalVariable

# ---------------------------------------------------------------------------
# Helpers – minimal concrete subclass of Measurements for testing
# ---------------------------------------------------------------------------


class StubMeasurements(Measurements):
    """Minimal concrete Measurements for testing validate_data()."""

    def validate(self) -> None:  # noqa: D102
        pmf = self.temporal_pmf_rv()
        self._validate_pmf(pmf, "temporal_pmf_rv")

    def _predicted_obs(self, infections):  # noqa: D102
        return infections


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def counts_proc():
    """Counts process used to access base validation helpers."""
    return Counts(
        name="hosp",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", jnp.array([0.3, 0.5, 0.2])),
        noise=PoissonNoise(),
    )


@pytest.fixture()
def subpop_proc():
    """CountsBySubpop process for validate_data() tests."""
    return CountsBySubpop(
        name="subpop_hosp",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.02),
        delay_distribution_rv=DeterministicPMF("delay", jnp.array([0.4, 0.4, 0.2])),
        noise=PoissonNoise(),
    )


@pytest.fixture()
def measurements_proc():
    """Measurements process for validate_data() tests."""
    sensor_mode_rv = VectorizedRV(
        DistributionalVariable("mode", dist.Normal(0, 0.5)),
        plate_name="sensor_mode",
    )
    sensor_sd_rv = VectorizedRV(
        DistributionalVariable("sd", dist.TruncatedNormal(0.3, 0.15, low=0.05)),
        plate_name="sensor_sd",
    )
    noise = HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)
    return StubMeasurements(
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
# _validate_obs_times_shape
# ===================================================================


class TestValidateObsTimesShape:
    """Tests for BaseObservationProcess._validate_obs_times_shape."""

    def test_matching_1d_shapes(self, counts_proc):
        """Obs and times with matching 1D shapes should not raise."""
        obs = jnp.array([1.0, 2.0, 3.0])
        times = jnp.array([0, 5, 10])
        counts_proc._validate_obs_times_shape(obs, times)

    def test_mismatched_lengths_raises(self, counts_proc):
        """Obs and times with different lengths should raise ValueError."""
        obs = jnp.array([1.0, 2.0, 3.0])
        times = jnp.array([0, 5])
        with pytest.raises(ValueError, match="must match times shape"):
            counts_proc._validate_obs_times_shape(obs, times)

    def test_empty_arrays_match(self, counts_proc):
        """Two empty arrays should not raise."""
        obs = jnp.array([])
        times = jnp.array([])
        counts_proc._validate_obs_times_shape(obs, times)

    def test_scalar_arrays_match(self, counts_proc):
        """Single-element arrays should not raise."""
        obs = jnp.array([5.0])
        times = jnp.array([0])
        counts_proc._validate_obs_times_shape(obs, times)

    def test_error_includes_both_shapes(self, counts_proc):
        """Error message should report both shapes."""
        obs = jnp.array([1.0, 2.0])
        times = jnp.array([0, 1, 2])
        with pytest.raises(ValueError, match=r"\(2,\).*\(3,\)"):
            counts_proc._validate_obs_times_shape(obs, times)


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


# ===================================================================
# Counts.validate_data()
# ===================================================================


class TestCountsValidateData:
    """Tests for Counts.validate_data()."""

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

    def test_extra_kwargs_ignored(self, counts_proc):
        """validate_data should ignore extra keyword arguments."""
        counts_proc.validate_data(
            n_total=30, n_subpops=1, obs=None, extra_param="ignored"
        )


# ===================================================================
# CountsBySubpop.validate_data()
# ===================================================================


class TestCountsBySubpopValidateData:
    """Tests for CountsBySubpop.validate_data()."""

    def test_all_none_passes(self, subpop_proc):
        """validate_data with all optional args None should not raise."""
        subpop_proc.validate_data(n_total=30, n_subpops=3)

    def test_valid_data_passes(self, subpop_proc):
        """validate_data with valid times, subpop_indices, obs should not raise."""
        times = jnp.array([5, 10, 15, 20])
        subpop_indices = jnp.array([0, 1, 2, 0])
        obs = jnp.array([10.0, 20.0, 30.0, 15.0])
        subpop_proc.validate_data(
            n_total=30,
            n_subpops=3,
            times=times,
            subpop_indices=subpop_indices,
            obs=obs,
        )

    def test_invalid_times_raises(self, subpop_proc):
        """validate_data with out-of-bounds times should raise."""
        times = jnp.array([5, 30])  # 30 == n_total, out of bounds
        with pytest.raises(ValueError, match="upper bound"):
            subpop_proc.validate_data(n_total=30, n_subpops=3, times=times)

    def test_negative_times_raises(self, subpop_proc):
        """validate_data with negative times should raise."""
        times = jnp.array([-1, 5])
        with pytest.raises(ValueError, match="cannot be negative"):
            subpop_proc.validate_data(n_total=30, n_subpops=3, times=times)

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
        """validate_data with obs/times shape mismatch should raise."""
        times = jnp.array([5, 10, 15])
        obs = jnp.array([1.0, 2.0])  # length 2 != length 3
        with pytest.raises(ValueError, match="must match times shape"):
            subpop_proc.validate_data(n_total=30, n_subpops=3, times=times, obs=obs)

    def test_obs_without_times_skips_shape_check(self, subpop_proc):
        """validate_data with obs but no times should not check shapes."""
        obs = jnp.array([1.0, 2.0])
        # times is None, so shape check is skipped
        subpop_proc.validate_data(n_total=30, n_subpops=3, obs=obs)

    def test_times_without_obs_skips_shape_check(self, subpop_proc):
        """validate_data with times but no obs should validate times only."""
        times = jnp.array([5, 10, 15])
        subpop_proc.validate_data(n_total=30, n_subpops=3, times=times)

    def test_non_contiguous_subpop_indices_valid(self, subpop_proc):
        """validate_data with non-contiguous but valid subpop_indices passes."""
        subpop_indices = jnp.array([0, 2])  # skip index 1
        subpop_proc.validate_data(
            n_total=30, n_subpops=3, subpop_indices=subpop_indices
        )


# ===================================================================
# Measurements.validate_data()
# ===================================================================


class TestMeasurementsValidateData:
    """Tests for Measurements.validate_data()."""

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
        with pytest.raises(ValueError, match="must match times shape"):
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
