"""
Tests for ascertainment models.
"""

import jax.numpy as jnp
import numpyro
import pytest

from pyrenew.ascertainment import (
    AscertainmentSignal,
    JointAscertainment,
)
from pyrenew.ascertainment.context import (
    ascertainment_context,
    get_ascertainment_value,
)


class TestJointAscertainmentValidation:
    """Test JointAscertainment constructor validation."""

    @pytest.mark.parametrize("name", ["", None])
    def test_requires_non_empty_name(self, name):
        """Test that ascertainment model names must be non-empty strings."""
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            JointAscertainment(
                name=name,
                signals=("hospital", "ed"),
                baseline_rates=jnp.full(2, 0.5),
                scale_tril=jnp.eye(2),
            )

    @pytest.mark.parametrize(
        "signals",
        [
            (),
            ["hospital", "ed"],
            ("hospital", ""),
            ("hospital", None),
        ],
    )
    def test_requires_non_empty_tuple_of_string_signals(self, signals):
        """Test that signals must be a non-empty tuple of non-empty strings."""
        with pytest.raises(ValueError, match="signals|all signals"):
            JointAscertainment(
                name="he_ascertainment",
                signals=signals,
                baseline_rates=jnp.full(2, 0.5),
                scale_tril=jnp.eye(2),
            )

    def test_requires_unique_signals(self):
        """Test that signal names must be unique."""
        with pytest.raises(ValueError, match="signals must be unique"):
            JointAscertainment(
                name="he_ascertainment",
                signals=("hospital", "hospital"),
                baseline_rates=jnp.full(2, 0.5),
                scale_tril=jnp.eye(2),
            )

    def test_rejects_unknown_signal(self):
        """Test that for_signal rejects unknown signals."""
        ascertainment = JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed"),
            baseline_rates=jnp.full(2, 0.5),
            scale_tril=jnp.eye(2),
        )

        with pytest.raises(ValueError, match="Unknown signal"):
            ascertainment.for_signal("wastewater")

    def test_requires_baseline_rates_shape_to_match_signals(self):
        """Test that baseline_rates must have one entry per signal."""
        with pytest.raises(ValueError, match="baseline_rates must have shape"):
            JointAscertainment(
                name="he_ascertainment",
                signals=("hospital", "ed"),
                baseline_rates=jnp.full(3, 0.5),
                scale_tril=jnp.eye(2),
            )

    @pytest.mark.parametrize("baseline_rates", [[0.0, 0.5], [1.0, 0.5], [-0.1, 0.5]])
    def test_requires_baseline_rates_in_open_unit_interval(self, baseline_rates):
        """Test that baseline_rates must be natural-scale probabilities."""
        with pytest.raises(ValueError, match="baseline_rates must contain"):
            JointAscertainment(
                name="he_ascertainment",
                signals=("hospital", "ed"),
                baseline_rates=jnp.array(baseline_rates),
                scale_tril=jnp.eye(2),
            )

    def test_requires_exactly_one_covariance_parameter(self):
        """Test that exactly one multivariate normal matrix parameter is set."""
        with pytest.raises(ValueError, match="Exactly one"):
            JointAscertainment(
                name="he_ascertainment",
                signals=("hospital", "ed"),
                baseline_rates=jnp.full(2, 0.5),
            )

        with pytest.raises(ValueError, match="Exactly one"):
            JointAscertainment(
                name="he_ascertainment",
                signals=("hospital", "ed"),
                baseline_rates=jnp.full(2, 0.5),
                scale_tril=jnp.eye(2),
                covariance_matrix=jnp.eye(2),
            )

    def test_requires_matrix_shape_to_match_signals(self):
        """Test that the covariance parameter must match signal count."""
        with pytest.raises(ValueError, match="scale_tril must have shape"):
            JointAscertainment(
                name="he_ascertainment",
                signals=("hospital", "ed"),
                baseline_rates=jnp.full(2, 0.5),
                scale_tril=jnp.eye(3),
            )

    def test_accepts_covariance_matrix(self):
        """Test that covariance_matrix is accepted as the covariance parameter."""
        ascertainment = JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed"),
            baseline_rates=jnp.full(2, 0.5),
            covariance_matrix=jnp.eye(2),
        )

        assert ascertainment.covariance_matrix.shape == (2, 2)

    def test_accepts_precision_matrix(self):
        """Test that precision_matrix is accepted as the covariance parameter."""
        ascertainment = JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed"),
            baseline_rates=jnp.full(2, 0.5),
            precision_matrix=jnp.eye(2),
        )

        assert ascertainment.precision_matrix.shape == (2, 2)


class TestAscertainmentSignalValidation:
    """Test AscertainmentSignal constructor validation."""

    @pytest.mark.parametrize("ascertainment_name", ["", None])
    def test_requires_non_empty_ascertainment_name(self, ascertainment_name):
        """Test that ascertainment_name must be a non-empty string."""
        with pytest.raises(ValueError, match="ascertainment_name"):
            AscertainmentSignal(
                ascertainment_name=ascertainment_name,
                signal_name="hospital",
            )

    @pytest.mark.parametrize("signal_name", ["", None])
    def test_requires_non_empty_signal_name(self, signal_name):
        """Test that signal_name must be a non-empty string."""
        with pytest.raises(ValueError, match="signal_name"):
            AscertainmentSignal(
                ascertainment_name="he_ascertainment",
                signal_name=signal_name,
            )


class TestJointAscertainmentSampling:
    """Test JointAscertainment sampling behavior."""

    def test_sample_creates_one_joint_sample_site_and_signal_deterministics(self):
        """Test expected NumPyro sites and returned signal values."""
        ascertainment = JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed"),
            baseline_rates=jnp.full(2, 0.5),
            scale_tril=jnp.eye(2),
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                values = ascertainment.sample()

        assert set(values) == {"hospital", "ed"}
        assert trace["he_ascertainment_eta"]["type"] == "sample"
        assert trace["he_ascertainment_eta"]["value"].shape == (2,)
        assert trace["he_ascertainment_hospital"]["type"] == "deterministic"
        assert trace["he_ascertainment_ed"]["type"] == "deterministic"
        assert jnp.array_equal(
            values["hospital"],
            trace["he_ascertainment_hospital"]["value"],
        )
        assert jnp.array_equal(
            values["ed"],
            trace["he_ascertainment_ed"]["value"],
        )

    def test_sample_accepts_covariance_matrix_parameterization(self):
        """Test joint ascertainment sampling with a covariance matrix."""
        ascertainment = JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed"),
            baseline_rates=jnp.full(2, 0.5),
            covariance_matrix=jnp.eye(2),
        )

        with numpyro.handlers.seed(rng_seed=42):
            values = ascertainment.sample()

        assert set(values) == {"hospital", "ed"}

    def test_sample_accepts_precision_matrix_parameterization(self):
        """Test joint ascertainment sampling with a precision matrix."""
        ascertainment = JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed"),
            baseline_rates=jnp.full(2, 0.5),
            precision_matrix=jnp.eye(2),
        )

        with numpyro.handlers.seed(rng_seed=42):
            values = ascertainment.sample()

        assert set(values) == {"hospital", "ed"}

    def test_signal_accessor_reads_context_without_creating_sites(self):
        """Test that signal accessors read context values and create no sites."""
        ascertainment = JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed"),
            baseline_rates=jnp.full(2, 0.5),
            scale_tril=jnp.eye(2),
        )
        hospital = ascertainment.for_signal("hospital")

        with numpyro.handlers.trace() as trace:
            with ascertainment_context(
                {"he_ascertainment": {"hospital": jnp.array(0.25)}}
            ):
                value = hospital()

        assert value == jnp.array(0.25)
        assert trace == {}

    def test_reused_signal_accessor_creates_no_duplicate_sites(self):
        """Test repeated accessor calls still create no NumPyro sites."""
        ascertainment = JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed"),
            baseline_rates=jnp.full(2, 0.5),
            scale_tril=jnp.eye(2),
        )
        hospital = ascertainment.for_signal("hospital")

        with numpyro.handlers.trace() as trace:
            with ascertainment_context(
                {"he_ascertainment": {"hospital": jnp.array(0.25)}}
            ):
                first = hospital()
                second = hospital()

        assert first == jnp.array(0.25)
        assert second == jnp.array(0.25)
        assert trace == {}

    def test_signal_accessor_requires_active_context(self):
        """Test that signal accessors fail clearly outside model context."""
        ascertainment = JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed"),
            baseline_rates=jnp.full(2, 0.5),
            scale_tril=jnp.eye(2),
        )

        with pytest.raises(RuntimeError, match="before ascertainment values"):
            ascertainment.for_signal("hospital")()


class TestAscertainmentContextSafety:
    """Test ascertainment context lifecycle and validation."""

    @pytest.mark.parametrize(
        "values, error_type",
        [
            (None, TypeError),
            ({"he_ascertainment": None}, TypeError),
            ({"": {"hospital": jnp.array(0.1)}}, ValueError),
            ({"he_ascertainment": {"": jnp.array(0.1)}}, ValueError),
        ],
    )
    def test_context_rejects_invalid_values(self, values, error_type):
        """Test that malformed context payloads fail at context entry."""
        with pytest.raises(error_type):
            with ascertainment_context(values):
                pass

    @pytest.mark.parametrize(
        "ascertainment_name, signal_name",
        [
            ("", "hospital"),
            ("he_ascertainment", ""),
        ],
    )
    def test_get_ascertainment_value_validates_lookup_names(
        self,
        ascertainment_name,
        signal_name,
    ):
        """Test that context lookup names must be non-empty strings."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            get_ascertainment_value(ascertainment_name, signal_name)

    def test_context_restores_outer_context_after_nested_context(self):
        """Test nested contexts restore previous values on exit."""
        with ascertainment_context({"he_ascertainment": {"hospital": jnp.array(0.1)}}):
            assert get_ascertainment_value("he_ascertainment", "hospital") == 0.1
            with ascertainment_context(
                {"he_ascertainment": {"hospital": jnp.array(0.2)}}
            ):
                assert get_ascertainment_value("he_ascertainment", "hospital") == 0.2
            assert get_ascertainment_value("he_ascertainment", "hospital") == 0.1

    def test_context_clears_after_exception(self):
        """Test context is cleared even when an exception is raised."""
        with pytest.raises(RuntimeError, match="boom"):
            with ascertainment_context(
                {"he_ascertainment": {"hospital": jnp.array(0.1)}}
            ):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="before ascertainment values"):
            get_ascertainment_value("he_ascertainment", "hospital")

    def test_missing_context_value_raises_clear_error(self):
        """Test unavailable context keys raise a clear RuntimeError."""
        with ascertainment_context({"he_ascertainment": {"hospital": jnp.array(0.1)}}):
            with pytest.raises(RuntimeError, match="not available"):
                get_ascertainment_value("he_ascertainment", "ed")
