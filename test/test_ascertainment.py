"""
Tests for ascertainment models.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax.scipy.special import logit
from jax.typing import ArrayLike

from pyrenew.ascertainment import (
    AscertainmentModel,
    AscertainmentSignal,
    JointAscertainment,
    RatioLinkedAscertainment,
    TimeVaryingAscertainment,
)
from pyrenew.ascertainment.context import (
    ascertainment_context,
    get_ascertainment_value,
)
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import WeeklyTemporalProcess
from pyrenew.randomvariable import DistributionalVariable


class DeterministicTemporalProcess:
    """Generate a predictable two-dimensional temporal trajectory for tests."""

    step_size = 1

    def __init__(
        self,
        initial: float = 0.0,
        increment: float = 0.0,
        result: ArrayLike | None = None,
    ) -> None:
        """Initialize a deterministic temporal-process test double."""
        self.initial = initial
        self.increment = increment
        self.result = result
        self.sample_calls: list[dict[str, object]] = []

    def sample(
        self,
        n_timepoints: int,
        initial_value: float | ArrayLike | None = None,
        n_processes: int = 1,
        name_prefix: str = "temporal",
        *,
        first_day_dow: int | None = None,
    ) -> ArrayLike:
        """Return a deterministic trajectory with the requested shape.

        Returns
        -------
        ArrayLike
            Configured result or generated two-dimensional trajectory.
        """
        self.sample_calls.append(
            {
                "n_timepoints": n_timepoints,
                "initial_value": initial_value,
                "n_processes": n_processes,
                "name_prefix": name_prefix,
                "first_day_dow": first_day_dow,
            }
        )
        if self.result is not None:
            return jnp.asarray(self.result)
        values = self.initial + self.increment * jnp.arange(n_timepoints)
        return jnp.broadcast_to(values[:, None], (n_timepoints, n_processes))


class FixedAscertainmentModel(AscertainmentModel):
    """Return fixed ascertainment values for baseline-contract tests."""

    def __init__(
        self,
        name: str,
        values: Mapping[str, ArrayLike],
        requires_calendar_anchor: bool = False,
    ) -> None:
        """Initialize a fixed ascertainment test double."""
        super().__init__(name=name, signals=tuple(values))
        self.values = dict(values)
        self._requires_calendar_anchor = requires_calendar_anchor
        self.n_sample_calls = 0

    def requires_calendar_anchor(self) -> bool:
        """Return the configured calendar requirement.

        Returns
        -------
        bool
            Configured calendar requirement.
        """
        return self._requires_calendar_anchor

    def sample(self, **kwargs: object) -> Mapping[str, ArrayLike]:
        """Return configured values and record the sampling call.

        Returns
        -------
        Mapping[str, ArrayLike]
            Configured signal values.
        """
        self.n_sample_calls += 1
        return self.values


class TestAscertainmentModelContract:
    """Test validation shared by all ascertainment models."""

    def test_accepts_scalar_and_full_axis_values(self) -> None:
        """Test scalar and exact full-axis trajectories are valid."""
        model = FixedAscertainmentModel(
            "ascertainment",
            {"hospital": 0.2, "ed": 0.3},
        )

        model.validate_sampled_values(
            {"hospital": jnp.array(0.2), "ed": jnp.arange(4)},
            n_timepoints=4,
        )

    def test_rejects_non_mapping_values(self) -> None:
        """Test sampled values must be returned as a mapping."""
        model = FixedAscertainmentModel("ascertainment", {"hospital": 0.2})

        with pytest.raises(TypeError, match="ascertainment.*mapping"):
            model.validate_sampled_values([0.2], n_timepoints=4)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "values",
        [
            {},
            {"hospital": 0.2, "ed": 0.3},
        ],
    )
    def test_rejects_missing_or_extra_signals(
        self,
        values: Mapping[str, ArrayLike],
    ) -> None:
        """Test sampled signal names must exactly match the model signals."""
        model = FixedAscertainmentModel("ascertainment", {"hospital": 0.2})

        with pytest.raises(ValueError, match="ascertainment.*signals"):
            model.validate_sampled_values(values, n_timepoints=4)

    @pytest.mark.parametrize(
        "value",
        [
            jnp.ones(1),
            jnp.ones(3),
            jnp.ones((4, 1)),
            jnp.ones((4, 1, 1)),
        ],
    )
    def test_rejects_invalid_sample_shapes(self, value: ArrayLike) -> None:
        """Test only scalar or exact full-axis values are accepted."""
        model = FixedAscertainmentModel("ascertainment", {"hospital": 0.2})

        with pytest.raises(
            ValueError,
            match=r"ascertainment.*hospital.*shape.*\(\).*\(4,\)",
        ):
            model.validate_sampled_values({"hospital": value}, n_timepoints=4)

    def test_scalar_models_do_not_require_a_calendar_anchor(self) -> None:
        """Test the base calendar declaration defaults to false."""
        model = JointAscertainment(
            name="ascertainment",
            signals=("hospital",),
            baseline_rates=jnp.array([0.2]),
            scale_tril=jnp.eye(1),
        )

        assert not model.requires_calendar_anchor()


class TestTimeVaryingAscertainmentConstruction:
    """Test time-varying ascertainment constructor validation."""

    def test_requires_an_ascertainment_baseline(self) -> None:
        """Test the baseline must implement the ascertainment contract."""
        with pytest.raises(TypeError, match="baseline_model"):
            TimeVaryingAscertainment(
                name="outer",
                baseline_model=object(),  # type: ignore[arg-type]
                processes={"hospital": DeterministicTemporalProcess()},
            )

    def test_requires_distinct_outer_and_baseline_names(self) -> None:
        """Test outer and baseline NumPyro namespaces must differ."""
        baseline = FixedAscertainmentModel("same", {"hospital": 0.2})

        with pytest.raises(ValueError, match="different names"):
            TimeVaryingAscertainment(
                name="same",
                baseline_model=baseline,
                processes={"hospital": DeterministicTemporalProcess()},
            )

    def test_requires_a_process_mapping(self) -> None:
        """Test temporal processes must be provided in a mapping."""
        baseline = FixedAscertainmentModel("baseline", {"hospital": 0.2})

        with pytest.raises(TypeError, match="processes must be a mapping"):
            TimeVaryingAscertainment(
                name="outer",
                baseline_model=baseline,
                processes=[],  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "processes",
        [
            {},
            {
                "hospital": DeterministicTemporalProcess(),
                "ed": DeterministicTemporalProcess(),
            },
        ],
    )
    def test_requires_exact_baseline_signal_keys(
        self,
        processes: Mapping[str, DeterministicTemporalProcess],
    ) -> None:
        """Test process signal names must exactly match baseline signals."""
        baseline = FixedAscertainmentModel("baseline", {"hospital": 0.2})

        with pytest.raises(ValueError, match="baseline model signals"):
            TimeVaryingAscertainment("outer", baseline, processes)

    def test_requires_temporal_process_protocol(self) -> None:
        """Test every configured process must satisfy TemporalProcess."""
        baseline = FixedAscertainmentModel("baseline", {"hospital": 0.2})

        with pytest.raises(TypeError, match="TemporalProcess"):
            TimeVaryingAscertainment(
                "outer",
                baseline,
                {"hospital": object()},  # type: ignore[dict-item]
            )

    def test_orders_processes_like_baseline_signals(self) -> None:
        """Test deterministic process order follows the baseline model."""
        baseline = FixedAscertainmentModel(
            "baseline",
            {"hospital": 0.2, "ed": 0.3},
        )
        processes = {
            "ed": DeterministicTemporalProcess(),
            "hospital": DeterministicTemporalProcess(),
        }

        model = TimeVaryingAscertainment("outer", baseline, processes)

        assert tuple(model.processes) == ("hospital", "ed")


class TestTimeVaryingAscertainmentSampling:
    """Test baseline and temporal sampling semantics."""

    def test_joint_baseline_sites_and_independent_trajectories(self) -> None:
        """Test a joint baseline is sampled once before independent deviations."""
        baseline_rates = jnp.array([0.2, 0.4])
        baseline = JointAscertainment(
            name="he_baseline",
            signals=("hospital", "ed"),
            baseline_rates=baseline_rates,
            scale_tril=jnp.eye(2),
        )
        model = TimeVaryingAscertainment(
            name="he_ascertainment",
            baseline_model=baseline,
            processes={
                "hospital": DeterministicTemporalProcess(increment=0.3),
                "ed": DeterministicTemporalProcess(increment=-0.2),
            },
        )

        with numpyro.handlers.substitute(
            data={"he_baseline_eta": logit(baseline_rates)}
        ):
            with numpyro.handlers.trace() as trace:
                values = model.sample(n_timepoints=4)

        assert trace["he_baseline_eta"]["type"] == "sample"
        assert trace["he_baseline_eta"]["value"].shape == (2,)
        assert trace["he_baseline_hospital"]["value"].shape == ()
        assert trace["he_baseline_ed"]["value"].shape == ()
        assert trace["he_ascertainment_hospital"]["value"].shape == (4,)
        assert trace["he_ascertainment_ed"]["value"].shape == (4,)
        assert jnp.allclose(values["hospital"][0], baseline_rates[0])
        assert jnp.allclose(values["ed"][0], baseline_rates[1])
        assert not jnp.allclose(
            values["hospital"][1:] / values["ed"][1:],
            baseline_rates[0] / baseline_rates[1],
        )

    def test_ratio_linked_baseline_relationship_is_not_pointwise(self) -> None:
        """Test linked baseline sites persist without constraining trajectories."""
        baseline = RatioLinkedAscertainment(
            name="he_baseline",
            base_signal="ed",
            linked_signal="hospital",
            base_rate_rv=DistributionalVariable("iedr", dist.Delta(0.4)),
            ratio_rv=DistributionalVariable("ihr_rel_iedr", dist.Delta(0.5)),
        )
        model = TimeVaryingAscertainment(
            name="he_ascertainment",
            baseline_model=baseline,
            processes={
                "ed": DeterministicTemporalProcess(increment=-0.1),
                "hospital": DeterministicTemporalProcess(increment=0.2),
            },
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                values = model.sample(n_timepoints=4)

        assert trace["iedr"]["type"] == "sample"
        assert trace["ihr_rel_iedr"]["type"] == "sample"
        assert jnp.allclose(trace["he_baseline_ed"]["value"], 0.4)
        assert jnp.allclose(trace["he_baseline_hospital"]["value"], 0.2)
        assert values["ed"].shape == (4,)
        assert values["hospital"].shape == (4,)
        trajectory_ratios = values["hospital"] / values["ed"]
        assert jnp.allclose(trajectory_ratios[0], 0.5)
        assert not jnp.allclose(trajectory_ratios[1:], 0.5)

    def test_calendar_requirement_and_weekly_alignment(self) -> None:
        """Test weekly processes declare and use the model-axis weekday."""
        baseline = FixedAscertainmentModel("baseline", {"hospital": 0.2})
        weekly = WeeklyTemporalProcess(
            DeterministicTemporalProcess(increment=1.0),
            start_dow=0,
        )
        model = TimeVaryingAscertainment(
            "outer",
            baseline,
            {"hospital": weekly},
        )

        values = model.sample(n_timepoints=10, first_day_dow=2)

        assert model.requires_calendar_anchor()
        assert jnp.allclose(values["hospital"][:5], 0.2)
        assert jnp.all(values["hospital"][5:] > 0.2)

    @pytest.mark.parametrize("baseline_value", [0.0, 1.0, -0.1, 1.1, jnp.nan])
    def test_rejects_invalid_concrete_baselines(
        self,
        baseline_value: float,
    ) -> None:
        """Test concrete baselines must be finite probabilities."""
        baseline = FixedAscertainmentModel(
            "baseline",
            {"hospital": baseline_value},
        )
        model = TimeVaryingAscertainment(
            "outer",
            baseline,
            {"hospital": DeterministicTemporalProcess()},
        )

        with pytest.raises(ValueError, match=r"hospital.*\(0, 1\)"):
            model.sample(n_timepoints=4)

    def test_rejects_trajectory_baseline(self) -> None:
        """Test nested time-varying baselines are rejected."""
        baseline = FixedAscertainmentModel(
            "baseline",
            {"hospital": jnp.full(4, 0.2)},
        )
        model = TimeVaryingAscertainment(
            "outer",
            baseline,
            {"hospital": DeterministicTemporalProcess()},
        )

        with pytest.raises(ValueError, match=r"scalar baseline.*\(4,\).*"):
            model.sample(n_timepoints=4)

    def test_rejects_malformed_temporal_output_before_squeezing(self) -> None:
        """Test temporal output must retain the singleton process dimension."""
        baseline = FixedAscertainmentModel("baseline", {"hospital": 0.2})
        model = TimeVaryingAscertainment(
            "outer",
            baseline,
            {"hospital": DeterministicTemporalProcess(result=jnp.zeros(4))},
        )

        with pytest.raises(ValueError, match=r"outer.*hospital.*\(4,\).*\(4, 1\)"):
            model.sample(n_timepoints=4)


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
        with pytest.raises(ValueError, match="Incompatible shapes"):
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

        assert ascertainment.distribution.covariance_matrix.shape == (2, 2)

    def test_accepts_precision_matrix(self):
        """Test that precision_matrix is accepted as the covariance parameter."""
        ascertainment = JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed"),
            baseline_rates=jnp.full(2, 0.5),
            precision_matrix=jnp.eye(2),
        )

        assert ascertainment.distribution.precision_matrix.shape == (2, 2)

    def test_baseline_rates_returns_natural_scale_rates(self):
        """Test that baseline_rates returns rates on the probability scale."""
        baseline_rates = jnp.array([0.2, 0.7])
        ascertainment = JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed"),
            baseline_rates=baseline_rates,
            scale_tril=jnp.eye(2),
        )

        assert jnp.allclose(ascertainment.baseline_rates, baseline_rates)


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


class TestRatioLinkedAscertainmentValidation:
    """Test RatioLinkedAscertainment constructor validation."""

    @pytest.mark.parametrize("name", ["", None])
    def test_requires_non_empty_name(self, name):
        """Test that ratio-linked model names must be non-empty strings."""
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            RatioLinkedAscertainment(
                name=name,
                base_signal="ed_visits",
                linked_signal="hospital",
                base_rate_rv=DeterministicVariable("iedr", 0.2),
                ratio_rv=DeterministicVariable("ihr_rel_iedr", 0.5),
            )

    @pytest.mark.parametrize(
        "base_signal, linked_signal, error_match",
        [
            ("", "hospital", "all signals"),
            ("ed_visits", "", "all signals"),
            ("hospital", "hospital", "signals must be unique"),
        ],
    )
    def test_requires_distinct_non_empty_signal_names(
        self,
        base_signal,
        linked_signal,
        error_match,
    ):
        """Test that base and linked signals are non-empty and distinct."""
        with pytest.raises(ValueError, match=error_match):
            RatioLinkedAscertainment(
                name="he_ascertainment",
                base_signal=base_signal,
                linked_signal=linked_signal,
                base_rate_rv=DeterministicVariable("iedr", 0.2),
                ratio_rv=DeterministicVariable("ihr_rel_iedr", 0.5),
            )

    def test_stores_configuration_and_creates_signal_accessors(self):
        """Test that constructor values and signal accessors are retained."""
        base_rate_rv = DeterministicVariable("iedr", 0.2)
        ratio_rv = DeterministicVariable("ihr_rel_iedr", 0.5)
        ascertainment = RatioLinkedAscertainment(
            name="he_ascertainment",
            base_signal="ed_visits",
            linked_signal="hospital",
            base_rate_rv=base_rate_rv,
            ratio_rv=ratio_rv,
        )

        assert ascertainment.signals == ("ed_visits", "hospital")
        assert ascertainment.base_signal == "ed_visits"
        assert ascertainment.linked_signal == "hospital"
        assert ascertainment.base_rate_rv is base_rate_rv
        assert ascertainment.ratio_rv is ratio_rv

        base_accessor = ascertainment.for_signal("ed_visits")
        linked_accessor = ascertainment.for_signal("hospital")
        assert isinstance(base_accessor, AscertainmentSignal)
        assert isinstance(linked_accessor, AscertainmentSignal)
        assert base_accessor.ascertainment_name == "he_ascertainment"
        assert linked_accessor.ascertainment_name == "he_ascertainment"
        assert base_accessor.signal_name == "ed_visits"
        assert linked_accessor.signal_name == "hospital"

        with pytest.raises(ValueError, match="Unknown signal"):
            ascertainment.for_signal("wastewater")


class TestRatioLinkedAscertainmentSampling:
    """Test RatioLinkedAscertainment sampling behavior."""

    def test_sample_returns_base_rate_and_scaled_linked_rate(self):
        """Test that the linked rate is the base rate times the ratio."""
        ascertainment = RatioLinkedAscertainment(
            name="he_ascertainment",
            base_signal="ed_visits",
            linked_signal="hospital",
            base_rate_rv=DeterministicVariable("iedr", jnp.array(0.2)),
            ratio_rv=DeterministicVariable("ihr_rel_iedr", jnp.array(0.5)),
        )

        with numpyro.handlers.trace() as trace:
            values = ascertainment.sample()

        assert set(values) == {"ed_visits", "hospital"}
        assert jnp.array_equal(values["ed_visits"], jnp.array(0.2))
        assert jnp.array_equal(values["hospital"], jnp.array(0.1))
        assert trace["he_ascertainment_ed_visits"]["type"] == "deterministic"
        assert trace["he_ascertainment_hospital"]["type"] == "deterministic"
        assert jnp.array_equal(
            values["ed_visits"],
            trace["he_ascertainment_ed_visits"]["value"],
        )
        assert jnp.array_equal(
            values["hospital"],
            trace["he_ascertainment_hospital"]["value"],
        )

    def test_sample_calls_each_random_variable_once(self):
        """Test that the base-rate site and the ratio site are sampled and converted into user-exposed, signal specific deterministic sites"""
        ascertainment = RatioLinkedAscertainment(
            name="he_ascertainment",
            base_signal="ed_visits",
            linked_signal="hospital",
            base_rate_rv=DistributionalVariable("iedr", dist.Delta(0.2)),
            ratio_rv=DistributionalVariable("ihr_rel_iedr", dist.Delta(0.5)),
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                values = ascertainment.sample()

        assert trace["iedr"]["type"] == "sample"
        assert trace["ihr_rel_iedr"]["type"] == "sample"
        assert jnp.array_equal(values["ed_visits"], trace["iedr"]["value"])
        assert jnp.array_equal(
            values["hospital"],
            trace["iedr"]["value"] * trace["ihr_rel_iedr"]["value"],
        )


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

    def test_missing_context_value_raises_clear_error(self):
        """Test unavailable context keys raise a clear RuntimeError."""
        with ascertainment_context({"he_ascertainment": {"hospital": jnp.array(0.1)}}):
            with pytest.raises(RuntimeError, match="not available"):
                get_ascertainment_value("he_ascertainment", "ed")

    def test_missing_context_model_raises_clear_error(self):
        """Test unavailable ascertainment model keys raise a clear RuntimeError."""
        with ascertainment_context({"he_ascertainment": {"hospital": jnp.array(0.1)}}):
            with pytest.raises(RuntimeError, match="Values for ascertainment model"):
                get_ascertainment_value("ww_ascertainment", "hospital")

    def test_context_clears_after_exception(self):
        """Test context is cleared even when an exception is raised."""
        with pytest.raises(RuntimeError, match="boom"):
            with ascertainment_context(
                {"he_ascertainment": {"hospital": jnp.array(0.1)}}
            ):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="before ascertainment values"):
            get_ascertainment_value("he_ascertainment", "hospital")
