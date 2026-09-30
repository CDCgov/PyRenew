"""
Tests for ascertainment models.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax.scipy.special import logit
from jax.typing import ArrayLike

from pyrenew.ascertainment import (
    AscertainmentModel,
    AscertainmentSignal,
    IndependentAscertainment,
    JointAscertainment,
    RatioLinkedAscertainment,
)
from pyrenew.ascertainment.context import (
    ascertainment_context,
    get_ascertainment_value,
)
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import TemporalProcess, WeeklyTemporalProcess
from pyrenew.metaclass import RandomVariable
from pyrenew.randomvariable import DistributionalVariable


class DeterministicTemporalProcess:
    """Generate a predictable two-dimensional temporal trajectory for tests."""

    step_size = 1

    def __init__(
        self,
        initial: float = 0.0,
        increment: float = 0.0,
        result: ArrayLike | None = None,
        requires_calendar_anchor: bool = False,
    ) -> None:
        """Initialize a deterministic temporal-process test double."""
        self.initial = initial
        self.increment = increment
        self.result = result
        self.requires_calendar_anchor = requires_calendar_anchor
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


class CountingVariable(RandomVariable):
    """Return a configured value and count sampling calls."""

    def __init__(self, name: str, value: ArrayLike) -> None:
        """Initialize a counting random variable."""
        super().__init__(name=name)
        self.value = value
        self.n_calls = 0

    def sample(self, **kwargs: object) -> ArrayLike:
        """Return the configured value and increment the call count.

        Returns
        -------
        ArrayLike
            Configured value.
        """
        self.n_calls += 1
        return self.value


class FixedAscertainmentModel(AscertainmentModel):
    """Return fixed ascertainment values for baseline-contract tests."""

    def __init__(
        self,
        name: str,
        values: Mapping[str, ArrayLike],
        temporal_processes: Mapping[str, TemporalProcess] | None = None,
    ) -> None:
        """Initialize a fixed ascertainment test double."""
        super().__init__(
            name=name,
            signals=tuple(values),
            temporal_processes=temporal_processes,
        )
        self.values = dict(values)
        self.n_sample_calls = 0

    def _sample_baseline_rates(self) -> Mapping[str, ArrayLike]:
        """Return configured scalar baselines and record the sampling call.

        Returns
        -------
        Mapping[str, ArrayLike]
            Configured signal baselines.
        """
        self.n_sample_calls += 1
        return self.values


class TestAscertainmentModelContract:
    """Test validation shared by all ascertainment models."""

    def test_baseline_sampling_is_the_only_abstract_method(self) -> None:
        """Test custom subclasses only need to implement baseline sampling."""
        assert AscertainmentModel.__abstractmethods__ == frozenset(
            {"_sample_baseline_rates"}
        )

    def test_temporal_processes_default_to_an_empty_dictionary(self) -> None:
        """Test models without temporal processes store an empty dictionary."""
        model = FixedAscertainmentModel("ascertainment", {"hospital": 0.2})

        assert model.temporal_processes == {}

    def test_accepts_an_empty_temporal_process_mapping(self) -> None:
        """Test an explicitly empty temporal mapping is valid."""
        model = FixedAscertainmentModel(
            "ascertainment",
            {"hospital": 0.2},
            temporal_processes={},
        )

        assert model.temporal_processes == {}

    def test_temporal_processes_are_copied_in_signal_order(self) -> None:
        """Test process storage follows signal order and does not alias input."""
        hospital_process = DeterministicTemporalProcess()
        ed_process = DeterministicTemporalProcess()
        processes: dict[str, TemporalProcess] = {
            "ed": ed_process,
            "hospital": hospital_process,
        }

        model = FixedAscertainmentModel(
            "ascertainment",
            {"hospital": 0.2, "ed": 0.3},
            temporal_processes=processes,
        )
        processes.clear()

        assert tuple(model.temporal_processes) == ("hospital", "ed")
        assert model.temporal_processes["hospital"] is hospital_process
        assert model.temporal_processes["ed"] is ed_process

    def test_temporal_processes_may_cover_a_signal_subset(self) -> None:
        """Test only selected signals need temporal processes."""
        process = DeterministicTemporalProcess()

        model = FixedAscertainmentModel(
            "ascertainment",
            {"hospital": 0.2, "ed": 0.3},
            temporal_processes={"ed": process},
        )

        assert model.temporal_processes == {"ed": process}

    def test_rejects_non_mapping_temporal_processes(self) -> None:
        """Test temporal process configuration must be a mapping."""
        with pytest.raises(TypeError, match="temporal_processes.*mapping"):
            FixedAscertainmentModel(
                "ascertainment",
                {"hospital": 0.2},
                temporal_processes=[],  # type: ignore[arg-type]
            )

    def test_rejects_unknown_temporal_process_signal(self) -> None:
        """Test processes cannot be configured for undeclared signals."""
        with pytest.raises(ValueError, match="unknown signals.*ed"):
            FixedAscertainmentModel(
                "ascertainment",
                {"hospital": 0.2},
                temporal_processes={"ed": DeterministicTemporalProcess()},
            )

    def test_rejects_invalid_temporal_process(self) -> None:
        """Test every process must satisfy the runtime protocol."""
        with pytest.raises(TypeError, match="hospital.*TemporalProcess"):
            FixedAscertainmentModel(
                "ascertainment",
                {"hospital": 0.2},
                temporal_processes={"hospital": object()},  # type: ignore[dict-item]
            )

    def test_calendar_requirement_checks_all_temporal_processes(self) -> None:
        """Test any calendar-aligned process makes the model require a date."""
        model = FixedAscertainmentModel(
            "ascertainment",
            {"hospital": 0.2, "ed": 0.3},
            temporal_processes={
                "hospital": DeterministicTemporalProcess(),
                "ed": DeterministicTemporalProcess(requires_calendar_anchor=True),
            },
        )

        assert model.requires_calendar_anchor()

    def test_daily_process_does_not_require_a_calendar_anchor(self) -> None:
        """Test ordinary daily processes do not require a date."""
        model = FixedAscertainmentModel(
            "ascertainment",
            {"hospital": 0.2},
            temporal_processes={"hospital": DeterministicTemporalProcess()},
        )

        assert not model.requires_calendar_anchor()

    @pytest.mark.parametrize("n_timepoints", [True, 1.0, jnp.array(1)])
    def test_rejects_non_integer_n_timepoints(self, n_timepoints: object) -> None:
        """Test only built-in integers are accepted for the axis length."""
        model = FixedAscertainmentModel("ascertainment", {"hospital": 0.2})

        with pytest.raises(TypeError, match="n_timepoints.*positive integer"):
            model.sample(n_timepoints=n_timepoints)  # type: ignore[arg-type]

        assert model.n_sample_calls == 0

    @pytest.mark.parametrize("n_timepoints", [0, -1])
    def test_rejects_non_positive_n_timepoints(self, n_timepoints: int) -> None:
        """Test the shared axis must contain at least one timepoint."""
        model = FixedAscertainmentModel("ascertainment", {"hospital": 0.2})

        with pytest.raises(ValueError, match="n_timepoints.*positive integer"):
            model.sample(n_timepoints=n_timepoints)

        assert model.n_sample_calls == 0

    def test_rejects_non_mapping_baseline_rates(self) -> None:
        """Test baseline sampling must return a mapping."""
        model = FixedAscertainmentModel("ascertainment", {"hospital": 0.2})
        model.values = [0.2]  # type: ignore[assignment]

        with pytest.raises(TypeError, match="baseline rates.*mapping"):
            model.sample(n_timepoints=4)

    @pytest.mark.parametrize(
        "values, expected_missing, expected_extra",
        [
            ({}, ("hospital",), ()),
            ({"hospital": 0.2, "ed": 0.3}, (), ("ed",)),
            ({"ed": 0.3}, ("hospital",), ("ed",)),
        ],
    )
    def test_rejects_missing_or_extra_baseline_signals(
        self,
        values: Mapping[str, ArrayLike],
        expected_missing: tuple[str, ...],
        expected_extra: tuple[str, ...],
    ) -> None:
        """Test sampled baseline keys must exactly match declared signals."""
        model = FixedAscertainmentModel("ascertainment", {"hospital": 0.2})
        model.values = dict(values)

        with pytest.raises(ValueError) as error:
            model.sample(n_timepoints=4)

        assert f"Missing: {expected_missing}" in str(error.value)
        assert f"Extra: {expected_extra}" in str(error.value)

    @pytest.mark.parametrize("value", [jnp.ones(1), jnp.ones((1, 1))])
    def test_rejects_non_scalar_baselines(self, value: ArrayLike) -> None:
        """Test baseline samplers cannot return vectors or trajectories."""
        model = FixedAscertainmentModel("ascertainment", {"hospital": value})

        with pytest.raises(
            ValueError,
            match=r"hospital.*baseline shape.*required shape.*\(\)",
        ):
            model.sample(n_timepoints=4)

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_fixed_baselines_allow_closed_interval_endpoints(
        self,
        value: float,
    ) -> None:
        """Test fixed scalar baselines may equal zero or one."""
        model = FixedAscertainmentModel("ascertainment", {"hospital": value})

        result = model.sample(n_timepoints=4)

        assert result["hospital"] == value

    @pytest.mark.parametrize("value", [-0.1, 1.1, jnp.nan, jnp.inf])
    def test_rejects_invalid_fixed_baselines(self, value: float) -> None:
        """Test fixed baselines must be finite probabilities."""
        model = FixedAscertainmentModel("ascertainment", {"hospital": value})

        with pytest.raises(ValueError, match=r"hospital.*finite.*\[0, 1\]"):
            model.sample(n_timepoints=4)

    @pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1, jnp.nan])
    def test_temporal_baselines_require_open_interval(
        self,
        value: float,
    ) -> None:
        """Test logit-transformed baselines must be strictly between bounds."""
        model = FixedAscertainmentModel(
            "ascertainment",
            {"hospital": value},
            temporal_processes={"hospital": DeterministicTemporalProcess()},
        )

        with pytest.raises(ValueError, match=r"hospital.*finite.*\(0, 1\)"):
            model.sample(n_timepoints=4)

    def test_traced_baseline_values_do_not_trigger_python_validation(self) -> None:
        """Test baseline value checks remain compatible with JAX tracing."""
        model = FixedAscertainmentModel("ascertainment", {"hospital": 0.2})

        @jax.jit
        def validate(value: ArrayLike) -> ArrayLike:
            """Validate and return a traced baseline value."""
            return model._validate_baseline_rates({"hospital": value})["hospital"]

        assert jnp.isnan(validate(jnp.nan))

    def test_validates_all_baselines_before_recording_or_temporal_sampling(
        self,
    ) -> None:
        """Test invalid baselines fail before deterministic or process sites."""
        process = DeterministicTemporalProcess()
        model = FixedAscertainmentModel(
            "ascertainment",
            {"hospital": 0.2, "ed": 1.1},
            temporal_processes={"hospital": process},
        )

        with numpyro.handlers.trace() as trace:
            with pytest.raises(ValueError, match="ed"):
                model.sample(n_timepoints=4)

        assert trace == {}
        assert process.sample_calls == []

    def test_samples_fixed_and_temporal_rates_in_signal_order(self) -> None:
        """Test fixed scalars and varying trajectories share one interface."""
        process = DeterministicTemporalProcess(increment=0.2)
        model = FixedAscertainmentModel(
            "ascertainment",
            {"hospital": 0.2, "ed": 0.3},
            temporal_processes={"ed": process},
        )

        with numpyro.handlers.trace() as trace:
            values = model.sample(n_timepoints=4, first_day_dow=3)

        assert tuple(values) == ("hospital", "ed")
        assert values["hospital"].shape == ()
        assert values["ed"].shape == (4,)
        assert set(trace) == {
            "ascertainment_baseline_hospital",
            "ascertainment_hospital",
            "ascertainment_baseline_ed",
            "ascertainment_ed",
        }
        assert process.sample_calls == [
            {
                "n_timepoints": 4,
                "initial_value": 0.0,
                "n_processes": 1,
                "name_prefix": "ascertainment_ed",
                "first_day_dow": 3,
            }
        ]
        expected_ed = jax.nn.sigmoid(logit(0.3) + 0.2 * jnp.arange(4))
        assert jnp.allclose(values["ed"], expected_ed)

    def test_rejects_malformed_temporal_output(self) -> None:
        """Test temporal output keeps its singleton process dimension."""
        model = FixedAscertainmentModel(
            "ascertainment",
            {"hospital": 0.2},
            temporal_processes={
                "hospital": DeterministicTemporalProcess(result=jnp.zeros(4))
            },
        )

        with pytest.raises(
            ValueError,
            match=r"ascertainment.*hospital.*\(4,\).*\(4, 1\)",
        ):
            model.sample(n_timepoints=4)

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


def _make_public_ascertainment(
    model_kind: str,
    temporal_processes: Mapping[str, TemporalProcess] | None,
) -> AscertainmentModel:
    """Construct a public ascertainment class with two scalar baselines."""
    if model_kind == "independent":
        return IndependentAscertainment(
            name="common",
            rate_rvs={
                "hospital": DeterministicVariable("common_ihr", 0.2),
                "ed": DeterministicVariable("common_iedr", 0.3),
            },
            temporal_processes=temporal_processes,
        )
    if model_kind == "joint":
        return JointAscertainment(
            name="common",
            signals=("hospital", "ed"),
            baseline_rates=jnp.array([0.2, 0.3]),
            scale_tril=jnp.eye(2) * 0.1,
            temporal_processes=temporal_processes,
        )
    if model_kind == "ratio":
        return RatioLinkedAscertainment(
            name="common",
            base_signal="ed",
            linked_signal="hospital",
            base_rate_rv=DeterministicVariable("common_iedr", 0.3),
            ratio_rv=DeterministicVariable("common_ratio", 2.0 / 3.0),
            temporal_processes=temporal_processes,
        )
    raise ValueError(f"Unknown model kind {model_kind!r}.")


class TestPublicAscertainmentBehavior:
    """Test common fixed and temporal behavior for every public class."""

    @pytest.mark.parametrize("model_kind", ["independent", "joint", "ratio"])
    @pytest.mark.parametrize(
        "varying_signals",
        [(), ("ed",), ("hospital", "ed")],
        ids=["fixed", "partial", "full"],
    )
    def test_fixed_partial_and_fully_varying_rates(
        self,
        model_kind: str,
        varying_signals: tuple[str, ...],
    ) -> None:
        """Test common output shapes and deterministic names across classes."""
        temporal_processes = {
            signal: DeterministicTemporalProcess(increment=0.1)
            for signal in varying_signals
        }
        ascertainment = _make_public_ascertainment(
            model_kind,
            temporal_processes or None,
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                values = ascertainment.sample(n_timepoints=4)

        assert set(trace) >= {
            "common_baseline_hospital",
            "common_baseline_ed",
            "common_hospital",
            "common_ed",
        }
        assert trace["common_baseline_hospital"]["value"].shape == ()
        assert trace["common_baseline_ed"]["value"].shape == ()
        assert trace["common_hospital"]["type"] == "deterministic"
        assert trace["common_ed"]["type"] == "deterministic"
        for signal in ("hospital", "ed"):
            expected_shape = (4,) if signal in varying_signals else ()
            assert values[signal].shape == expected_shape
            assert trace[f"common_{signal}"]["value"].shape == expected_shape


class TestIndependentAscertainmentValidation:
    """Test IndependentAscertainment constructor validation."""

    def test_requires_a_rate_rv_mapping(self) -> None:
        """Test rate_rvs must be supplied as a mapping."""
        with pytest.raises(TypeError, match="rate_rvs.*mapping"):
            IndependentAscertainment(
                name="ascertainment",
                rate_rvs=[],  # type: ignore[arg-type]
            )

    def test_requires_at_least_one_rate_rv(self) -> None:
        """Test at least one signal must be configured."""
        with pytest.raises(ValueError, match="rate_rvs.*non-empty"):
            IndependentAscertainment(name="ascertainment", rate_rvs={})

    @pytest.mark.parametrize("signal", ["", None])
    def test_requires_non_empty_string_signals(self, signal: object) -> None:
        """Test rate_rv keys must be non-empty strings."""
        with pytest.raises(ValueError, match="keys.*non-empty strings"):
            IndependentAscertainment(
                name="ascertainment",
                rate_rvs={signal: DeterministicVariable("rate", 0.2)},  # type: ignore[dict-item]
            )

    def test_requires_random_variable_values(self) -> None:
        """Test each baseline sampler must be a RandomVariable."""
        with pytest.raises(TypeError, match="hospital.*RandomVariable"):
            IndependentAscertainment(
                name="ascertainment",
                rate_rvs={"hospital": object()},  # type: ignore[dict-item]
            )

    def test_preserves_mapping_order_and_copies_rate_rvs(self) -> None:
        """Test insertion order defines signals without aliasing the input."""
        ed_rv = DeterministicVariable("iedr", 0.2)
        hospital_rv = DeterministicVariable("ihr", 0.1)
        rate_rvs: dict[str, RandomVariable] = {
            "ed": ed_rv,
            "hospital": hospital_rv,
        }

        ascertainment = IndependentAscertainment(
            name="ascertainment",
            rate_rvs=rate_rvs,
        )
        rate_rvs.clear()

        assert ascertainment.signals == ("ed", "hospital")
        assert ascertainment.rate_rvs == {
            "ed": ed_rv,
            "hospital": hospital_rv,
        }


class TestIndependentAscertainmentSampling:
    """Test IndependentAscertainment sampling behavior."""

    def test_calls_each_random_variable_once(self) -> None:
        """Test every configured baseline variable is sampled exactly once."""
        ed_rv = CountingVariable("iedr", 0.2)
        hospital_rv = CountingVariable("ihr", 0.1)
        ascertainment = IndependentAscertainment(
            name="ascertainment",
            rate_rvs={"ed": ed_rv, "hospital": hospital_rv},
        )

        with numpyro.handlers.trace() as trace:
            values = ascertainment.sample(n_timepoints=4)

        assert tuple(values) == ("ed", "hospital")
        assert ed_rv.n_calls == 1
        assert hospital_rv.n_calls == 1
        assert set(trace) == {
            "ascertainment_baseline_ed",
            "ascertainment_ed",
            "ascertainment_baseline_hospital",
            "ascertainment_hospital",
        }

    def test_one_signal_trace_retains_underlying_rv_name(self) -> None:
        """Test the user-named sample site and standard sites are recorded."""
        ascertainment = IndependentAscertainment(
            name="ascertainment",
            rate_rvs={"ed": DistributionalVariable("iedr", dist.Delta(0.2))},
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                values = ascertainment.sample(n_timepoints=4)

        assert ascertainment.signals == ("ed",)
        assert trace["iedr"]["type"] == "sample"
        assert trace["ascertainment_baseline_ed"]["type"] == "deterministic"
        assert trace["ascertainment_ed"]["type"] == "deterministic"
        assert jnp.array_equal(values["ed"], trace["iedr"]["value"])

    def test_supports_calendar_aligned_temporal_process(self) -> None:
        """Test independent rates can use calendar-aligned trajectories."""
        weekly = WeeklyTemporalProcess(
            DeterministicTemporalProcess(increment=1.0),
            start_dow=0,
        )
        ascertainment = IndependentAscertainment(
            name="ascertainment",
            rate_rvs={"hospital": DeterministicVariable("ihr", 0.2)},
            temporal_processes={"hospital": weekly},
        )

        values = ascertainment.sample(n_timepoints=10, first_day_dow=2)

        assert ascertainment.requires_calendar_anchor()
        assert jnp.allclose(values["hospital"][:5], 0.2)
        assert jnp.all(values["hospital"][5:] > 0.2)


class TestConcreteTemporalAscertainment:
    """Test temporal behavior supplied by concrete baseline relationships."""

    def test_joint_baselines_support_independent_trajectories(self) -> None:
        """Test joint baselines are sampled before separate deviations."""
        baseline_rates = jnp.array([0.2, 0.4])
        ascertainment = JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed"),
            baseline_rates=baseline_rates,
            scale_tril=jnp.eye(2),
            temporal_processes={
                "hospital": DeterministicTemporalProcess(increment=0.3),
                "ed": DeterministicTemporalProcess(increment=-0.2),
            },
        )

        with numpyro.handlers.substitute(
            data={"he_ascertainment_eta": logit(baseline_rates)}
        ):
            with numpyro.handlers.trace() as trace:
                values = ascertainment.sample(n_timepoints=4)

        assert trace["he_ascertainment_eta"]["type"] == "sample"
        assert trace["he_ascertainment_baseline_hospital"]["value"].shape == ()
        assert trace["he_ascertainment_baseline_ed"]["value"].shape == ()
        assert trace["he_ascertainment_hospital"]["value"].shape == (4,)
        assert trace["he_ascertainment_ed"]["value"].shape == (4,)
        assert jnp.allclose(values["hospital"][0], baseline_rates[0])
        assert jnp.allclose(values["ed"][0], baseline_rates[1])
        assert not jnp.allclose(
            values["hospital"][1:] / values["ed"][1:],
            baseline_rates[0] / baseline_rates[1],
        )

    def test_ratio_linked_baseline_relationship_is_not_pointwise(self) -> None:
        """Test temporal deviations need not preserve the baseline ratio."""
        ascertainment = RatioLinkedAscertainment(
            name="he_ascertainment",
            base_signal="ed",
            linked_signal="hospital",
            base_rate_rv=DistributionalVariable("iedr", dist.Delta(0.4)),
            ratio_rv=DistributionalVariable("ihr_rel_iedr", dist.Delta(0.5)),
            temporal_processes={
                "ed": DeterministicTemporalProcess(increment=-0.1),
                "hospital": DeterministicTemporalProcess(increment=0.2),
            },
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                values = ascertainment.sample(n_timepoints=4)

        assert trace["iedr"]["type"] == "sample"
        assert trace["ihr_rel_iedr"]["type"] == "sample"
        assert jnp.allclose(
            trace["he_ascertainment_baseline_ed"]["value"],
            0.4,
        )
        assert jnp.allclose(
            trace["he_ascertainment_baseline_hospital"]["value"],
            0.2,
        )
        trajectory_ratios = values["hospital"] / values["ed"]
        assert jnp.allclose(trajectory_ratios[0], 0.5)
        assert not jnp.allclose(trajectory_ratios[1:], 0.5)


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
                values = ascertainment.sample(n_timepoints=4)

        assert set(values) == {"hospital", "ed"}
        assert trace["he_ascertainment_eta"]["type"] == "sample"
        assert trace["he_ascertainment_eta"]["value"].shape == (2,)
        assert trace["he_ascertainment_baseline_hospital"]["type"] == ("deterministic")
        assert trace["he_ascertainment_baseline_ed"]["type"] == "deterministic"
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
            values = ascertainment.sample(n_timepoints=4)

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
            values = ascertainment.sample(n_timepoints=4)

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
            values = ascertainment.sample(n_timepoints=4)

        assert set(values) == {"ed_visits", "hospital"}
        assert jnp.array_equal(values["ed_visits"], jnp.array(0.2))
        assert jnp.array_equal(values["hospital"], jnp.array(0.1))
        assert trace["he_ascertainment_baseline_ed_visits"]["type"] == ("deterministic")
        assert trace["he_ascertainment_baseline_hospital"]["type"] == ("deterministic")
        assert trace["he_ascertainment_ed_visits"]["type"] == "deterministic"
        assert trace["he_ascertainment_hospital"]["type"] == "deterministic"
        assert jnp.allclose(
            trace["he_ascertainment_baseline_hospital"]["value"]
            / trace["he_ascertainment_baseline_ed_visits"]["value"],
            0.5,
        )
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
                values = ascertainment.sample(n_timepoints=4)

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
