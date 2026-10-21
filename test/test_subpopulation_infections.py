# numpydoc ignore=GL08
"""
Unit tests for SubpopulationInfections.
"""

import re
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import Predictive

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    AR1,
    DifferencedAR1,
    Infections,
    InfectionsWithFeedback,
    SubpopulationInfections,
)
from pyrenew.math import r_approx_from_R
from pyrenew.model import PyrenewBuilder
from pyrenew.randomvariable import DistributionalVariable
from test.test_helpers import fixed_random_walk


class TestSubpopulationInfectionsSample:
    """Test sample method with population structure at sample time."""

    def test_jurisdiction_total_is_weighted_sum(self, subpopulation_infections):
        """Test that jurisdiction total equals weighted sum of subpopulations."""
        fractions = jnp.array([0.3, 0.25, 0.45])

        with numpyro.handlers.seed(rng_seed=42):
            inf_juris, inf_all = subpopulation_infections.sample(
                n_days_post_init=30,
                subpop_fractions=fractions,
            )

        expected = jnp.sum(inf_all * fractions[jnp.newaxis, :], axis=1)

        assert jnp.allclose(inf_juris, expected, atol=1e-6)

    def test_deviations_sum_to_zero(self, subpopulation_infections):
        """Test that subpopulation deviations sum to zero (identifiability)."""
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                subpopulation_infections.sample(
                    n_days_post_init=30,
                    subpop_fractions=jnp.array([0.3, 0.25, 0.45]),
                )

        deviations = trace["subpopulation::subpop_deviations"]["value"]
        deviation_sums = jnp.sum(deviations, axis=1)

        assert jnp.allclose(deviation_sums, 0.0, atol=1e-6)

    def test_default_adjusted_rt_matches_rt_subpop(self, subpopulation_infections):
        """Plain renewal records adjusted Rt equal to raw subpopulation Rt."""
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                subpopulation_infections.sample(
                    n_days_post_init=30,
                    subpop_fractions=jnp.array([0.3, 0.25, 0.45]),
                )

        rt = trace["subpopulation::rt_subpop"]["value"]
        rt_adjusted = trace["subpopulation::rt_subpop_adjusted"]["value"]

        assert rt_adjusted.shape == rt.shape
        assert jnp.allclose(rt_adjusted, rt, atol=1e-6)

    def test_infections_are_positive(self, subpopulation_infections):
        """Test that all infections are positive (epidemiological invariant)."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_juris, inf_all = subpopulation_infections.sample(
                n_days_post_init=30,
                subpop_fractions=jnp.array([0.3, 0.25, 0.45]),
            )

        assert jnp.all(inf_juris > 0)
        assert jnp.all(inf_all > 0)

    @pytest.mark.parametrize(
        "fractions",
        [
            jnp.array([1.0]),
            jnp.array([0.3, 0.25, 0.45]),
            jnp.array([0.10, 0.14, 0.21, 0.22, 0.07, 0.26]),
        ],
        ids=["K=1", "K=3", "K=6"],
    )
    def test_shape_and_positivity_across_subpop_counts(
        self, subpopulation_infections, fractions
    ):
        """Test correct shapes and positivity for varying numbers of subpops."""
        n_days_post_init = 30
        n_total = subpopulation_infections.n_initialization_points + n_days_post_init
        n_subpops = len(fractions)

        with numpyro.handlers.seed(rng_seed=42):
            inf_juris, inf_all = subpopulation_infections.sample(
                n_days_post_init=n_days_post_init,
                subpop_fractions=fractions,
            )

        assert inf_juris.shape == (n_total,)
        assert inf_all.shape == (n_total, n_subpops)
        assert jnp.all(inf_juris > 0)
        assert jnp.all(inf_all > 0)

        # Weighted sum property
        expected = jnp.sum(inf_all * fractions[jnp.newaxis, :], axis=1)
        assert jnp.allclose(inf_juris, expected, atol=1e-6)

    def test_wrong_baseline_rt_shape_raises(
        self, gen_int_rv, wrong_shape_temporal_process_cls
    ):
        """Baseline temporal process must return one daily baseline trajectory."""
        process = SubpopulationInfections(
            name="subpopulation",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
            baseline_rt_process=wrong_shape_temporal_process_cls(jnp.zeros((16, 2))),
            subpop_rt_deviation_process=fixed_random_walk(innovation_sd=1.0),
            n_initialization_points=7,
        )

        with pytest.raises(ValueError, match="baseline_rt_process must return shape"):
            process.sample(
                n_days_post_init=10,
                subpop_fractions=jnp.array([0.3, 0.25, 0.45]),
            )

    def test_wrong_subpop_deviation_shape_raises(
        self, gen_int_rv, wrong_shape_temporal_process_cls, constant_temporal_process
    ):
        """Deviation temporal process must return one trajectory per subpop."""
        process = SubpopulationInfections(
            name="subpopulation",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
            baseline_rt_process=constant_temporal_process,
            subpop_rt_deviation_process=wrong_shape_temporal_process_cls(
                jnp.zeros((17, 2))
            ),
            n_initialization_points=7,
        )

        with pytest.raises(
            ValueError, match="subpop_rt_deviation_process must return shape"
        ):
            process.sample(
                n_days_post_init=10,
                subpop_fractions=jnp.array([0.3, 0.25, 0.45]),
            )

    def test_feedback_lowers_adjusted_rt(
        self,
        subpopulation_infections_factory: Callable[..., SubpopulationInfections],
        infection_feedback_strength_rv: DeterministicVariable,
        infection_feedback_pmf_rv: DeterministicPMF,
        subpop_fractions: jnp.ndarray,
    ) -> None:
        """Negative feedback strength lowers adjusted Rt after initialization."""
        process = subpopulation_infections_factory(
            infection_feedback_strength_rv=infection_feedback_strength_rv,
            infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                sample = process.sample(
                    n_days_post_init=30,
                    subpop_fractions=subpop_fractions,
                )

        n_init = process.n_initialization_points
        n_total = n_init + 30
        rt = trace["subpopulation::rt_subpop"]["value"]
        rt_adjusted = trace["subpopulation::rt_subpop_adjusted"]["value"]
        assert sample.aggregate.shape == (n_total,)
        assert sample.all_subpops.shape == (n_total, len(subpop_fractions))
        assert jnp.all(sample.aggregate > 0)
        assert jnp.all(sample.all_subpops > 0)
        assert jnp.array_equal(rt_adjusted[:n_init], rt[:n_init])
        assert jnp.all(rt_adjusted[n_init:] < rt[n_init:])

    def test_feedback_matches_infections_with_feedback(
        self,
        subpopulation_infections_factory: Callable[..., SubpopulationInfections],
        infection_feedback_strength_rv: DeterministicVariable,
        infection_feedback_pmf_rv: DeterministicPMF,
        gen_int_rv: DeterministicPMF,
        subpop_fractions: jnp.ndarray,
    ) -> None:
        """Feedback gives the same infections and adjusted Rt as InfectionsWithFeedback."""
        process = subpopulation_infections_factory(
            infection_feedback_strength_rv=infection_feedback_strength_rv,
            infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                sample = process.sample(
                    n_days_post_init=30,
                    subpop_fractions=subpop_fractions,
                )

        n_init = process.n_initialization_points
        reference = InfectionsWithFeedback(
            name="infections",
            infection_feedback_strength=infection_feedback_strength_rv,
            infection_feedback_pmf=infection_feedback_pmf_rv,
        ).sample(
            Rt=trace["subpopulation::rt_subpop"]["value"][n_init:, :],
            I0=trace["subpopulation::I0_init_all_subpops"]["value"],
            gen_int=gen_int_rv(),
        )
        assert jnp.allclose(
            sample.all_subpops[n_init:], reference.post_initialization_infections
        )
        assert jnp.allclose(
            trace["subpopulation::rt_subpop_adjusted"]["value"][n_init:],
            reference.rt,
        )

    def test_zero_feedback_strength_matches_plain_renewal(
        self,
        subpopulation_infections_factory: Callable[..., SubpopulationInfections],
        infection_feedback_pmf_rv: DeterministicPMF,
        subpop_fractions: jnp.ndarray,
    ) -> None:
        """Feedback with strength 0 gives the same results as no feedback."""
        plain = subpopulation_infections_factory()
        zero = subpopulation_infections_factory(
            infection_feedback_strength_rv=DeterministicVariable(
                "infection_feedback_strength", 0.0
            ),
            infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as plain_trace:
                plain_sample = plain.sample(
                    n_days_post_init=30,
                    subpop_fractions=subpop_fractions,
                )
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as zero_trace:
                zero_sample = zero.sample(
                    n_days_post_init=30,
                    subpop_fractions=subpop_fractions,
                )

        assert jnp.allclose(zero_sample.aggregate, plain_sample.aggregate)
        assert jnp.allclose(zero_sample.all_subpops, plain_sample.all_subpops)
        assert jnp.allclose(
            zero_trace["subpopulation::rt_subpop_adjusted"]["value"],
            plain_trace["subpopulation::rt_subpop_adjusted"]["value"],
        )

    def test_per_subpopulation_feedback_strength(
        self,
        subpopulation_infections_factory: Callable[..., SubpopulationInfections],
        infection_feedback_pmf_rv: DeterministicPMF,
        subpop_fractions: jnp.ndarray,
    ) -> None:
        """A strength of shape (n_subpops,) applies each value to its subpopulation."""
        strengths = jnp.array([0.0, -100.0, -1000.0])
        per_subpop = subpopulation_infections_factory(
            infection_feedback_strength_rv=DeterministicVariable(
                "infection_feedback_strength", strengths
            ),
            infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        )

        with numpyro.handlers.seed(rng_seed=42):
            sample = per_subpop.sample(
                n_days_post_init=30,
                subpop_fractions=subpop_fractions,
            )

        for k, strength in enumerate(strengths):
            shared = subpopulation_infections_factory(
                infection_feedback_strength_rv=DeterministicVariable(
                    "infection_feedback_strength", strength
                ),
                infection_feedback_pmf_rv=infection_feedback_pmf_rv,
            )
            with numpyro.handlers.seed(rng_seed=42):
                reference = shared.sample(
                    n_days_post_init=30,
                    subpop_fractions=subpop_fractions,
                )
            assert jnp.allclose(sample.all_subpops[:, k], reference.all_subpops[:, k])


class TestSubpopulationInfectionsValidation:
    """Test validation of inputs."""

    def test_rejects_missing_I0_rv(self, gen_int_rv):
        """Test that None I0_rv is rejected."""
        with pytest.raises(ValueError, match="I0_rv is required"):
            SubpopulationInfections(
                name="subpopulation",
                gen_int_rv=gen_int_rv,
                I0_rv=None,
                log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=fixed_random_walk(innovation_sd=1.0),
                subpop_rt_deviation_process=fixed_random_walk(innovation_sd=1.0),
                n_initialization_points=7,
            )

    def test_rejects_missing_log_rt_time_0_rv(self, gen_int_rv):
        """Test that None log_rt_time_0_rv is rejected."""
        with pytest.raises(ValueError, match="log_rt_time_0_rv is required"):
            SubpopulationInfections(
                name="subpopulation",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=None,
                baseline_rt_process=fixed_random_walk(innovation_sd=1.0),
                subpop_rt_deviation_process=fixed_random_walk(innovation_sd=1.0),
                n_initialization_points=7,
            )

    def test_rejects_missing_baseline_rt_process(self, gen_int_rv):
        """Test that None baseline_rt_process is rejected."""
        with pytest.raises(ValueError, match="baseline_rt_process is required"):
            SubpopulationInfections(
                name="subpopulation",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=None,
                subpop_rt_deviation_process=fixed_random_walk(innovation_sd=1.0),
                n_initialization_points=7,
            )

    def test_rejects_missing_subpop_rt_deviation_process(self, gen_int_rv):
        """Test that None subpop_rt_deviation_process is rejected."""
        with pytest.raises(ValueError, match="subpop_rt_deviation_process is required"):
            SubpopulationInfections(
                name="subpopulation",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=fixed_random_walk(innovation_sd=1.0),
                subpop_rt_deviation_process=None,
                n_initialization_points=7,
            )

    def test_rejects_invalid_I0(self, gen_int_rv):
        """Test that invalid I0 values are rejected."""
        with pytest.raises(ValueError, match="I0 must be positive"):
            SubpopulationInfections(
                name="subpopulation",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", -0.1),
                log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=fixed_random_walk(innovation_sd=1.0),
                subpop_rt_deviation_process=fixed_random_walk(innovation_sd=1.0),
                n_initialization_points=7,
            )

    def test_rejects_insufficient_n_initialization_points(self, gen_int_rv):
        """Test that n_initialization_points < gen_int length is rejected."""
        with pytest.raises(
            ValueError, match="n_initialization_points must be at least"
        ):
            SubpopulationInfections(
                name="subpopulation",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=fixed_random_walk(innovation_sd=1.0),
                subpop_rt_deviation_process=fixed_random_walk(innovation_sd=1.0),
                n_initialization_points=2,
            )

    def test_rejects_fractions_not_summing_to_one(self, subpopulation_infections):
        """Test that fractions not summing to 1 raises error at sample time."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            with numpyro.handlers.seed(rng_seed=42):
                subpopulation_infections.sample(
                    n_days_post_init=30,
                    subpop_fractions=jnp.array([0.3, 0.25, 0.40]),
                )

    @pytest.mark.parametrize("given", ["strength", "pmf"])
    def test_rejects_feedback_strength_without_pmf(
        self,
        subpopulation_infections_factory: Callable[..., SubpopulationInfections],
        infection_feedback_strength_rv: DeterministicVariable,
        infection_feedback_pmf_rv: DeterministicPMF,
        given: str,
    ) -> None:
        """One infection feedback argument without the other is rejected."""
        options = (
            {"infection_feedback_strength_rv": infection_feedback_strength_rv}
            if given == "strength"
            else {"infection_feedback_pmf_rv": infection_feedback_pmf_rv}
        )
        with pytest.raises(ValueError, match="must be given together"):
            subpopulation_infections_factory(**options)

    def test_rejects_feedback_pmf_length_mismatch(
        self,
        subpopulation_infections_factory: Callable[..., SubpopulationInfections],
        infection_feedback_strength_rv: DeterministicVariable,
        subpop_fractions: jnp.ndarray,
    ) -> None:
        """A feedback PMF shorter than the generation interval is rejected at sample time."""
        process = subpopulation_infections_factory(
            infection_feedback_strength_rv=infection_feedback_strength_rv,
            infection_feedback_pmf_rv=DeterministicPMF(
                "infection_feedback_pmf", jnp.array([0.5, 0.3, 0.2])
            ),
        )

        with pytest.raises(ValueError, match="same length as the generation interval"):
            with numpyro.handlers.seed(rng_seed=42):
                process.sample(n_days_post_init=30, subpop_fractions=subpop_fractions)

    @pytest.mark.parametrize("strength_shape", [(2,), (30,)])
    def test_rejects_unbroadcastable_feedback_strength(
        self,
        subpopulation_infections_factory: Callable[..., SubpopulationInfections],
        infection_feedback_pmf_rv: DeterministicPMF,
        subpop_fractions: jnp.ndarray,
        strength_shape: tuple[int, ...],
    ) -> None:
        """A feedback strength that does not broadcast to Rt names both shapes."""
        process = subpopulation_infections_factory(
            infection_feedback_strength_rv=DeterministicVariable(
                "infection_feedback_strength", jnp.ones(strength_shape)
            ),
            infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        )

        with pytest.raises(
            ValueError,
            match=re.escape(f"shape {strength_shape}") + ".*" + re.escape("(30, 3)"),
        ):
            with numpyro.handlers.seed(rng_seed=42):
                process.sample(n_days_post_init=30, subpop_fractions=subpop_fractions)

    def test_rejects_infection_process_argument(
        self, gen_int_rv: DeterministicPMF
    ) -> None:
        """The removed infection_process argument is rejected, directly and at build()."""
        params = {
            "gen_int_rv": gen_int_rv,
            "I0_rv": DeterministicVariable("I0", 0.001),
            "log_rt_time_0_rv": DeterministicVariable("log_rt_time_0", 0.0),
            "baseline_rt_process": fixed_random_walk(innovation_sd=1.0),
            "subpop_rt_deviation_process": fixed_random_walk(innovation_sd=1.0),
            "infection_process": Infections(name="infections"),
        }
        with pytest.raises(TypeError, match="infection_process"):
            SubpopulationInfections(
                name="subpopulation", n_initialization_points=7, **params
            )

        builder = PyrenewBuilder()
        builder.configure_latent(SubpopulationInfections, **params)
        with pytest.raises(TypeError, match="infection_process"):
            builder.build()


class TestSubpopulationInfectionsPerSubpopI0:
    """Test per-subpopulation I0 values."""

    def test_per_subpop_I0_array(self, gen_int_rv):
        """Test with per-subpopulation I0 values and verify positivity."""
        process = SubpopulationInfections(
            name="subpopulation",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", jnp.array([0.001, 0.002, 0.0015])),
            log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
            baseline_rt_process=fixed_random_walk(innovation_sd=1.0),
            subpop_rt_deviation_process=fixed_random_walk(innovation_sd=1.0),
            n_initialization_points=7,
        )

        with numpyro.handlers.seed(rng_seed=42):
            inf_juris, inf_all = process.sample(
                n_days_post_init=30,
                subpop_fractions=jnp.array([0.3, 0.25, 0.45]),
            )

        n_total = process.n_initialization_points + 30

        assert inf_juris.shape == (n_total,)
        assert inf_all.shape == (n_total, 3)
        assert jnp.all(inf_juris > 0)
        assert jnp.all(inf_all > 0)


class TestSubpopulationInfectionsStartingRt:
    """Test scalar and per-subpopulation starting log Rt."""

    @pytest.mark.parametrize("parameterization", ["innovation", "state"])
    def test_differenced_ar1_preserves_starting_rt(
        self,
        gen_int_rv: DeterministicPMF,
        subpop_log_rt_time_0: jnp.ndarray,
        subpop_fractions: jnp.ndarray,
        parameterization: Literal["innovation", "state"],
    ) -> None:
        """DifferencedAR1 samples initial rates but preserves starting log Rt."""
        v = subpop_log_rt_time_0
        temporal_process = DifferencedAR1(
            autoreg_rv=DeterministicVariable("autoreg", 0.8),
            innovation_sd_rv=DeterministicVariable("innovation_sd", 0.05),
            parameterization=parameterization,
        )
        process = SubpopulationInfections(
            name="subpopulation",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", v),
            baseline_rt_process=temporal_process,
            subpop_rt_deviation_process=temporal_process,
            n_initialization_points=7,
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                process.sample(n_days_post_init=10, subpop_fractions=subpop_fractions)

        assert jnp.allclose(trace["subpopulation::rt_subpop"]["value"][0], jnp.exp(v))
        assert jnp.allclose(
            trace["subpopulation::rt_baseline"]["value"][0, 0], jnp.exp(jnp.mean(v))
        )
        assert jnp.allclose(
            trace["subpopulation::subpop_deviations"]["value"][0], v - jnp.mean(v)
        )

    @pytest.mark.parametrize("parameterization", ["innovation", "state"])
    @pytest.mark.parametrize("ar1_role", ["baseline", "deviations", "both"])
    def test_ar1_starting_rt_uses_draws_around_supplied_means(
        self,
        gen_int_rv: DeterministicPMF,
        subpop_log_rt_time_0: jnp.ndarray,
        subpop_fractions: jnp.ndarray,
        parameterization: Literal["innovation", "state"],
        ar1_role: Literal["baseline", "deviations", "both"],
    ) -> None:
        """AR1 initial draws determine centered starting Rt and infection growth."""
        v = subpop_log_rt_time_0
        autoreg, innovation_sd = 0.8, 0.05
        kwargs = {
            "autoreg_rv": DeterministicVariable("autoreg", autoreg),
            "innovation_sd_rv": DeterministicVariable("innovation_sd", innovation_sd),
        }
        ar1 = AR1(**kwargs, parameterization=parameterization)
        differenced_ar1 = DifferencedAR1(**kwargs, parameterization=parameterization)
        process = SubpopulationInfections(
            name="subpopulation",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", v),
            baseline_rt_process=ar1 if ar1_role != "deviations" else differenced_ar1,
            subpop_rt_deviation_process=ar1
            if ar1_role != "baseline"
            else differenced_ar1,
            n_initialization_points=7,
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                process.sample(n_days_post_init=10, subpop_fractions=subpop_fractions)

        baseline_start = jnp.mean(v)
        deviations_start = v - baseline_start
        stationary_sd = innovation_sd / jnp.sqrt(1 - autoreg**2)
        if ar1_role != "deviations":
            baseline_init = trace["log_rt_baseline_init"]
            assert jnp.allclose(baseline_init["fn"].mean, baseline_start)
            assert jnp.allclose(baseline_init["fn"].variance, stationary_sd**2)
            baseline_start = baseline_init["value"][0]
        if ar1_role != "baseline":
            deviations_init = trace["subpop_deviations_init"]
            assert jnp.allclose(deviations_init["fn"].mean, deviations_start)
            assert jnp.allclose(deviations_init["fn"].variance, stationary_sd**2)
            deviations_start = deviations_init["value"] - jnp.mean(
                deviations_init["value"]
            )

        expected_rt = jnp.exp(baseline_start + deviations_start)
        assert jnp.allclose(
            trace["subpopulation::rt_baseline"]["value"][0, 0], jnp.exp(baseline_start)
        )
        assert jnp.allclose(
            trace["subpopulation::subpop_deviations"]["value"][0], deviations_start
        )
        assert jnp.allclose(trace["subpopulation::rt_subpop"]["value"][0], expected_rt)
        assert not jnp.allclose(expected_rt, jnp.exp(v))

        initial_infections = trace["subpopulation::I0_init_all_subpops"]["value"]
        growth = jnp.log(initial_infections[1:] / initial_infections[:-1])
        expected_growth = r_approx_from_R(
            R=expected_rt, g=gen_int_rv(), n_newton_steps=4
        )
        assert jnp.allclose(growth, expected_growth[jnp.newaxis, :])

    def test_vector_log_rt_time_0_sets_starting_rt(
        self,
        random_walk_subpopulation_infections_factory: Callable[
            ..., SubpopulationInfections
        ],
        subpop_log_rt_time_0: jnp.ndarray,
        subpop_fractions: jnp.ndarray,
    ) -> None:
        """Per-subpopulation starting log Rt sets day-0 subpopulation and baseline Rt."""
        v = subpop_log_rt_time_0
        process = random_walk_subpopulation_infections_factory(
            DeterministicVariable("log_rt_time_0", v)
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                process.sample(n_days_post_init=30, subpop_fractions=subpop_fractions)

        assert jnp.allclose(trace["subpopulation::rt_subpop"]["value"][0], jnp.exp(v))
        assert jnp.allclose(
            trace["subpopulation::rt_baseline"]["value"][0, 0], jnp.exp(jnp.mean(v))
        )
        assert jnp.allclose(
            trace["subpopulation::subpop_deviations"]["value"][0], v - jnp.mean(v)
        )

    def test_equal_vector_log_rt_time_0_matches_scalar(
        self,
        random_walk_subpopulation_infections_factory: Callable[
            ..., SubpopulationInfections
        ],
        subpop_fractions: jnp.ndarray,
    ) -> None:
        """A vector of equal starting values gives the same results as the scalar."""
        scalar = random_walk_subpopulation_infections_factory(
            DeterministicVariable("log_rt_time_0", 0.3)
        )
        vector = random_walk_subpopulation_infections_factory(
            DeterministicVariable("log_rt_time_0", jnp.full(3, 0.3))
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as scalar_trace:
                scalar_sample = scalar.sample(
                    n_days_post_init=30, subpop_fractions=subpop_fractions
                )
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as vector_trace:
                vector_sample = vector.sample(
                    n_days_post_init=30, subpop_fractions=subpop_fractions
                )

        assert list(scalar_trace) == list(vector_trace)
        for name, site in scalar_trace.items():
            if site["type"] == "deterministic":
                assert jnp.allclose(site["value"], vector_trace[name]["value"])
        assert jnp.allclose(scalar_sample.aggregate, vector_sample.aggregate)
        assert jnp.allclose(scalar_sample.all_subpops, vector_sample.all_subpops)

    def test_vector_log_rt_time_0_gives_distinct_initial_growth(
        self,
        random_walk_subpopulation_infections_factory: Callable[
            ..., SubpopulationInfections
        ],
        subpop_log_rt_time_0: jnp.ndarray,
        subpop_fractions: jnp.ndarray,
        gen_int_rv: DeterministicPMF,
    ) -> None:
        """Each subpopulation's initial growth rate follows its own starting Rt."""
        v = subpop_log_rt_time_0
        process = random_walk_subpopulation_infections_factory(
            DeterministicVariable("log_rt_time_0", v)
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                process.sample(n_days_post_init=30, subpop_fractions=subpop_fractions)

        I0_all = trace["subpopulation::I0_init_all_subpops"]["value"]
        growth = jnp.log(I0_all[1:] / I0_all[:-1])
        growth_from_rt = r_approx_from_R(R=jnp.exp(v), g=gen_int_rv(), n_newton_steps=4)
        assert jnp.allclose(growth, growth_from_rt[jnp.newaxis, :])
        assert jnp.all(jnp.diff(growth[0]) > 0)

    @pytest.mark.parametrize("shape", [(1,), (2,), (4,), (3, 1), (1, 3)])
    def test_rejects_wrong_shape_log_rt_time_0(
        self,
        random_walk_subpopulation_infections_factory: Callable[
            ..., SubpopulationInfections
        ],
        subpop_fractions: jnp.ndarray,
        shape: tuple[int, ...],
    ) -> None:
        """A starting log Rt that is neither scalar nor one per subpopulation is rejected."""
        process = random_walk_subpopulation_infections_factory(
            DeterministicVariable("log_rt_time_0", jnp.zeros(shape))
        )

        with pytest.raises(ValueError) as excinfo:
            with numpyro.handlers.seed(rng_seed=42):
                process.sample(n_days_post_init=30, subpop_fractions=subpop_fractions)

        message = str(excinfo.value)
        assert "log_rt_time_0_rv" in message
        assert "(3,)" in message
        assert str(shape) in message

    def test_vector_log_rt_time_0_under_predictive(
        self,
        random_walk_subpopulation_infections_factory: Callable[
            ..., SubpopulationInfections
        ],
        subpop_fractions: jnp.ndarray,
    ) -> None:
        """Each Predictive draw starts at its own per-subpopulation starting Rt."""
        process = random_walk_subpopulation_infections_factory(
            DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5).expand([3]))
        )

        def sampler() -> None:
            """Sample the latent process for Predictive."""
            process.sample(n_days_post_init=30, subpop_fractions=subpop_fractions)

        samples = Predictive(sampler, num_samples=10)(jax.random.PRNGKey(0))

        assert samples["log_rt_time_0"].shape == (10, 3)
        assert jnp.allclose(
            samples["subpopulation::rt_subpop"][:, 0, :],
            jnp.exp(samples["log_rt_time_0"]),
        )


class TestSubpopulationValidateAndPrepareI0:
    """Test _validate_and_prepare_I0 for SubpopulationInfections."""

    def test_broadcasts_scalar_to_subpop_array(self, subpopulation_infections):
        """Test that scalar I0 is broadcast to per-subpopulation array."""
        pop = subpopulation_infections._parse_and_validate_fractions(
            subpop_fractions=jnp.array([0.3, 0.25, 0.45])
        )
        I0 = jnp.array(0.01)
        result = subpopulation_infections._validate_and_prepare_I0(I0, pop)
        assert result.shape == (3,)
        assert jnp.allclose(result, 0.01)

    def test_passes_through_matching_array(self, subpopulation_infections):
        """Test that a per-subpopulation I0 array passes through unchanged."""
        pop = subpopulation_infections._parse_and_validate_fractions(
            subpop_fractions=jnp.array([0.3, 0.25, 0.45])
        )
        I0 = jnp.array([0.001, 0.002, 0.0015])
        result = subpopulation_infections._validate_and_prepare_I0(I0, pop)
        assert result.shape == (3,)
        assert jnp.allclose(result, I0)

    def test_rejects_negative(self, subpopulation_infections):
        """Test that negative I0 is rejected."""
        pop = subpopulation_infections._parse_and_validate_fractions(
            subpop_fractions=jnp.array([0.5, 0.5])
        )
        I0 = jnp.array(-0.01)
        with pytest.raises(ValueError, match="I0 must be positive"):
            subpopulation_infections._validate_and_prepare_I0(I0, pop)

    def test_rejects_greater_than_one(self, subpopulation_infections):
        """Test that I0 > 1 is rejected."""
        pop = subpopulation_infections._parse_and_validate_fractions(
            subpop_fractions=jnp.array([0.5, 0.5])
        )
        I0 = jnp.array(1.5)
        with pytest.raises(ValueError, match="I0 must be <= 1"):
            subpopulation_infections._validate_and_prepare_I0(I0, pop)

    @pytest.mark.parametrize("shape", [(1,), (2,), (4,), (3, 1), (1, 3)])
    def test_rejects_wrong_shape_I0(
        self,
        subpopulation_infections: SubpopulationInfections,
        subpop_fractions: jnp.ndarray,
        shape: tuple[int, ...],
    ) -> None:
        """An I0 that is neither scalar nor one per subpopulation is rejected."""
        pop = subpopulation_infections._parse_and_validate_fractions(
            subpop_fractions=subpop_fractions
        )

        with pytest.raises(ValueError) as excinfo:
            subpopulation_infections._validate_and_prepare_I0(
                jnp.full(shape, 0.001), pop
            )

        message = str(excinfo.value)
        assert "I0_rv" in message
        assert "(3,)" in message
        assert str(shape) in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
