# numpydoc ignore=GL08
"""
Unit tests for PopulationInfections.
"""

import re
from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    AR1,
    Infections,
    InfectionsWithFeedback,
    WeeklyTemporalProcess,
)
from pyrenew.latent.population_infections import PopulationInfections
from pyrenew.math import r_approx_from_R
from pyrenew.model import PyrenewBuilder
from pyrenew.randomvariable import DistributionalVariable
from pyrenew.time import MMWR_WEEK
from test.test_helpers import fixed_ar1, fixed_random_walk


class TestPopulationInfectionsSample:
    """Test sample method for PopulationInfections."""

    @pytest.mark.parametrize("parameterization", ["innovation", "state"])
    def test_ar1_uses_sampled_initial_rt(
        self,
        gen_int_rv: DeterministicPMF,
        parameterization: Literal["innovation", "state"],
    ) -> None:
        """The sampled initial log Rt sets the first state and infection growth."""
        process = PopulationInfections(
            name="population",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DistributionalVariable(
                "log_rt_time_0", dist.Normal(0, 0.5)
            ),
            single_rt_process=AR1(
                autoreg_rv=DeterministicVariable("autoreg", 0.9),
                innovation_sd_rv=DeterministicVariable("innovation_sd", 0.05),
                parameterization=parameterization,
            ),
            n_initialization_points=7,
        )
        trace = numpyro.handlers.trace(
            numpyro.handlers.seed(process.sample, rng_seed=42)
        ).get_trace(n_days_post_init=10)

        initial_log_rt = trace["log_rt_time_0"]["value"]
        assert trace["population::log_rt_single"]["value"][7, 0] == initial_log_rt
        assert "log_rt_single_init" not in trace
        assert "log_rt_single_init_decentered" not in trace
        initial_infections = trace["population::I0_init"]["value"]
        expected_growth = r_approx_from_R(jnp.exp(initial_log_rt), gen_int_rv(), 4)
        assert jnp.allclose(
            jnp.log(initial_infections[1:] / initial_infections[:-1]), expected_growth
        )

    def test_output_shapes(self, population_infections):
        """Test that output shapes are correct for a single-population model."""
        n_days_post_init = 30
        n_total = population_infections.n_initialization_points + n_days_post_init

        with numpyro.handlers.seed(rng_seed=42):
            inf_agg, inf_all = population_infections.sample(
                n_days_post_init=n_days_post_init,
            )

        assert inf_agg.shape == (n_total,)
        assert inf_all.shape == (n_total, 1)

    def test_aggregate_equals_single_subpop(self, population_infections):
        """Test that aggregate infections equal the single subpopulation column."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_agg, inf_all = population_infections.sample(
                n_days_post_init=30,
            )

        assert jnp.allclose(inf_agg, inf_all[:, 0], atol=1e-6)

    def test_infections_are_positive(self, population_infections):
        """Test that all infections are positive."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_agg, inf_all = population_infections.sample(
                n_days_post_init=30,
            )

        assert jnp.all(inf_agg > 0)
        assert jnp.all(inf_all > 0)

    def test_deterministic_sites_recorded(self, population_infections):
        """Test that expected numpyro deterministic sites are recorded."""
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                population_infections.sample(n_days_post_init=30)

        expected_sites = [
            "population::I0_init",
            "population::log_rt_single",
            "population::rt_single",
            "population::rt_single_adjusted",
            "population::infections_aggregate",
        ]
        for site in expected_sites:
            assert site in trace, f"Missing deterministic site: {site}"

    def test_rt_is_exp_of_log_rt(self, population_infections):
        """Test that recorded Rt equals exp of recorded log Rt."""
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                population_infections.sample(n_days_post_init=30)

        log_rt = trace["population::log_rt_single"]["value"]
        rt = trace["population::rt_single"]["value"]

        assert jnp.allclose(rt, jnp.exp(log_rt), atol=1e-6)

    def test_default_adjusted_rt_matches_rt_single(self, population_infections):
        """Plain renewal records adjusted Rt equal to raw Rt."""
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                population_infections.sample(n_days_post_init=30)

        rt = trace["population::rt_single"]["value"]
        rt_adjusted = trace["population::rt_single_adjusted"]["value"]

        assert rt_adjusted.shape == rt.shape
        assert jnp.allclose(rt_adjusted, rt, atol=1e-6)

    def test_default_fractions_used_when_none(self, population_infections):
        """Test that default fractions [1.0] are used when not provided."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_agg, inf_all = population_infections.sample(
                n_days_post_init=30,
                subpop_fractions=None,
            )

        assert inf_all.shape[1] == 1

    def test_explicit_fractions_one(self, population_infections):
        """Test that explicit fractions [1.0] produce same results as default."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_agg_default, inf_all_default = population_infections.sample(
                n_days_post_init=30,
            )

        with numpyro.handlers.seed(rng_seed=42):
            inf_agg_explicit, inf_all_explicit = population_infections.sample(
                n_days_post_init=30,
                subpop_fractions=jnp.array([1.0]),
            )

        assert jnp.allclose(inf_agg_default, inf_agg_explicit, atol=1e-6)
        assert jnp.allclose(inf_all_default, inf_all_explicit, atol=1e-6)

    def test_different_seeds_give_different_results(self, population_infections):
        """Test that different RNG seeds produce different trajectories."""
        with numpyro.handlers.seed(rng_seed=1):
            inf_agg_1, _ = population_infections.sample(n_days_post_init=30)

        with numpyro.handlers.seed(rng_seed=999):
            inf_agg_2, _ = population_infections.sample(n_days_post_init=30)

        assert not jnp.allclose(inf_agg_1, inf_agg_2)

    def test_custom_name_prefix(self, gen_int_rv):
        """Test that custom name prefix is used in deterministic sites."""
        process = PopulationInfections(
            name="my_infections",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
            single_rt_process=fixed_random_walk(innovation_sd=1.0),
            n_initialization_points=7,
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                process.sample(n_days_post_init=30)

        assert "my_infections::rt_single" in trace

    def test_first_day_dow_reaches_calendar_aligned_rt_process(self, gen_int_rv):
        """Weekly Rt receives the weekday shifted past initialization."""
        process = PopulationInfections(
            name="population",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
            single_rt_process=WeeklyTemporalProcess(
                fixed_ar1(autoreg=0.9, innovation_sd=0.05),
                start_dow=MMWR_WEEK,
            ),
            n_initialization_points=9,
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                process.sample(n_days_post_init=10, first_day_dow=3)

        log_rt = trace["population::log_rt_single"]["value"]
        weekly = trace["log_rt_single_weekly"]["value"]

        assert log_rt.shape == (19, 1)
        assert weekly.shape == (3, 1)
        assert jnp.allclose(log_rt[:10], weekly[0])
        assert jnp.allclose(log_rt[10:17], weekly[1])
        assert jnp.allclose(log_rt[17:], weekly[2])

    def test_wrong_rt_shape_raises(self, gen_int_rv, wrong_shape_temporal_process_cls):
        """Temporal processes must return daily-length single-process Rt."""
        process = PopulationInfections(
            name="population",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
            single_rt_process=wrong_shape_temporal_process_cls(jnp.zeros((16, 2))),
            n_initialization_points=7,
        )

        with pytest.raises(ValueError, match="single_rt_process must return shape"):
            process.sample(n_days_post_init=10)

    def test_feedback_lowers_adjusted_rt(
        self,
        population_infections_factory: Callable[..., PopulationInfections],
        infection_feedback_strength_rv: DeterministicVariable,
        infection_feedback_pmf_rv: DeterministicPMF,
    ) -> None:
        """Negative feedback strength lowers adjusted Rt after initialization."""
        process = population_infections_factory(
            infection_feedback_strength_rv=infection_feedback_strength_rv,
            infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                sample = process.sample(n_days_post_init=30)

        n_init = process.n_initialization_points
        n_total = n_init + 30
        rt = trace["population::rt_single"]["value"]
        rt_adjusted = trace["population::rt_single_adjusted"]["value"]
        assert sample.aggregate.shape == (n_total,)
        assert sample.all_subpops.shape == (n_total, 1)
        assert jnp.all(sample.aggregate > 0)
        assert jnp.all(sample.all_subpops > 0)
        assert jnp.array_equal(rt_adjusted[:n_init], rt[:n_init])
        assert jnp.all(rt_adjusted[n_init:] < rt[n_init:])

    def test_feedback_matches_infections_with_feedback(
        self,
        population_infections_factory: Callable[..., PopulationInfections],
        infection_feedback_strength_rv: DeterministicVariable,
        infection_feedback_pmf_rv: DeterministicPMF,
        gen_int_rv: DeterministicPMF,
    ) -> None:
        """Feedback gives the same infections and adjusted Rt as InfectionsWithFeedback."""
        process = population_infections_factory(
            infection_feedback_strength_rv=infection_feedback_strength_rv,
            infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                sample = process.sample(n_days_post_init=30)

        n_init = process.n_initialization_points
        reference = InfectionsWithFeedback(
            name="infections",
            infection_feedback_strength=infection_feedback_strength_rv,
            infection_feedback_pmf=infection_feedback_pmf_rv,
        ).sample(
            Rt=trace["population::rt_single"]["value"][n_init:, 0],
            I0=trace["population::I0_init"]["value"],
            gen_int=gen_int_rv(),
        )
        assert jnp.allclose(
            sample.aggregate[n_init:], reference.post_initialization_infections
        )
        assert jnp.allclose(
            trace["population::rt_single_adjusted"]["value"][n_init:, 0],
            reference.rt,
        )

    def test_zero_feedback_strength_matches_plain_renewal(
        self,
        population_infections_factory: Callable[..., PopulationInfections],
        infection_feedback_pmf_rv: DeterministicPMF,
    ) -> None:
        """Feedback with strength 0 gives the same results as no feedback."""
        plain = population_infections_factory()
        zero = population_infections_factory(
            infection_feedback_strength_rv=DeterministicVariable(
                "infection_feedback_strength", 0.0
            ),
            infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as plain_trace:
                plain_sample = plain.sample(n_days_post_init=30)
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as zero_trace:
                zero_sample = zero.sample(n_days_post_init=30)

        assert jnp.allclose(zero_sample.aggregate, plain_sample.aggregate)
        assert jnp.allclose(zero_sample.all_subpops, plain_sample.all_subpops)
        assert jnp.allclose(
            zero_trace["population::rt_single_adjusted"]["value"],
            plain_trace["population::rt_single_adjusted"]["value"],
        )


class TestPopulationInfectionsValidation:
    """Test validation of inputs."""

    def test_rejects_missing_I0_rv(self, gen_int_rv):
        """Test that None I0_rv is rejected."""
        with pytest.raises(ValueError, match="I0_rv is required"):
            PopulationInfections(
                name="population",
                gen_int_rv=gen_int_rv,
                I0_rv=None,
                log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
                single_rt_process=fixed_random_walk(innovation_sd=1.0),
                n_initialization_points=7,
            )

    def test_rejects_missing_log_rt_time_0_rv(self, gen_int_rv):
        """Test that None log_rt_time_0_rv is rejected."""
        with pytest.raises(ValueError, match="log_rt_time_0_rv is required"):
            PopulationInfections(
                name="population",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=None,
                single_rt_process=fixed_random_walk(innovation_sd=1.0),
                n_initialization_points=7,
            )

    def test_rejects_missing_single_rt_process(self, gen_int_rv):
        """Test that None single_rt_process is rejected."""
        with pytest.raises(ValueError, match="single_rt_process is required"):
            PopulationInfections(
                name="population",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
                single_rt_process=None,
                n_initialization_points=7,
            )

    def test_rejects_invalid_I0(self, gen_int_rv):
        """Test that invalid I0 values are rejected at construction."""
        with pytest.raises(ValueError, match="I0 must be positive"):
            PopulationInfections(
                name="population",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", -0.1),
                log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
                single_rt_process=fixed_random_walk(innovation_sd=1.0),
                n_initialization_points=7,
            )

    def test_rejects_I0_greater_than_one(self, gen_int_rv):
        """Test that I0 > 1 is rejected at construction."""
        with pytest.raises(ValueError, match="I0 must be <= 1"):
            PopulationInfections(
                name="population",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 1.5),
                log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
                single_rt_process=fixed_random_walk(innovation_sd=1.0),
                n_initialization_points=7,
            )

    def test_rejects_insufficient_n_initialization_points(self, gen_int_rv):
        """Test that n_initialization_points < gen_int length is rejected."""
        with pytest.raises(
            ValueError, match="n_initialization_points must be at least"
        ):
            PopulationInfections(
                name="population",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
                single_rt_process=fixed_random_walk(innovation_sd=1.0),
                n_initialization_points=2,
            )

    def test_rejects_fractions_not_summing_to_one(self, population_infections):
        """Test that fractions not summing to 1 raises error at sample time."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            with numpyro.handlers.seed(rng_seed=42):
                population_infections.sample(
                    n_days_post_init=30,
                    subpop_fractions=jnp.array([0.5]),
                )

    def test_rejects_multiple_subpop_fractions_even_if_sum_to_one(
        self, population_infections
    ):
        """Test that multi-element fractions are rejected for population infections."""
        with pytest.raises(
            ValueError,
            match="requires exactly one subpopulation with fraction \\[1.0\\]",
        ):
            with numpyro.handlers.seed(rng_seed=42):
                population_infections.sample(
                    n_days_post_init=30,
                    subpop_fractions=jnp.array([0.5, 0.5]),
                )

    def test_rejects_non_scalar_I0(self, gen_int_rv):
        """Test that vector-valued I0 is rejected with a clear error."""
        process = PopulationInfections(
            name="population",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", jnp.array([0.001, 0.002])),
            log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
            single_rt_process=fixed_random_walk(innovation_sd=1.0),
            n_initialization_points=7,
        )

        with pytest.raises(
            ValueError,
            match="requires I0_rv to return a scalar",
        ):
            with numpyro.handlers.seed(rng_seed=42):
                process.sample(n_days_post_init=30)

    def test_validate_passes(self, population_infections):
        """Test that validate() succeeds for a properly constructed instance."""
        population_infections.validate()

    def test_default_subpop_fractions(self, population_infections):
        """Test that default_subpop_fractions returns [1.0]."""
        fracs = population_infections.default_subpop_fractions()
        assert jnp.allclose(fracs, jnp.array([1.0]))

    @pytest.mark.parametrize("given", ["strength", "pmf"])
    def test_rejects_feedback_strength_without_pmf(
        self,
        population_infections_factory: Callable[..., PopulationInfections],
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
            population_infections_factory(**options)

    def test_rejects_feedback_pmf_length_mismatch(
        self,
        population_infections_factory: Callable[..., PopulationInfections],
        infection_feedback_strength_rv: DeterministicVariable,
    ) -> None:
        """A feedback PMF shorter than the generation interval is rejected at sample time."""
        process = population_infections_factory(
            infection_feedback_strength_rv=infection_feedback_strength_rv,
            infection_feedback_pmf_rv=DeterministicPMF(
                "infection_feedback_pmf", jnp.array([0.5, 0.3, 0.2])
            ),
        )

        with pytest.raises(ValueError, match="same length as the generation interval"):
            with numpyro.handlers.seed(rng_seed=42):
                process.sample(n_days_post_init=30)

    def test_rejects_unbroadcastable_feedback_strength(
        self,
        population_infections_factory: Callable[..., PopulationInfections],
        infection_feedback_pmf_rv: DeterministicPMF,
    ) -> None:
        """A feedback strength that does not broadcast to Rt names both shapes."""
        process = population_infections_factory(
            infection_feedback_strength_rv=DeterministicVariable(
                "infection_feedback_strength", jnp.ones(5)
            ),
            infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        )

        with pytest.raises(
            ValueError,
            match=re.escape("shape (5,)") + ".*" + re.escape("(30,)"),
        ):
            with numpyro.handlers.seed(rng_seed=42):
                process.sample(n_days_post_init=30)

    def test_rejects_infection_process_argument(
        self, gen_int_rv: DeterministicPMF
    ) -> None:
        """The removed infection_process argument is rejected, directly and at build()."""
        params = {
            "gen_int_rv": gen_int_rv,
            "I0_rv": DeterministicVariable("I0", 0.001),
            "log_rt_time_0_rv": DeterministicVariable("log_rt_time_0", 0.0),
            "single_rt_process": fixed_random_walk(innovation_sd=1.0),
            "infection_process": Infections(name="infections"),
        }
        with pytest.raises(TypeError, match="infection_process"):
            PopulationInfections(name="population", n_initialization_points=7, **params)

        builder = PyrenewBuilder()
        builder.configure_latent(PopulationInfections, **params)
        with pytest.raises(TypeError, match="infection_process"):
            builder.build()


class TestPopulationValidateAndPrepareI0:
    """Test _validate_and_prepare_I0 for PopulationInfections."""

    def test_accepts_valid_scalar(self, population_infections):
        """Test that a valid scalar I0 passes through unchanged."""
        pop = population_infections._parse_and_validate_fractions()
        I0 = jnp.array(0.01)
        result = population_infections._validate_and_prepare_I0(I0, pop)
        assert result.ndim == 0
        assert jnp.isclose(result, 0.01)

    def test_rejects_vector(self, population_infections):
        """Test that a vector I0 is rejected."""
        pop = population_infections._parse_and_validate_fractions()
        I0 = jnp.array([0.01, 0.02])
        with pytest.raises(ValueError, match="requires I0_rv to return a scalar"):
            population_infections._validate_and_prepare_I0(I0, pop)

    def test_rejects_negative(self, population_infections):
        """Test that negative I0 is rejected."""
        pop = population_infections._parse_and_validate_fractions()
        I0 = jnp.array(-0.01)
        with pytest.raises(ValueError, match="I0 must be positive"):
            population_infections._validate_and_prepare_I0(I0, pop)

    def test_rejects_greater_than_one(self, population_infections):
        """Test that I0 > 1 is rejected."""
        pop = population_infections._parse_and_validate_fractions()
        I0 = jnp.array(1.5)
        with pytest.raises(ValueError, match="I0 must be <= 1"):
            population_infections._validate_and_prepare_I0(I0, pop)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
