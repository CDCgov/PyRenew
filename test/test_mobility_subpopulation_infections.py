"""
Unit tests for MobilitySubpopulationInfections.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpyro.infer import Predictive

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    LatentSample,
    MobilitySubpopulationInfections,
    SubpopulationInfections,
)
from pyrenew.latent.base import BaseLatentInfectionProcess
from pyrenew.metaclass import RandomVariable
from pyrenew.model import MultiSignalModel
from pyrenew.randomvariable import DistributionalVariable
from test.test_helpers import fixed_random_walk


def _sample_with_trace(
    process: BaseLatentInfectionProcess,
    subpop_fractions: jnp.ndarray,
) -> tuple[dict, LatentSample]:
    """
    Sample a latent process for 30 days with a fixed seed, recording its sites.

    Parameters
    ----------
    process
        Latent infection process to sample.
    subpop_fractions
        Population fractions for all subpopulations.

    Returns
    -------
    tuple[dict, LatentSample]
        The NumPyro trace and the sampled latent infections.
    """
    with numpyro.handlers.seed(rng_seed=42):
        with numpyro.handlers.trace() as trace:
            latent = process.sample(
                n_days_post_init=30, subpop_fractions=subpop_fractions
            )
    return trace, latent


class TestMobilitySample:
    """Test sampling the mobility-coupled latent process."""

    @pytest.mark.parametrize("per_region", [False, True])
    def test_identity_without_depletion_matches_subpopulation_infections(
        self,
        per_region: bool,
        mobility_subpopulation_infections_factory: Callable[
            ..., MobilitySubpopulationInfections
        ],
        random_walk_subpopulation_infections_factory: Callable[
            [RandomVariable], SubpopulationInfections
        ],
        roy_subpop_fractions: jnp.ndarray,
        subpop_log_rt_time_0: jnp.ndarray,
    ) -> None:
        """Identity matrix without depletion records what SubpopulationInfections records."""
        log_rt_time_0 = subpop_log_rt_time_0 if per_region else 0.0
        uncoupled = random_walk_subpopulation_infections_factory(
            DeterministicVariable("log_rt_time_0", log_rt_time_0)
        )
        coupled = mobility_subpopulation_infections_factory(
            jnp.eye(3),
            DeterministicVariable("log_rt_time_0", log_rt_time_0),
            susceptible_depletion=False,
        )

        trace_uncoupled, latent_uncoupled = _sample_with_trace(
            uncoupled, roy_subpop_fractions
        )
        trace_coupled, latent_coupled = _sample_with_trace(
            coupled, roy_subpop_fractions
        )

        for site_name, site in trace_uncoupled.items():
            assert_allclose(
                trace_coupled[site_name]["value"], site["value"], rtol=1e-12
            )
        assert_allclose(
            latent_coupled.aggregate, latent_uncoupled.aggregate, rtol=1e-12
        )
        assert_allclose(
            latent_coupled.all_subpops, latent_uncoupled.all_subpops, rtol=1e-12
        )

    def test_aggregate_is_weighted_sum(
        self,
        mobility_subpopulation_infections: MobilitySubpopulationInfections,
        roy_subpop_fractions: jnp.ndarray,
    ) -> None:
        """Aggregate infections are the population-weighted sum over regions."""
        _, latent = _sample_with_trace(
            mobility_subpopulation_infections, roy_subpop_fractions
        )

        assert_allclose(
            latent.aggregate, latent.all_subpops @ roy_subpop_fractions, rtol=1e-12
        )

    @pytest.mark.parametrize("susceptible_depletion", [True, False])
    def test_records_susceptible_fraction(
        self,
        susceptible_depletion: bool,
        mobility_subpopulation_infections_factory: Callable[
            ..., MobilitySubpopulationInfections
        ],
        roy_mobility_matrix: jnp.ndarray,
        roy_subpop_fractions: jnp.ndarray,
    ) -> None:
        """The susceptible fraction is recorded on the full axis, starts at 1, and never increases."""
        process = mobility_subpopulation_infections_factory(
            roy_mobility_matrix,
            DeterministicVariable("log_rt_time_0", 0.0),
            susceptible_depletion=susceptible_depletion,
        )
        trace, latent = _sample_with_trace(process, roy_subpop_fractions)
        susceptible_fraction = trace["subpopulation::susceptible_fraction"]["value"]

        assert susceptible_fraction.shape == (process.n_initialization_points + 30, 3)
        assert jnp.all(susceptible_fraction[0] == 1.0)
        assert jnp.all(jnp.diff(susceptible_fraction, axis=0) <= 0.0)
        assert_allclose(
            susceptible_fraction[1:],
            1.0 - jnp.cumsum(latent.all_subpops, axis=0)[:-1],
            atol=1e-12,
        )

    def test_adjusted_rt_uses_susceptible_fraction(
        self,
        mobility_subpopulation_infections: MobilitySubpopulationInfections,
        roy_mobility_matrix: jnp.ndarray,
        roy_subpop_fractions: jnp.ndarray,
    ) -> None:
        """With depletion, adjusted Rt is raw Rt times the susceptible share present."""
        trace, _ = _sample_with_trace(
            mobility_subpopulation_infections, roy_subpop_fractions
        )
        n_init = mobility_subpopulation_infections.n_initialization_points
        susceptible_fraction = trace["subpopulation::susceptible_fraction"]["value"]
        rt_subpop = trace["subpopulation::rt_subpop"]["value"]
        rt_subpop_adjusted = trace["subpopulation::rt_subpop_adjusted"]["value"]
        share_present = (
            roy_mobility_matrix
            * roy_subpop_fractions[jnp.newaxis, :]
            / (roy_mobility_matrix @ roy_subpop_fractions)[:, jnp.newaxis]
        )

        assert rt_subpop.shape == rt_subpop_adjusted.shape == (30, 3)
        assert_allclose(
            rt_subpop_adjusted,
            rt_subpop * (susceptible_fraction[n_init:] @ share_present.T),
            rtol=1e-10,
        )

    def test_susceptible_fraction_without_depletion_is_not_floored(
        self,
        mobility_subpopulation_infections: MobilitySubpopulationInfections,
        roy_subpop_fractions: jnp.ndarray,
    ) -> None:
        """Without depletion, the recorded fraction shows cumulative overshoot."""
        process = mobility_subpopulation_infections
        process.susceptible_depletion = False
        process.I0_rv = DeterministicVariable("I0", 0.2)
        trace, latent = _sample_with_trace(process, roy_subpop_fractions)
        susceptible = trace["subpopulation::susceptible_fraction"]["value"]

        assert jnp.any(susceptible < 0.0)
        assert_allclose(
            susceptible[1:],
            1.0 - jnp.cumsum(latent.all_subpops, axis=0)[:-1],
        )

    def test_adjusted_rt_without_depletion(
        self,
        mobility_subpopulation_infections_factory: Callable[
            ..., MobilitySubpopulationInfections
        ],
        roy_mobility_matrix: jnp.ndarray,
        roy_subpop_fractions: jnp.ndarray,
    ) -> None:
        """Without depletion, adjusted Rt equals raw Rt."""
        process = mobility_subpopulation_infections_factory(
            roy_mobility_matrix,
            DeterministicVariable("log_rt_time_0", 0.0),
            susceptible_depletion=False,
        )
        trace, _ = _sample_with_trace(process, roy_subpop_fractions)

        assert_array_equal(
            trace["subpopulation::rt_subpop_adjusted"]["value"],
            trace["subpopulation::rt_subpop"]["value"],
        )


class TestMobilityValidation:
    """Test validation of the mobility matrix and its fit to the population."""

    def test_rejects_excessive_initialization(
        self,
        mobility_subpopulation_infections: MobilitySubpopulationInfections,
        roy_subpop_fractions: jnp.ndarray,
    ) -> None:
        """Sampling rejects initialization whose total exceeds a region's population."""
        mobility_subpopulation_infections.I0_rv = DeterministicVariable("I0", 0.2)
        with pytest.raises(ValueError, match="<= 1 in every region"):
            _sample_with_trace(mobility_subpopulation_infections, roy_subpop_fractions)

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("non_square", "square"),
            ("three_dimensional", "square"),
            ("negative", "non-negative"),
            ("nan", "finite"),
            ("row_stochastic", "transpose"),
            ("columns_not_one", "sum to 1"),
        ],
    )
    def test_rejects_invalid_matrix(
        self,
        case: str,
        match: str,
        mobility_subpopulation_infections_factory: Callable[
            ..., MobilitySubpopulationInfections
        ],
        roy_mobility_matrix: jnp.ndarray,
    ) -> None:
        """Invalid matrices are rejected at construction with a message naming the problem."""
        matrices = {
            "non_square": jnp.ones((2, 3)) / 2,
            "three_dimensional": jnp.stack([roy_mobility_matrix, roy_mobility_matrix]),
            "negative": jnp.array([[1.1, 0.0, 0.0], [-0.1, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "nan": roy_mobility_matrix.at[0, 0].set(jnp.nan),
            "row_stochastic": roy_mobility_matrix.T,
            "columns_not_one": 0.5 * jnp.eye(3),
        }

        with pytest.raises(ValueError, match=match):
            mobility_subpopulation_infections_factory(
                matrices[case], DeterministicVariable("log_rt_time_0", 0.0)
            )

    def test_column_sum_message_without_transpose(
        self,
        mobility_subpopulation_infections_factory: Callable[
            ..., MobilitySubpopulationInfections
        ],
    ) -> None:
        """The transpose hint is given only when the rows sum to 1."""
        with pytest.raises(ValueError) as excinfo:
            mobility_subpopulation_infections_factory(
                0.5 * jnp.eye(3), DeterministicVariable("log_rt_time_0", 0.0)
            )

        assert "transpose" not in str(excinfo.value)

    def test_rejects_random_matrix_rv(self, gen_int_rv: DeterministicPMF) -> None:
        """A random mobility matrix is rejected."""
        with pytest.raises(TypeError, match="DeterministicVariable"):
            MobilitySubpopulationInfections(
                name="subpopulation",
                gen_int_rv=gen_int_rv,
                n_initialization_points=7,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
                baseline_rt_process=fixed_random_walk(innovation_sd=0.05),
                subpop_rt_deviation_process=fixed_random_walk(innovation_sd=0.025),
                mobility_matrix_rv=DistributionalVariable(
                    "mobility_matrix", dist.Dirichlet(jnp.ones(3)).expand([3])
                ),
            )

    def test_rejects_feedback_arguments(
        self,
        gen_int_rv: DeterministicPMF,
        roy_mobility_matrix: jnp.ndarray,
    ) -> None:
        """The class has no infection feedback option."""
        with pytest.raises(TypeError, match="infection_feedback_strength_rv"):
            MobilitySubpopulationInfections(
                name="subpopulation",
                gen_int_rv=gen_int_rv,
                n_initialization_points=7,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
                baseline_rt_process=fixed_random_walk(innovation_sd=0.05),
                subpop_rt_deviation_process=fixed_random_walk(innovation_sd=0.025),
                mobility_matrix_rv=DeterministicVariable(
                    "mobility_matrix", roy_mobility_matrix
                ),
                infection_feedback_strength_rv=DeterministicVariable("strength", -1.0),
            )

    def test_rejects_size_mismatch(
        self,
        mobility_subpopulation_infections: MobilitySubpopulationInfections,
    ) -> None:
        """A matrix that does not match the number of subpopulations is rejected."""
        with pytest.raises(ValueError) as excinfo:
            mobility_subpopulation_infections._parse_and_validate_fractions(
                jnp.array([0.5, 0.5])
            )

        message = str(excinfo.value)
        assert "(3, 3)" in message
        assert "2 entries" in message

    def test_accepts_location_without_population(
        self,
        empty_location_mobility: tuple[jnp.ndarray, jnp.ndarray],
        mobility_subpopulation_infections_factory: Callable[
            ..., MobilitySubpopulationInfections
        ],
    ) -> None:
        """A location with no one present samples finite infections and adjusted Rt 0 there."""
        mobility_matrix, subpop_fractions = empty_location_mobility
        process = mobility_subpopulation_infections_factory(
            mobility_matrix, DeterministicVariable("log_rt_time_0", 0.0)
        )
        trace, latent = _sample_with_trace(process, subpop_fractions)
        rt_subpop_adjusted = trace["subpopulation::rt_subpop_adjusted"]["value"]

        assert jnp.all(jnp.isfinite(latent.all_subpops))
        assert_array_equal(rt_subpop_adjusted[:, 2], 0.0)


class TestMobilityModel:
    """Test the mobility latent process inside a builder-constructed model."""

    def test_builder_prior_predictive(
        self,
        mobility_model_factory: Callable[[jnp.ndarray], MultiSignalModel],
        roy_mobility_matrix: jnp.ndarray,
        mobility_model_args: dict[str, object],
    ) -> None:
        """Prior predictive sampling through PyrenewBuilder gives the expected shapes."""
        model = mobility_model_factory(roy_mobility_matrix)
        samples = Predictive(model.sample, num_samples=5)(
            jax.random.PRNGKey(0), **mobility_model_args
        )
        n_total = model.latent.n_initialization_points + 30

        assert samples["latent_infections_by_subpop"].shape == (5, n_total, 3)
        assert jnp.all(samples["latent_infections_by_subpop"] > 0)
        assert samples["subpopulation::susceptible_fraction"].shape == (5, n_total, 3)

    def test_validate_data_rejects_before_mcmc(
        self,
        mobility_model_factory: Callable[[jnp.ndarray], MultiSignalModel],
        roy_mobility_matrix: jnp.ndarray,
    ) -> None:
        """A matrix that does not match the number of subpopulations is rejected before MCMC starts."""
        model = mobility_model_factory(roy_mobility_matrix)
        fractions = jnp.array([0.5, 0.5])

        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            model.validate_data(n_days_post_init=30, subpop_fractions=fractions)
        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            model.run(
                num_warmup=10,
                num_samples=10,
                n_days_post_init=30,
                population_size=45_000_000,
                subpop_fractions=fractions,
                hospital_subpop={
                    "obs": None,
                    "subpop_indices": jnp.arange(fractions.size),
                },
            )
        assert model.mcmc is None

    def test_counterfactual_with_another_matrix(
        self,
        mobility_model_factory: Callable[[jnp.ndarray], MultiSignalModel],
        roy_mobility_matrix: jnp.ndarray,
        mobility_model_args: dict[str, object],
    ) -> None:
        """Draws replayed through a model with another matrix give different infections."""
        model = mobility_model_factory(roy_mobility_matrix)
        identity_model = mobility_model_factory(jnp.eye(3))
        draws = Predictive(model.sample, num_samples=4)(
            jax.random.PRNGKey(0), **mobility_model_args
        )

        replay = Predictive(model.sample, posterior_samples=draws)(
            jax.random.PRNGKey(1), **mobility_model_args
        )
        counterfactual = Predictive(identity_model.sample, posterior_samples=draws)(
            jax.random.PRNGKey(1), **mobility_model_args
        )

        assert_allclose(
            replay["latent_infections_by_subpop"],
            draws["latent_infections_by_subpop"],
            rtol=1e-12,
        )
        assert not jnp.allclose(
            counterfactual["latent_infections_by_subpop"],
            draws["latent_infections_by_subpop"],
        )
