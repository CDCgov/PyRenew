"""
Tests for PyrenewBuilder and MultiSignalModel.
"""

import jax.numpy as jnp
import numpyro
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import HierarchicalInfections, RandomWalk
from pyrenew.model import MultiSignalModel, PyrenewBuilder
from pyrenew.observation import Counts, CountsBySubpop, NegativeBinomialNoise

# Standard population structure for tests (3 subpopulations)
SUBPOP_FRACTIONS = jnp.array([0.3, 0.25, 0.45])


@pytest.fixture
def simple_builder():
    """
    Create a configured builder (no population structure at configure time).

    Returns
    -------
    PyrenewBuilder
        Configured model builder.
    """
    builder = PyrenewBuilder()
    gen_int = DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))

    builder.configure_latent(
        HierarchicalInfections,
        gen_int_rv=gen_int,
        I0_rv=DeterministicVariable("I0", 0.001),
        initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
        baseline_rt_process=RandomWalk(),
        subpop_rt_deviation_process=RandomWalk(),
    )

    delay = DeterministicPMF("delay", jnp.array([0.1, 0.3, 0.4, 0.2]))
    obs = Counts(
        name="hospital",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=delay,
        noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
    )
    builder.add_observation(obs)

    return builder


@pytest.fixture
def validation_builder():
    """
    Create a builder with both aggregate and subpop observations.

    Used for testing validate_data() delegation to different
    observation types.

    Returns
    -------
    PyrenewBuilder
        Builder with Counts ("hospital") and CountsBySubpop
        ("hospital_subpop") observations.
    """
    builder = PyrenewBuilder()
    gen_int = DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))

    builder.configure_latent(
        HierarchicalInfections,
        gen_int_rv=gen_int,
        I0_rv=DeterministicVariable("I0", 0.001),
        initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
        baseline_rt_process=RandomWalk(),
        subpop_rt_deviation_process=RandomWalk(),
    )

    delay = DeterministicPMF("delay", jnp.array([0.1, 0.3, 0.4, 0.2]))
    builder.add_observation(
        Counts(
            name="hospital",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=delay,
            noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
        )
    )
    builder.add_observation(
        CountsBySubpop(
            name="hospital_subpop",
            ascertainment_rate_rv=DeterministicVariable("ihr_subpop", 0.01),
            delay_distribution_rv=delay,
            noise=NegativeBinomialNoise(DeterministicVariable("conc_subpop", 10.0)),
        )
    )

    return builder


class TestPyrenewBuilderConfiguration:
    """Test PyrenewBuilder configuration."""

    def test_rejects_population_structure_at_configure_time(self):
        """Test that population structure params are rejected at configure time."""
        builder = PyrenewBuilder()
        gen_int = DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))

        with pytest.raises(ValueError, match="Do not specify"):
            builder.configure_latent(
                HierarchicalInfections,
                gen_int_rv=gen_int,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=RandomWalk(),
                subpop_rt_deviation_process=RandomWalk(),
                subpop_fractions=jnp.array([0.5, 0.5]),  # Should fail
            )

    def test_rejects_n_initialization_points_at_configure_time(self):
        """Test that n_initialization_points is rejected at configure time."""
        builder = PyrenewBuilder()
        gen_int = DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))

        with pytest.raises(ValueError, match="Do not specify n_initialization_points"):
            builder.configure_latent(
                HierarchicalInfections,
                gen_int_rv=gen_int,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=RandomWalk(),
                subpop_rt_deviation_process=RandomWalk(),
                n_initialization_points=10,  # Should fail
            )

    def test_rejects_reconfiguring_latent(self):
        """Test that configuring latent twice raises RuntimeError."""
        builder = PyrenewBuilder()
        gen_int = DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))

        builder.configure_latent(
            HierarchicalInfections,
            gen_int_rv=gen_int,
            I0_rv=DeterministicVariable("I0", 0.001),
            initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
            baseline_rt_process=RandomWalk(),
            subpop_rt_deviation_process=RandomWalk(),
        )

        with pytest.raises(RuntimeError, match="already configured"):
            builder.configure_latent(
                HierarchicalInfections,
                gen_int_rv=gen_int,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=RandomWalk(),
                subpop_rt_deviation_process=RandomWalk(),
            )

    def test_rejects_duplicate_observation_name(self, simple_builder):
        """Test that adding duplicate observation name raises ValueError."""
        delay = DeterministicPMF("delay2", jnp.array([0.5, 0.5]))
        obs = Counts(
            name="hospital",  # Same name as existing observation
            ascertainment_rate_rv=DeterministicVariable("ihr2", 0.02),
            delay_distribution_rv=delay,
            noise=NegativeBinomialNoise(DeterministicVariable("conc2", 20.0)),
        )

        with pytest.raises(ValueError, match="already added"):
            simple_builder.add_observation(obs)

    def test_build_creates_model(self, simple_builder):
        """Test that build() creates a MultiSignalModel."""
        model = simple_builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_build_without_latent_raises_error(self):
        """Test that build() without configure_latent raises ValueError."""
        builder = PyrenewBuilder()

        with pytest.raises(ValueError, match="Must call configure_latent"):
            builder.build()

    def test_compute_n_initialization_points_without_latent_raises(self):
        """Test that compute_n_initialization_points without latent raises."""
        builder = PyrenewBuilder()

        with pytest.raises(ValueError, match="Must call configure_latent"):
            builder.compute_n_initialization_points()

    def test_compute_n_initialization_points_without_gen_int_raises(self):
        """Test that compute_n_initialization_points without gen_int_rv raises."""
        builder = PyrenewBuilder()
        # Configure latent without gen_int_rv
        builder.latent_class = HierarchicalInfections
        builder.latent_params = {}  # Missing gen_int_rv

        with pytest.raises(ValueError, match="gen_int_rv is required"):
            builder.compute_n_initialization_points()

    def test_compute_n_initialization_points_returns_correct_value(
        self, simple_builder
    ):
        """Test that compute_n_initialization_points returns max of lookbacks."""
        n_init = simple_builder.compute_n_initialization_points()
        # gen_int has 3 elements (1-indexed) -> 3
        # delay has 4 elements (0-indexed) -> lookback_days = 3
        # max(3, 3) = 3
        assert n_init == 3


class TestMultiSignalModelSampling:
    """Test MultiSignalModel sampling with population structure at sample time."""

    def test_sample_with_population_structure(self, simple_builder):
        """Test that sample() works with population structure at sample time."""
        model = simple_builder.build()
        n_days = 30
        n_total = model.latent.n_initialization_points + n_days

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as tr:
                model.sample(
                    n_days_post_init=n_days,
                    population_size=1_000_000,
                    subpop_fractions=SUBPOP_FRACTIONS,
                    hospital={"obs": None},
                )

        inf_aggregate = tr["latent_infections"]["value"]
        inf_all = tr["latent_infections_by_subpop"]["value"]
        assert inf_aggregate.shape == (n_total,)
        assert inf_all.shape == (n_total, 3)  # K=3

    def test_run_with_population_structure(self, simple_builder):
        """Test that run() works with population structure at sample time."""
        model = simple_builder.build()
        n_days = 10
        n_total = model.latent.n_initialization_points + n_days

        # Create dense observations with NaN padding for initialization period
        obs_values = jnp.array([10.0, 12.0, 15.0, 14.0, 11.0])
        obs = model.pad_observations(obs_values)
        # Pad with NaN for remaining days
        obs = jnp.concatenate([obs, jnp.full(n_days - len(obs_values), jnp.nan)])

        model.run(
            num_warmup=5,
            num_samples=5,
            n_days_post_init=n_days,
            population_size=1_000_000,
            subpop_fractions=SUBPOP_FRACTIONS,
            hospital={"obs": obs},
        )

        samples = model.mcmc.get_samples()
        assert "latent_infections" in samples
        assert samples["latent_infections"].shape == (5, n_total)


class TestMultiSignalModelValidation:
    """Test data validation."""

    def test_validate_data_accepts_valid_data(self, validation_builder):
        """Test that validate_data accepts valid dense and sparse data."""
        model = validation_builder.build()
        n_total = model.latent.n_initialization_points + 30

        model.validate_data(
            n_days_post_init=30,
            subpop_fractions=SUBPOP_FRACTIONS,
            hospital={
                "obs": jnp.full(n_total, jnp.nan),
            },
            hospital_subpop={
                "obs": jnp.array([10, 20]),
                "times": jnp.array([5, 10]),
            },
        )

    def test_validate_data_rejects_out_of_bounds_times(self, validation_builder):
        """Test that times exceeding n_total_days raises error."""
        model = validation_builder.build()
        n_total = model.latent.n_initialization_points + 30

        with pytest.raises(ValueError, match="times"):
            model.validate_data(
                n_days_post_init=30,
                subpop_fractions=SUBPOP_FRACTIONS,
                hospital_subpop={
                    "obs": jnp.array([10]),
                    "times": jnp.array([n_total + 10]),
                },
            )

    def test_validate_data_rejects_negative_times(self, validation_builder):
        """Test that negative times raises error."""
        model = validation_builder.build()

        with pytest.raises(ValueError, match="times.*negative"):
            model.validate_data(
                n_days_post_init=30,
                subpop_fractions=SUBPOP_FRACTIONS,
                hospital_subpop={
                    "obs": jnp.array([10]),
                    "times": jnp.array([-1]),
                },
            )

    def test_validate_data_rejects_unknown_observation(self, validation_builder):
        """Test that unknown observation name raises error."""
        model = validation_builder.build()

        with pytest.raises(ValueError, match="Unknown"):
            model.validate_data(
                n_days_post_init=30,
                subpop_fractions=SUBPOP_FRACTIONS,
                unknown_obs={
                    "obs": jnp.array([10]),
                    "times": jnp.array([5]),
                },
            )

    def test_validate_data_rejects_mismatched_obs_times_length(
        self, validation_builder
    ):
        """Test that mismatched obs and times lengths raises error."""
        model = validation_builder.build()

        with pytest.raises(ValueError, match="obs.*times"):
            model.validate_data(
                n_days_post_init=30,
                subpop_fractions=SUBPOP_FRACTIONS,
                hospital_subpop={
                    "obs": jnp.array([10, 20, 30]),  # 3 elements
                    "times": jnp.array([5, 10]),  # 2 elements
                },
            )

    def test_validate_method_calls_internal_validate(self, simple_builder):
        """Test that validate() method calls _validate()."""
        model = simple_builder.build()
        # Should not raise
        model.validate()

    def test_validate_data_rejects_negative_subpop_indices(self, validation_builder):
        """Test that negative subpop_indices raises error."""
        model = validation_builder.build()

        with pytest.raises(ValueError, match="subpop_indices.*negative"):
            model.validate_data(
                n_days_post_init=30,
                subpop_fractions=SUBPOP_FRACTIONS,
                hospital_subpop={
                    "subpop_indices": jnp.array([-1, 0, 1]),
                    "times": jnp.array([5, 6, 7]),
                },
            )

    def test_validate_data_rejects_out_of_bounds_subpop_indices(
        self, validation_builder
    ):
        """Test that subpop_indices >= K raises error."""
        model = validation_builder.build()

        # K is 3 (from SUBPOP_FRACTIONS = [0.3, 0.25, 0.45])
        with pytest.raises(ValueError, match="subpop_indices"):
            model.validate_data(
                n_days_post_init=30,
                subpop_fractions=SUBPOP_FRACTIONS,
                hospital_subpop={
                    "subpop_indices": jnp.array([0, 1, 5]),  # 5 >= 3
                    "times": jnp.array([5, 6, 7]),
                },
            )

    def test_validate_data_rejects_wrong_length_dense_obs(self, validation_builder):
        """Test that dense obs with wrong length raises error."""
        model = validation_builder.build()

        with pytest.raises(ValueError, match="obs.*n_total"):
            model.validate_data(
                n_days_post_init=30,
                subpop_fractions=SUBPOP_FRACTIONS,
                hospital={
                    "obs": jnp.array([10, 20, 30]),  # wrong length
                },
            )


class TestPyrenewBuilderErrorHandling:
    """Test PyrenewBuilder error handling."""

    def test_build_raises_on_construction_error(self):
        """Test that build() raises TypeError on latent construction failure."""
        builder = PyrenewBuilder()
        gen_int = DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))

        # Configure with an invalid parameter that will cause construction to fail
        builder.configure_latent(
            HierarchicalInfections,
            gen_int_rv=gen_int,
            I0_rv=DeterministicVariable("I0", 0.001),
            initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
            baseline_rt_process=RandomWalk(),
            subpop_rt_deviation_process=RandomWalk(),
            invalid_extra_param="this will cause TypeError",  # Invalid param
        )

        delay = DeterministicPMF("delay", jnp.array([0.5, 0.5]))
        obs = Counts(
            name="hospital",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=delay,
            noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
        )
        builder.add_observation(obs)

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            builder.build()


class TestMultiSignalModelObservationValidation:
    """Test observation process validation in MultiSignalModel."""

    def test_rejects_observation_without_infection_resolution(self):
        """Test that observations must implement infection_resolution()."""
        from pyrenew.observation.base import BaseObservationProcess

        class BadObservation(BaseObservationProcess):
            """Observation that raises NotImplementedError for infection_resolution."""

            def __init__(self):
                """Initialize without temporal_pmf_rv."""
                self.name = "bad"
                self.temporal_pmf_rv = None

            def sample(self, **kwargs):
                """Sample stub."""
                pass

            def validate(self):
                """Validate stub."""
                pass

            def lookback_days(self):
                """
                Return lookback.

                Returns
                -------
                int
                    The lookback value of 1.
                """
                return 1

            def infection_resolution(self):
                """
                Return an invalid resolution to simulate bad implementation.

                Returns
                -------
                str
                    An invalid resolution string.
                """
                return "invalid_resolution"

            def _predicted_obs(self, infections):
                """
                Predicted obs stub.

                Returns
                -------
                ArrayLike
                    The infections array unchanged.
                """
                return infections

            def validate_data(self, n_total, n_subpops, **obs_data):
                """Validate data stub."""
                pass

        builder = PyrenewBuilder()
        gen_int = DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))

        builder.configure_latent(
            HierarchicalInfections,
            gen_int_rv=gen_int,
            I0_rv=DeterministicVariable("I0", 0.001),
            initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
            baseline_rt_process=RandomWalk(),
            subpop_rt_deviation_process=RandomWalk(),
        )

        bad_obs = BadObservation()
        builder.add_observation(bad_obs)

        with pytest.raises(ValueError, match="invalid infection_resolution"):
            builder.build()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
