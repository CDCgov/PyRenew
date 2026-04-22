"""
Tests for PyrenewBuilder and MultiSignalModel.
"""

import jax.numpy as jnp
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    AR1,
    PopulationInfections,
    RandomWalk,
    StepwiseTemporalProcess,
    SubpopulationInfections,
)
from pyrenew.model import MultiSignalModel, PyrenewBuilder
from pyrenew.observation import (
    NegativeBinomialNoise,
    PoissonNoise,
    PopulationCounts,
    SubpopulationCounts,
)

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
        SubpopulationInfections,
        gen_int_rv=gen_int,
        I0_rv=DeterministicVariable("I0", 0.001),
        log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
        baseline_rt_process=RandomWalk(),
        subpop_rt_deviation_process=RandomWalk(),
    )

    delay = DeterministicPMF("delay", jnp.array([0.1, 0.3, 0.4, 0.2]))
    obs = PopulationCounts(
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
        Builder with PopulationCounts ("hospital") and SubpopulationCounts
        ("hospital_subpop") observations.
    """
    builder = PyrenewBuilder()
    gen_int = DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))

    builder.configure_latent(
        SubpopulationInfections,
        gen_int_rv=gen_int,
        I0_rv=DeterministicVariable("I0", 0.001),
        log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
        baseline_rt_process=RandomWalk(),
        subpop_rt_deviation_process=RandomWalk(),
    )

    delay = DeterministicPMF("delay", jnp.array([0.1, 0.3, 0.4, 0.2]))
    builder.add_observation(
        PopulationCounts(
            name="hospital",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=delay,
            noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
        )
    )
    builder.add_observation(
        SubpopulationCounts(
            name="hospital_subpop",
            ascertainment_rate_rv=DeterministicVariable("ihr_subpop", 0.01),
            delay_distribution_rv=delay,
            noise=NegativeBinomialNoise(DeterministicVariable("conc_subpop", 10.0)),
            reporting_schedule="irregular",
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
                SubpopulationInfections,
                gen_int_rv=gen_int,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=RandomWalk(),
                subpop_rt_deviation_process=RandomWalk(),
                subpop_fractions=jnp.array([0.5, 0.5]),
            )

    def test_rejects_n_initialization_points_at_configure_time(self):
        """Test that n_initialization_points is rejected at configure time."""
        builder = PyrenewBuilder()
        gen_int = DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))

        with pytest.raises(ValueError, match="Do not specify n_initialization_points"):
            builder.configure_latent(
                SubpopulationInfections,
                gen_int_rv=gen_int,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=RandomWalk(),
                subpop_rt_deviation_process=RandomWalk(),
                n_initialization_points=10,
            )

    def test_rejects_reconfiguring_latent(self):
        """Test that configuring latent twice raises RuntimeError."""
        builder = PyrenewBuilder()
        gen_int = DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))

        builder.configure_latent(
            SubpopulationInfections,
            gen_int_rv=gen_int,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
            baseline_rt_process=RandomWalk(),
            subpop_rt_deviation_process=RandomWalk(),
        )

        with pytest.raises(RuntimeError, match="already configured"):
            builder.configure_latent(
                SubpopulationInfections,
                gen_int_rv=gen_int,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=RandomWalk(),
                subpop_rt_deviation_process=RandomWalk(),
            )

    def test_rejects_duplicate_observation_name(self, simple_builder):
        """Test that adding duplicate observation name raises ValueError."""
        delay = DeterministicPMF("delay2", jnp.array([0.5, 0.5]))
        obs = PopulationCounts(
            name="hospital",
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
        builder.latent_class = SubpopulationInfections
        builder.latent_params = {}

        with pytest.raises(ValueError, match="gen_int_rv is required"):
            builder.compute_n_initialization_points()

    def test_compute_n_initialization_points_returns_correct_value(
        self, simple_builder
    ):
        """Test that compute_n_initialization_points returns max of lookbacks."""
        n_init = simple_builder.compute_n_initialization_points()
        # gen_int has 3 elements -> 3
        # delay has 4 elements -> lookback_days = 3
        # max(3, 3) = 3
        assert n_init == 3


class TestMultiSignalModelSampling:
    """Test MultiSignalModel sampling with population structure at sample time."""

    def test_run_with_population_structure(self, simple_builder):
        """Test that run() works and produces reasonable posterior samples."""
        model = simple_builder.build()
        n_days = 10
        n_total = model.latent.n_initialization_points + n_days

        obs_values = jnp.array([10.0, 12.0, 15.0, 14.0, 11.0])
        obs = model.pad_observations(obs_values)
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
        # All infection samples should be positive
        assert jnp.all(samples["latent_infections"] > 0)

    def test_prior_predictive_multi_signal(self, simple_builder):
        """Test prior predictive sampling from a builder-constructed model."""
        import jax.random
        from numpyro.infer import Predictive

        model = simple_builder.build()
        n_days = 20

        predictive = Predictive(
            model.sample,
            num_samples=5,
        )

        rng_key = jax.random.PRNGKey(42)
        prior_samples = predictive(
            rng_key,
            n_days_post_init=n_days,
            population_size=1_000_000,
            subpop_fractions=SUBPOP_FRACTIONS,
            hospital={"obs": None},
        )

        n_total = model.latent.n_initialization_points + n_days

        assert "latent_infections" in prior_samples
        assert prior_samples["latent_infections"].shape == (5, n_total)
        # All prior predictive infections should be positive
        assert jnp.all(prior_samples["latent_infections"] > 0)


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
                "period_end_times": jnp.array([5, 10]),
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
                    "period_end_times": jnp.array([n_total + 10]),
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
                    "period_end_times": jnp.array([-1]),
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
                    "period_end_times": jnp.array([5]),
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
                    "period_end_times": jnp.array([5, 10]),  # 2 elements
                },
            )

    def test_validate_method_calls_internal_validate(self, simple_builder):
        """Test that validate() succeeds on a valid model."""
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
                    "period_end_times": jnp.array([5, 6, 7]),
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
                    "period_end_times": jnp.array([5, 6, 7]),
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


class TestMultiSignalModelHelpers:
    """Test MultiSignalModel helper methods."""

    def test_pad_observations_prepends_nans(self, simple_builder):
        """Test that pad_observations prepends correct NaN padding."""
        model = simple_builder.build()
        n_init = model.latent.n_initialization_points

        obs = jnp.array([10, 20, 30])
        padded = model.pad_observations(obs)

        # Shape should include initialization period
        assert padded.shape == (n_init + 3,)
        # First n_init values should be NaN
        assert jnp.all(jnp.isnan(padded[:n_init]))
        # Remaining values should match input
        assert jnp.array_equal(padded[n_init:], jnp.array([10.0, 20.0, 30.0]))
        # Integer input should be converted to float
        assert jnp.issubdtype(padded.dtype, jnp.floating)

    @pytest.mark.parametrize(
        "obs_start_dow, expected",
        [
            (0, (0 - 3) % 7),
            (3, (3 - 3) % 7),
            (6, (6 - 3) % 7),
        ],
    )
    def test_compute_first_day_dow(self, simple_builder, obs_start_dow, expected):
        """Test that compute_first_day_dow offsets by n_initialization_points."""
        model = simple_builder.build()
        assert model.compute_first_day_dow(obs_start_dow) == expected

    def test_shift_times_adds_offset(self, simple_builder):
        """Test that shift_times shifts by n_initialization_points."""
        model = simple_builder.build()
        n_init = model.latent.n_initialization_points

        times = jnp.array([0, 5, 10])
        shifted = model.shift_times(times)

        assert jnp.array_equal(shifted, times + n_init)


def _coherence_builder(
    *,
    single_rt_process,
    observations,
):
    """
    Build a configured PyrenewBuilder with a PopulationInfections latent and
    a supplied temporal process, plus the given observation instances.

    Returns
    -------
    PyrenewBuilder
        Builder configured but not yet built.
    """
    builder = PyrenewBuilder()
    builder.configure_latent(
        PopulationInfections,
        gen_int_rv=DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3])),
        I0_rv=DeterministicVariable("I0", 0.001),
        log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
        single_rt_process=single_rt_process,
    )
    for obs in observations:
        builder.add_observation(obs)
    return builder


def _weekly_hosp_counts(name="hospital", period_end_dow=5):
    """
    Build a weekly-aggregated PopulationCounts observation with PoissonNoise.

    Returns
    -------
    PopulationCounts
        Weekly-regular observation anchored to the specified period end dow.
    """
    return PopulationCounts(
        name=name,
        ascertainment_rate_rv=DeterministicVariable(f"{name}_ihr", 0.01),
        delay_distribution_rv=DeterministicPMF(f"{name}_delay", jnp.array([1.0])),
        noise=PoissonNoise(),
        aggregation_period=7,
        reporting_schedule="regular",
        period_end_dow=period_end_dow,
    )


def _daily_ed_counts(name="ed"):
    """
    Build a daily PopulationCounts observation with PoissonNoise.

    Returns
    -------
    PopulationCounts
        Daily-regular observation with no aggregation.
    """
    return PopulationCounts(
        name=name,
        ascertainment_rate_rv=DeterministicVariable(f"{name}_ihr", 0.01),
        delay_distribution_rv=DeterministicPMF(f"{name}_delay", jnp.array([1.0])),
        noise=PoissonNoise(),
    )


class TestBuilderCoherence:
    """PyrenewBuilder._validate_coherence enforcement at build() time."""

    def test_daily_rt_with_daily_observation_passes(self):
        """step_size=1 and P=1: valid."""
        builder = _coherence_builder(
            single_rt_process=AR1(autoreg=0.9, innovation_sd=0.05),
            observations=[_daily_ed_counts()],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_weekly_rt_with_weekly_observation_passes(self):
        """step_size=7 and P=7: valid when weekly is the only obs cadence."""
        builder = _coherence_builder(
            single_rt_process=StepwiseTemporalProcess(
                AR1(autoreg=0.9, innovation_sd=0.05), step_size=7
            ),
            observations=[_weekly_hosp_counts()],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_daily_rt_with_mixed_observations_passes(self):
        """step_size=1 with mixed P=1 + P=7: valid (R(t) at finest cadence)."""
        builder = _coherence_builder(
            single_rt_process=AR1(autoreg=0.9, innovation_sd=0.05),
            observations=[_weekly_hosp_counts(), _daily_ed_counts()],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_weekly_rt_with_daily_observation_raises(self):
        """step_size=7 with any daily obs: rule 2 violation."""
        builder = _coherence_builder(
            single_rt_process=StepwiseTemporalProcess(
                AR1(autoreg=0.9, innovation_sd=0.05), step_size=7
            ),
            observations=[_weekly_hosp_counts(), _daily_ed_counts()],
        )
        with pytest.raises(ValueError, match="exceeding the finest"):
            builder.build()

    def test_mismatched_weekly_period_end_dow_raises(self):
        """Two weekly observations with different period_end_dow: rule 1 violation."""
        builder = _coherence_builder(
            single_rt_process=AR1(autoreg=0.9, innovation_sd=0.05),
            observations=[
                _weekly_hosp_counts(name="hospital", period_end_dow=5),
                _weekly_hosp_counts(name="other", period_end_dow=6),
            ],
        )
        with pytest.raises(ValueError, match="must agree on period_end_dow"):
            builder.build()

    def test_matching_weekly_period_end_dow_passes(self):
        """Two weekly observations agreeing on period_end_dow: rule 1 passes."""
        builder = _coherence_builder(
            single_rt_process=AR1(autoreg=0.9, innovation_sd=0.05),
            observations=[
                _weekly_hosp_counts(name="hospital", period_end_dow=5),
                _weekly_hosp_counts(name="other", period_end_dow=5),
            ],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_step_size_mismatches_coarse_period_raises(self):
        """step_size > 1 must equal every coarse observation aggregation_period."""
        builder = _coherence_builder(
            single_rt_process=StepwiseTemporalProcess(
                AR1(autoreg=0.9, innovation_sd=0.05), step_size=2
            ),
            observations=[_weekly_hosp_counts()],
        )
        with pytest.raises(ValueError, match="All coarse cadences must agree"):
            builder.build()


class TestMultiSignalValidateDataAnchor:
    """MultiSignalModel.validate_data sample-time anchor check for first_day_dow."""

    def test_missing_first_day_dow_for_weekly_obs_raises(self):
        """An observation with aggregation_period>1 must have first_day_dow supplied."""
        builder = _coherence_builder(
            single_rt_process=StepwiseTemporalProcess(
                AR1(autoreg=0.9, innovation_sd=0.05), step_size=7
            ),
            observations=[_weekly_hosp_counts()],
        )
        model = builder.build()
        with pytest.raises(ValueError, match="requires 'first_day_dow'"):
            model.validate_data(
                n_days_post_init=28,
                hospital={"obs": jnp.ones(4) * 5.0},
            )

    def test_first_day_dow_supplied_for_weekly_obs_passes(self):
        """Supplying first_day_dow satisfies the anchor check."""
        builder = _coherence_builder(
            single_rt_process=StepwiseTemporalProcess(
                AR1(autoreg=0.9, innovation_sd=0.05), step_size=7
            ),
            observations=[_weekly_hosp_counts()],
        )
        model = builder.build()
        model.validate_data(
            n_days_post_init=28,
            hospital={"obs": jnp.ones(4) * 5.0, "first_day_dow": 6},
        )

    def test_anchor_check_skipped_for_daily_obs(self):
        """Daily observations do not require first_day_dow at validate_data time."""
        builder = _coherence_builder(
            single_rt_process=AR1(autoreg=0.9, innovation_sd=0.05),
            observations=[_daily_ed_counts()],
        )
        model = builder.build()
        n_total = model.latent.n_initialization_points + 30
        model.validate_data(
            n_days_post_init=30,
            ed={"obs": jnp.ones(n_total) * 5.0},
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
