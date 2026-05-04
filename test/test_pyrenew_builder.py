"""
Tests for PyrenewBuilder and MultiSignalModel.
"""

from datetime import date, timedelta

import jax.numpy as jnp
import numpyro
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    AR1,
    PopulationInfections,
    RandomWalk,
    StepwiseTemporalProcess,
    SubpopulationInfections,
    WeeklyTemporalProcess,
)
from pyrenew.model import MultiSignalModel, PyrenewBuilder
from pyrenew.observation import (
    NegativeBinomialNoise,
    PoissonNoise,
    PopulationCounts,
    SubpopulationCounts,
)
from pyrenew.time import ISO_WEEK, MMWR_WEEK

# Standard population structure for tests (3 subpopulations)
SUBPOP_FRACTIONS = jnp.array([0.3, 0.25, 0.45])


def fixed_ar1(autoreg=0.9, innovation_sd=0.05):
    """
    Construct an AR1 process with fixed parameters.

    Returns
    -------
    AR1
        Temporal process with deterministic autoregression and innovation scale.
    """
    return AR1(
        autoreg_rv=DeterministicVariable("autoreg", autoreg),
        innovation_sd_rv=DeterministicVariable("innovation_sd", innovation_sd),
    )


def fixed_random_walk(innovation_sd=1.0):
    """
    Construct a RandomWalk with a fixed innovation scale.

    Returns
    -------
    RandomWalk
        Temporal process with deterministic innovation standard deviation.
    """
    return RandomWalk(
        innovation_sd_rv=DeterministicVariable("innovation_sd", innovation_sd)
    )


def _obs_date_for_dow(target_first_day_dow: int, n_init: int) -> date:
    """
    Return an ``obs_start_date`` whose axis-origin day-of-week matches.

    Given a desired ``first_day_dow`` (day-of-week of element 0 of the
    padded axis) and the model's ``n_init``, pick a concrete date for
    the first observation day such that
    ``(obs_start_date.weekday() - n_init) % 7 == target_first_day_dow``.

    Parameters
    ----------
    target_first_day_dow
        Desired axis-origin day-of-week in ``{0, ..., 6}``.
    n_init
        Initialization-period length in days.

    Returns
    -------
    datetime.date
        A date with the required day-of-week, drawn from January 2024.
    """
    obs_dow = (target_first_day_dow + n_init) % 7
    return date(2024, 1, 1) + timedelta(days=obs_dow)


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
        baseline_rt_process=fixed_random_walk(),
        subpop_rt_deviation_process=fixed_random_walk(),
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
        baseline_rt_process=fixed_random_walk(),
        subpop_rt_deviation_process=fixed_random_walk(),
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
                baseline_rt_process=fixed_random_walk(),
                subpop_rt_deviation_process=fixed_random_walk(),
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
                baseline_rt_process=fixed_random_walk(),
                subpop_rt_deviation_process=fixed_random_walk(),
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
            baseline_rt_process=fixed_random_walk(),
            subpop_rt_deviation_process=fixed_random_walk(),
        )

        with pytest.raises(RuntimeError, match="already configured"):
            builder.configure_latent(
                SubpopulationInfections,
                gen_int_rv=gen_int,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=fixed_random_walk(),
                subpop_rt_deviation_process=fixed_random_walk(),
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

    def test_first_day_dow_reaches_calendar_aligned_latent_process(self):
        """MultiSignalModel forwards model-axis day of week to the latent process."""
        latent = PopulationInfections(
            name="PopulationInfections",
            gen_int_rv=DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3])),
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
            single_rt_process=WeeklyTemporalProcess(
                fixed_ar1(autoreg=0.9, innovation_sd=0.05),
                start_dow=MMWR_WEEK,
            ),
            n_initialization_points=3,
        )
        model = MultiSignalModel(latent, {"ed": _daily_ed_counts()})

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                model.sample(
                    n_days_post_init=10,
                    population_size=1_000_000,
                    obs_start_date=_obs_date_for_dow(target_first_day_dow=3, n_init=3),
                    ed={"obs": None},
                )

        log_rt = trace["PopulationInfections::log_rt_single"]["value"]
        weekly = trace["log_rt_single_weekly"]["value"]

        assert log_rt.shape == (model.latent.n_initialization_points + 10, 1)
        assert weekly.shape[0] < log_rt.shape[0]
        assert jnp.allclose(log_rt[:3], log_rt[0])
        assert jnp.allclose(log_rt[3:10], log_rt[3])

    def test_missing_obs_start_date_for_calendar_aligned_latent_process_raises(self):
        """Calendar-aligned latent temporal processes trigger the model-entry anchor check."""
        latent = PopulationInfections(
            name="PopulationInfections",
            gen_int_rv=DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3])),
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
            single_rt_process=WeeklyTemporalProcess(
                fixed_ar1(autoreg=0.9, innovation_sd=0.05),
                start_dow=MMWR_WEEK,
            ),
            n_initialization_points=3,
        )
        model = MultiSignalModel(latent, {"ed": _daily_ed_counts()})

        with numpyro.handlers.seed(rng_seed=42):
            with pytest.raises(ValueError, match="obs_start_date is required"):
                model.sample(
                    n_days_post_init=10,
                    population_size=1_000_000,
                    ed={"obs": None},
                )

    def test_builder_mixed_cadence_weekly_rt_samples(self):
        """
        Mixed daily/weekly observations can use weekly-parameterized Rt.

        The latent temporal process records a coarse Rt trajectory, the latent
        process records daily Rt values, ED remains on the daily likelihood
        scale, and hospital observations are scored on the weekly likelihood
        scale.
        """
        builder = _coherence_builder(
            single_rt_process=WeeklyTemporalProcess(
                fixed_ar1(autoreg=0.9, innovation_sd=0.05),
                start_dow=MMWR_WEEK,
            ),
            observations=[_weekly_hosp_counts(), _daily_ed_counts()],
        )
        model = builder.build()
        n_days_post_init = 28
        n_total = model.latent.n_initialization_points + n_days_post_init
        obs_start_date = _obs_date_for_dow(
            target_first_day_dow=6,
            n_init=model.latent.n_initialization_points,
        )
        hospital_obs = jnp.array([jnp.nan, 5.0, 7.0, 6.0], dtype=float)
        ed_obs = jnp.concatenate(
            [
                jnp.full(model.latent.n_initialization_points, jnp.nan),
                jnp.ones(n_days_post_init) * 3.0,
            ]
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                model.sample(
                    n_days_post_init=n_days_post_init,
                    population_size=1_000_000,
                    obs_start_date=obs_start_date,
                    hospital={"obs": hospital_obs},
                    ed={"obs": ed_obs},
                )

        assert trace["log_rt_single_weekly"]["value"].shape == (5, 1)
        assert trace["PopulationInfections::log_rt_single"]["value"].shape == (
            n_total,
            1,
        )
        assert trace["ed_obs"]["value"].shape == (n_total,)
        assert trace["hospital_obs"]["value"].shape == hospital_obs.shape
        assert trace["hospital_predicted_daily"]["value"].shape == (n_total,)
        assert trace["hospital_predicted"]["value"].shape == hospital_obs.shape


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
                "subpop_indices": jnp.array([0, 1]),
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
                    "subpop_indices": jnp.array([0]),
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
                    "subpop_indices": jnp.array([0]),
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
                    "subpop_indices": jnp.array([0, 1]),
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
    def test_resolve_first_day_dow(self, simple_builder, obs_start_dow, expected):
        """_resolve_first_day_dow offsets by n_initialization_points."""
        model = simple_builder.build()
        obs_date = date(2024, 1, 1) + timedelta(days=obs_start_dow)
        assert model._resolve_first_day_dow(obs_date) == expected

    def test_resolve_first_day_dow_none_passthrough(self, simple_builder):
        """_resolve_first_day_dow returns None when obs_start_date is None."""
        model = simple_builder.build()
        assert model._resolve_first_day_dow(None) is None

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


def _weekly_hosp_counts(name="hospital", start_dow=MMWR_WEEK):
    """
    Build a weekly-aggregated PopulationCounts observation with PoissonNoise.

    Returns
    -------
    PopulationCounts
        Weekly-regular observation anchored to the specified ``start_dow``.
    """
    return PopulationCounts(
        name=name,
        ascertainment_rate_rv=DeterministicVariable(f"{name}_ihr", 0.01),
        delay_distribution_rv=DeterministicPMF(f"{name}_delay", jnp.array([1.0])),
        noise=PoissonNoise(),
        aggregation="weekly",
        reporting_schedule="regular",
        start_dow=start_dow,
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


class TestBuilderConfigurations:
    """PyrenewBuilder.build() accepts varied R(t) and observation cadences."""

    def test_daily_rt_with_daily_observation_passes(self):
        """step_size=1 and P=1: valid."""
        builder = _coherence_builder(
            single_rt_process=fixed_ar1(autoreg=0.9, innovation_sd=0.05),
            observations=[_daily_ed_counts()],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_weekly_rt_with_weekly_observation_passes(self):
        """step_size=7 and P=7: valid when weekly is the only obs cadence."""
        builder = _coherence_builder(
            single_rt_process=StepwiseTemporalProcess(
                fixed_ar1(autoreg=0.9, innovation_sd=0.05), step_size=7
            ),
            observations=[_weekly_hosp_counts()],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_daily_rt_with_mixed_observations_passes(self):
        """step_size=1 with mixed P=1 + P=7: valid (R(t) at finest cadence)."""
        builder = _coherence_builder(
            single_rt_process=fixed_ar1(autoreg=0.9, innovation_sd=0.05),
            observations=[_weekly_hosp_counts(), _daily_ed_counts()],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_weekly_rt_with_daily_observation_passes(self):
        """Coarse Rt parameter cadence is allowed with daily observations."""
        builder = _coherence_builder(
            single_rt_process=StepwiseTemporalProcess(
                fixed_ar1(autoreg=0.9, innovation_sd=0.05), step_size=7
            ),
            observations=[_weekly_hosp_counts(), _daily_ed_counts()],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_mismatched_weekly_week_passes(self):
        """Weekly observations can use different start_dow; each aggregates independently."""
        builder = _coherence_builder(
            single_rt_process=fixed_ar1(autoreg=0.9, innovation_sd=0.05),
            observations=[
                _weekly_hosp_counts(name="hospital", start_dow=MMWR_WEEK),
                _weekly_hosp_counts(name="other", start_dow=ISO_WEEK),
            ],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_matching_weekly_week_passes(self):
        """Two weekly observations sharing a start_dow build normally."""
        builder = _coherence_builder(
            single_rt_process=fixed_ar1(autoreg=0.9, innovation_sd=0.05),
            observations=[
                _weekly_hosp_counts(name="hospital", start_dow=MMWR_WEEK),
                _weekly_hosp_counts(name="other", start_dow=MMWR_WEEK),
            ],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_arbitrary_step_size_with_weekly_observation_passes(self):
        """Parameter cadence need not match observation aggregation period."""
        builder = _coherence_builder(
            single_rt_process=StepwiseTemporalProcess(
                fixed_ar1(autoreg=0.9, innovation_sd=0.05), step_size=2
            ),
            observations=[_weekly_hosp_counts()],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_weekly_rt_matches_weekly_observation_week_passes(self):
        """Sunday-start weeks pair with Saturday-ending weekly observations."""
        builder = _coherence_builder(
            single_rt_process=WeeklyTemporalProcess(
                fixed_ar1(autoreg=0.9, innovation_sd=0.05),
                start_dow=MMWR_WEEK,
            ),
            observations=[_weekly_hosp_counts(start_dow=MMWR_WEEK)],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_weekly_rt_mismatches_weekly_observation_week_passes(self):
        """A weekly Rt anchor can differ from a weekly observation anchor."""
        builder = _coherence_builder(
            single_rt_process=WeeklyTemporalProcess(
                fixed_ar1(autoreg=0.9, innovation_sd=0.05),
                start_dow=ISO_WEEK,
            ),
            observations=[_weekly_hosp_counts(start_dow=MMWR_WEEK)],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_weekly_rt_with_only_daily_observations_passes(self):
        """Calendar-week-aligned R(t) pairs with daily observations."""
        builder = _coherence_builder(
            single_rt_process=WeeklyTemporalProcess(
                fixed_ar1(autoreg=0.9, innovation_sd=0.05),
                start_dow=MMWR_WEEK,
            ),
            observations=[_daily_ed_counts()],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_model_index_alignment_with_weekly_observation_passes(self):
        """Model-index-aligned R(t) pairs with weekly observations."""
        builder = _coherence_builder(
            single_rt_process=StepwiseTemporalProcess(
                fixed_ar1(autoreg=0.9, innovation_sd=0.05), step_size=7
            ),
            observations=[_weekly_hosp_counts(start_dow=MMWR_WEEK)],
        )
        model = builder.build()
        assert isinstance(model, MultiSignalModel)


class TestMultiSignalValidateDataAnchor:
    """MultiSignalModel.validate_data sample-time anchor check for obs_start_date."""

    def test_missing_obs_start_date_for_weekly_obs_raises(self):
        """An observation with aggregation='weekly' must have obs_start_date supplied."""
        builder = _coherence_builder(
            single_rt_process=StepwiseTemporalProcess(
                fixed_ar1(autoreg=0.9, innovation_sd=0.05), step_size=7
            ),
            observations=[_weekly_hosp_counts()],
        )
        model = builder.build()
        with pytest.raises(ValueError, match="obs_start_date is required"):
            model.validate_data(
                n_days_post_init=28,
                hospital={"obs": jnp.ones(4) * 5.0},
            )

    def test_obs_start_date_supplied_for_weekly_obs_passes(self):
        """Supplying obs_start_date satisfies the anchor check."""
        builder = _coherence_builder(
            single_rt_process=StepwiseTemporalProcess(
                fixed_ar1(autoreg=0.9, innovation_sd=0.05), step_size=7
            ),
            observations=[_weekly_hosp_counts()],
        )
        model = builder.build()
        obs_start_date = _obs_date_for_dow(
            target_first_day_dow=6,
            n_init=model.latent.n_initialization_points,
        )
        model.validate_data(
            n_days_post_init=28,
            obs_start_date=obs_start_date,
            hospital={"obs": jnp.ones(4) * 5.0},
        )

    def test_anchor_check_skipped_for_daily_obs(self):
        """Daily observations do not require obs_start_date at validate_data time."""
        builder = _coherence_builder(
            single_rt_process=fixed_ar1(autoreg=0.9, innovation_sd=0.05),
            observations=[_daily_ed_counts()],
        )
        model = builder.build()
        n_total = model.latent.n_initialization_points + 30
        model.validate_data(
            n_days_post_init=30,
            ed={"obs": jnp.ones(n_total) * 5.0},
        )

    def test_missing_obs_start_date_for_calendar_aligned_latent_raises(self):
        """A calendar-week-aligned latent temporal process requires obs_start_date."""
        builder = _coherence_builder(
            single_rt_process=WeeklyTemporalProcess(
                fixed_ar1(autoreg=0.9, innovation_sd=0.05),
                start_dow=MMWR_WEEK,
            ),
            observations=[_daily_ed_counts()],
        )
        model = builder.build()
        n_total = model.latent.n_initialization_points + 30
        with pytest.raises(ValueError, match="obs_start_date is required"):
            model.validate_data(
                n_days_post_init=30,
                ed={"obs": jnp.ones(n_total) * 5.0},
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
