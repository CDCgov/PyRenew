"""
Shared pytest fixtures for PyRenew tests.

This module provides reusable fixtures for creating observation processes,
test data, and common configurations used across multiple test files.

JAX defaults are configured through pytest-env in ``pyproject.toml``:
64-bit floats and four logical host devices for parallel MCMC chains.
Existing environment values take precedence; set ``JAX_ENABLE_X64=false``
to run tests with 32-bit floats.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from jax.typing import ArrayLike

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    MobilitySubpopulationInfections,
    PopulationInfections,
    SubpopulationInfections,
)
from pyrenew.metaclass import RandomVariable
from pyrenew.model import MultiSignalModel, PyrenewBuilder
from pyrenew.observation import (
    HierarchicalNormalNoise,
    NegativeBinomialNoise,
    PoissonNoise,
    PopulationCounts,
    SubpopulationCounts,
)
from pyrenew.randomvariable import DistributionalVariable, VectorizedVariable
from pyrenew.time import MMWR_WEEK
from test.test_helpers import fixed_ar1, fixed_random_walk

# =============================================================================
# PMF Fixtures
# =============================================================================


@pytest.fixture
def simple_delay_pmf():
    """
    Simple 1-day delay PMF (no delay).

    Returns
    -------
    jnp.ndarray
        A single-element PMF array representing no delay.
    """
    return jnp.array([1.0])


@pytest.fixture
def short_delay_pmf():
    """
    Short 2-day delay PMF.

    Returns
    -------
    jnp.ndarray
        A 2-element PMF array.
    """
    return jnp.array([0.5, 0.5])


@pytest.fixture
def long_delay_pmf():
    """
    Long 10-day delay PMF for edge case testing.

    Returns
    -------
    jnp.ndarray
        A 10-element PMF array.
    """
    return jnp.array([0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.03, 0.01, 0.01])


@pytest.fixture
def short_shedding_pmf():
    """
    Short 3-day shedding PMF.

    Returns
    -------
    jnp.ndarray
        A 3-element PMF array.
    """
    return jnp.array([0.3, 0.4, 0.3])


# =============================================================================
# Generation Interval Fixture
# =============================================================================


@pytest.fixture
def gen_int_rv():
    """
    COVID-like generation interval (7-day PMF).

    Returns
    -------
    DeterministicPMF
        Generation interval random variable.
    """
    pmf = jnp.array([0.16, 0.32, 0.25, 0.14, 0.07, 0.04, 0.02])
    return DeterministicPMF("gen_int", pmf)


# =============================================================================
# Noise Fixtures
# =============================================================================


@pytest.fixture
def hierarchical_normal_noise():
    """
    Standard HierarchicalNormalNoise with VectorizedVariable wrappers.

    Returns
    -------
    HierarchicalNormalNoise
        Noise model for continuous measurements.
    """
    sensor_mode_rv = VectorizedVariable(
        name="sensor_mode_rv",
        rv=DistributionalVariable("ww_sensor_mode", dist.Normal(0, 0.5)),
    )
    sensor_sd_rv = VectorizedVariable(
        name="sensor_sd_rv",
        rv=DistributionalVariable(
            "ww_sensor_sd", dist.TruncatedNormal(0.3, 0.15, low=0.10)
        ),
    )
    return HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)


@pytest.fixture
def hierarchical_normal_noise_tight():
    """
    Tight HierarchicalNormalNoise for near-deterministic testing.

    Returns
    -------
    HierarchicalNormalNoise
        Noise model with very small variance.
    """
    sensor_mode_rv = VectorizedVariable(
        name="sensor_mode_rv",
        rv=DistributionalVariable("ww_sensor_mode", dist.Normal(0, 0.01)),
    )
    sensor_sd_rv = VectorizedVariable(
        name="sensor_sd_rv",
        rv=DistributionalVariable(
            "ww_sensor_sd", dist.TruncatedNormal(0.01, 0.005, low=0.001)
        ),
    )
    return HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)


# =============================================================================
# Latent Infections Fixtures
# =============================================================================


@pytest.fixture
def subpop_fractions() -> jnp.ndarray:
    """
    Population fractions for three subpopulations.

    Returns
    -------
    jnp.ndarray
        Fractions that sum to 1.
    """
    return jnp.array([0.3, 0.25, 0.45])


@pytest.fixture
def infection_feedback_strength_rv() -> DeterministicVariable:
    """
    Constant negative infection feedback strength.

    Returns
    -------
    DeterministicVariable
        Feedback strength that lowers adjusted Rt below raw Rt.
    """
    return DeterministicVariable("infection_feedback_strength", -100.0)


@pytest.fixture
def infection_feedback_pmf_rv() -> DeterministicPMF:
    """
    Infection feedback PMF with the same length as ``gen_int_rv``.

    Returns
    -------
    DeterministicPMF
        Feedback PMF whose values differ from the generation interval.
    """
    return DeterministicPMF(
        "infection_feedback_pmf",
        jnp.array([0.4, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01]),
    )


@pytest.fixture
def subpopulation_infections_factory(
    gen_int_rv: DeterministicPMF,
) -> Callable[..., SubpopulationInfections]:
    """
    Factory for SubpopulationInfections with optional infection feedback.

    Returns
    -------
    Callable[..., SubpopulationInfections]
        Function that builds the standard configuration with the given
        feedback options.
    """

    def make(
        infection_feedback_strength_rv: RandomVariable | None = None,
        infection_feedback_pmf_rv: RandomVariable | None = None,
    ) -> SubpopulationInfections:
        """
        Build SubpopulationInfections with the given feedback options.

        Parameters
        ----------
        infection_feedback_strength_rv
            Infection feedback strength, or None.
        infection_feedback_pmf_rv
            Infection feedback PMF, or None.

        Returns
        -------
        SubpopulationInfections
            Configured infection process.
        """
        return SubpopulationInfections(
            name="subpopulation",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
            baseline_rt_process=fixed_ar1(autoreg=0.9, innovation_sd=0.05),
            subpop_rt_deviation_process=fixed_random_walk(innovation_sd=0.025),
            n_initialization_points=7,
            infection_feedback_strength_rv=infection_feedback_strength_rv,
            infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        )

    return make


@pytest.fixture
def subpop_log_rt_time_0() -> jnp.ndarray:
    """
    Distinct starting log Rt for three subpopulations.

    Returns
    -------
    jnp.ndarray
        Values in increasing order whose mean is not equal to any entry.
    """
    return jnp.array([-0.2, 0.0, 0.3])


@pytest.fixture
def random_walk_subpopulation_infections_factory(
    gen_int_rv: DeterministicPMF,
) -> Callable[[RandomVariable], SubpopulationInfections]:
    """
    Factory for SubpopulationInfections with random walk baseline and deviations.

    Returns
    -------
    Callable[[RandomVariable], SubpopulationInfections]
        Function that builds the configuration with the given starting log Rt.
    """

    def make(log_rt_time_0_rv: RandomVariable) -> SubpopulationInfections:
        """
        Build SubpopulationInfections with the given starting log Rt.

        Parameters
        ----------
        log_rt_time_0_rv
            Starting log Rt, a scalar or one value per subpopulation.

        Returns
        -------
        SubpopulationInfections
            Configured infection process.
        """
        return SubpopulationInfections(
            name="subpopulation",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=log_rt_time_0_rv,
            baseline_rt_process=fixed_random_walk(innovation_sd=0.05),
            subpop_rt_deviation_process=fixed_random_walk(innovation_sd=0.025),
            n_initialization_points=7,
        )

    return make


# =============================================================================
# Mobility Fixtures
# =============================================================================


@pytest.fixture
def roy_mobility_matrix() -> jnp.ndarray:
    """
    Mobility matrix of Roy et al. (2025) Fig 2b.

    Returns
    -------
    jnp.ndarray
        Column-stochastic matrix; entry [j, k] is the fraction of
        residents of region k present at location j.
    """
    return jnp.array([[0.9, 0.07, 0.21], [0.05, 0.85, 0.09], [0.05, 0.08, 0.7]])


@pytest.fixture
def roy_subpop_fractions() -> jnp.ndarray:
    """
    Population fractions of the three regions of Roy et al. (2025) Fig 2.

    Returns
    -------
    jnp.ndarray
        Fractions for populations of 20, 10, and 15 million.
    """
    return jnp.array([20.0, 10.0, 15.0]) / 45.0


@pytest.fixture
def mobility_initial_infections() -> jnp.ndarray:
    """
    Initialization-period infections per resident for three regions.

    Returns
    -------
    jnp.ndarray
        Shape (10, 3): ten days, longer than the 7-day generation
        interval, with values that differ by region.
    """
    return jnp.tile(jnp.array([1e-5, 2e-5, 5e-6]), (10, 1))


@pytest.fixture
def mobility_rt_raw() -> jnp.ndarray:
    """
    Unequal random-walk Rt for three locations over 150 days.

    Returns
    -------
    jnp.ndarray
        Shape (150, 3), starting at 2.0, 1.6, and 1.4.
    """
    steps = 0.01 * jax.random.normal(jax.random.key(901), (150, 3))
    return jnp.exp(jnp.log(jnp.array([2.0, 1.6, 1.4])) + jnp.cumsum(steps, axis=0))


def _mobility_latent_params(
    gen_int_rv: DeterministicPMF,
    mobility_matrix: ArrayLike,
    log_rt_time_0_rv: RandomVariable,
    susceptible_depletion: bool,
) -> dict[str, object]:
    """
    Constructor arguments shared by the mobility latent-process fixtures.

    Matches random_walk_subpopulation_infections_factory in name, initial
    infections, and temporal processes.

    Parameters
    ----------
    gen_int_rv
        Generation interval PMF.
    mobility_matrix
        Mobility matrix value.
    log_rt_time_0_rv
        Starting log Rt, a scalar or one value per region.
    susceptible_depletion
        Whether susceptible depletion is on.

    Returns
    -------
    dict[str, object]
        Keyword arguments for MobilitySubpopulationInfections, without
        n_initialization_points.
    """
    return {
        "name": "subpopulation",
        "gen_int_rv": gen_int_rv,
        "I0_rv": DeterministicVariable("I0", 0.001),
        "log_rt_time_0_rv": log_rt_time_0_rv,
        "baseline_rt_process": fixed_random_walk(innovation_sd=0.05),
        "subpop_rt_deviation_process": fixed_random_walk(innovation_sd=0.025),
        "mobility_matrix_rv": DeterministicVariable("mobility_matrix", mobility_matrix),
        "susceptible_depletion": susceptible_depletion,
    }


@pytest.fixture
def mobility_subpopulation_infections_factory(
    gen_int_rv: DeterministicPMF,
) -> Callable[..., MobilitySubpopulationInfections]:
    """
    Factory for MobilitySubpopulationInfections.

    Returns
    -------
    Callable[..., MobilitySubpopulationInfections]
        Function that builds the process with the given matrix, starting
        log Rt, and depletion switch.
    """

    def make(
        mobility_matrix: ArrayLike,
        log_rt_time_0_rv: RandomVariable,
        susceptible_depletion: bool = True,
    ) -> MobilitySubpopulationInfections:
        """
        Build MobilitySubpopulationInfections.

        Parameters
        ----------
        mobility_matrix
            Mobility matrix value.
        log_rt_time_0_rv
            Starting log Rt, a scalar or one value per region.
        susceptible_depletion
            Whether susceptible depletion is on.

        Returns
        -------
        MobilitySubpopulationInfections
            Configured infection process.
        """
        return MobilitySubpopulationInfections(
            n_initialization_points=7,
            **_mobility_latent_params(
                gen_int_rv,
                mobility_matrix,
                log_rt_time_0_rv,
                susceptible_depletion,
            ),
        )

    return make


@pytest.fixture
def mobility_subpopulation_infections(
    mobility_subpopulation_infections_factory: Callable[
        ..., MobilitySubpopulationInfections
    ],
    roy_mobility_matrix: jnp.ndarray,
) -> MobilitySubpopulationInfections:
    """
    MobilitySubpopulationInfections with the Fig 2b matrix.

    Returns
    -------
    MobilitySubpopulationInfections
        Process with a scalar starting log Rt of 0 and depletion on.
    """
    return mobility_subpopulation_infections_factory(
        roy_mobility_matrix, DeterministicVariable("log_rt_time_0", 0.0)
    )


@pytest.fixture
def mobility_model_factory(
    gen_int_rv: DeterministicPMF,
) -> Callable[[ArrayLike], MultiSignalModel]:
    """
    Factory for a builder-constructed model with mobility coupling.

    Returns
    -------
    Callable[[ArrayLike], MultiSignalModel]
        Function that builds the model with the given mobility matrix.
    """

    def make(mobility_matrix: ArrayLike) -> MultiSignalModel:
        """
        Build a model with one subpopulation count observation.

        Parameters
        ----------
        mobility_matrix
            Mobility matrix value.

        Returns
        -------
        MultiSignalModel
            Model with the latent configuration of the mobility factory
            and a SubpopulationCounts observation named "hospital_subpop".
        """
        builder = PyrenewBuilder()
        builder.configure_latent(
            MobilitySubpopulationInfections,
            **_mobility_latent_params(
                gen_int_rv,
                mobility_matrix,
                DeterministicVariable("log_rt_time_0", 0.0),
                True,
            ),
        )
        builder.add_observation(
            SubpopulationCounts(
                name="hospital_subpop",
                ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
                delay_distribution_rv=DeterministicPMF(
                    "delay", jnp.array([0.1, 0.3, 0.4, 0.2])
                ),
                noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
            )
        )
        return builder.build()

    return make


@pytest.fixture
def mobility_model_args(roy_subpop_fractions: jnp.ndarray) -> dict[str, object]:
    """
    Keyword arguments for sampling the mobility model.

    Returns
    -------
    dict[str, object]
        Arguments for model.sample: 30 days, the Fig 2 populations, and
        prior predictive counts for all three regions.
    """
    return {
        "n_days_post_init": 30,
        "population_size": 45_000_000,
        "subpop_fractions": roy_subpop_fractions,
        "hospital_subpop": {"obs": None, "subpop_indices": jnp.arange(3)},
    }


@pytest.fixture
def population_infections_factory(
    gen_int_rv: DeterministicPMF,
) -> Callable[..., PopulationInfections]:
    """
    Factory for PopulationInfections with optional infection feedback.

    Returns
    -------
    Callable[..., PopulationInfections]
        Function that builds the standard configuration with the given
        feedback options.
    """

    def make(
        infection_feedback_strength_rv: RandomVariable | None = None,
        infection_feedback_pmf_rv: RandomVariable | None = None,
    ) -> PopulationInfections:
        """
        Build PopulationInfections with the given feedback options.

        Parameters
        ----------
        infection_feedback_strength_rv
            Infection feedback strength, or None.
        infection_feedback_pmf_rv
            Infection feedback PMF, or None.

        Returns
        -------
        PopulationInfections
            Configured infection process.
        """
        return PopulationInfections(
            name="population",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
            single_rt_process=fixed_ar1(autoreg=0.9, innovation_sd=0.05),
            n_initialization_points=7,
            infection_feedback_strength_rv=infection_feedback_strength_rv,
            infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        )

    return make


@pytest.fixture
def subpopulation_infections(
    subpopulation_infections_factory: Callable[..., SubpopulationInfections],
) -> SubpopulationInfections:
    """
    Pre-configured SubpopulationInfections instance.

    Returns
    -------
    SubpopulationInfections
        Configured infection process with realistic parameters.
    """
    return subpopulation_infections_factory()


@pytest.fixture
def population_infections(
    population_infections_factory: Callable[..., PopulationInfections],
) -> PopulationInfections:
    """
    Pre-configured PopulationInfections instance.

    Returns
    -------
    PopulationInfections
        Configured infection process with realistic parameters.
    """
    return population_infections_factory()


# =============================================================================
# PopulationCounts Process Fixtures
# =============================================================================


@pytest.fixture
def counts_process(simple_delay_pmf):
    """
    Standard PopulationCounts observation process with simple delay.

    Returns
    -------
    PopulationCounts
        A PopulationCounts observation process with no delay.
    """
    return PopulationCounts(
        name="test_counts",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
    )


class CountsProcessFactory:
    """Factory for creating PopulationCounts observation processes with custom parameters."""

    @staticmethod
    def create(
        name="test_counts",
        delay_pmf=None,
        ascertainment_rate=0.01,
        concentration=10.0,
    ):
        """
        Create a PopulationCounts observation process with specified parameters.

        Returns
        -------
        PopulationCounts
            A PopulationCounts observation process with the specified parameters.
        """
        if delay_pmf is None:
            delay_pmf = jnp.array([1.0])
        return PopulationCounts(
            name=name,
            ascertainment_rate_rv=DeterministicVariable("ihr", ascertainment_rate),
            delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
            noise=NegativeBinomialNoise(DeterministicVariable("conc", concentration)),
        )


@pytest.fixture
def counts_factory():
    """
    Factory fixture for creating custom PopulationCounts observation processes.

    Returns
    -------
    CountsProcessFactory
        A factory for creating Counts processes.
    """
    return CountsProcessFactory()


@pytest.fixture
def weekly_regular_counts(simple_delay_pmf):
    """
    PopulationCounts with weekly aggregation and regular (dense) reporting.

    Uses MMWR Sunday-Saturday epiweeks (``start_dow=6``).

    Returns
    -------
    PopulationCounts
        A weekly-regular PopulationCounts process.
    """
    return PopulationCounts(
        name="hosp",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=PoissonNoise(),
        aggregation="weekly",
        reporting_schedule="regular",
        start_dow=MMWR_WEEK,
    )


@pytest.fixture
def weekly_irregular_counts(simple_delay_pmf):
    """
    PopulationCounts with weekly aggregation and irregular (sparse) reporting.

    Uses MMWR Sunday-Saturday epiweeks (``start_dow=6``).

    Returns
    -------
    PopulationCounts
        A weekly-irregular PopulationCounts process.
    """
    return PopulationCounts(
        name="hosp",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=PoissonNoise(),
        aggregation="weekly",
        reporting_schedule="irregular",
        start_dow=MMWR_WEEK,
    )


@pytest.fixture
def daily_irregular_counts(simple_delay_pmf):
    """
    PopulationCounts with daily scale and irregular (sparse) reporting.

    Returns
    -------
    PopulationCounts
        A daily-irregular PopulationCounts process.
    """
    return PopulationCounts(
        name="hosp",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=PoissonNoise(),
        reporting_schedule="irregular",
    )


@pytest.fixture
def weekly_regular_subpop_counts(simple_delay_pmf):
    """
    SubpopulationCounts with weekly aggregation and regular (dense) reporting.

    Uses MMWR Sunday-Saturday epiweeks (``start_dow=6``).

    Returns
    -------
    SubpopulationCounts
        A weekly-regular SubpopulationCounts process.
    """
    return SubpopulationCounts(
        name="ed",
        ascertainment_rate_rv=DeterministicVariable("iedr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=PoissonNoise(),
        aggregation="weekly",
        reporting_schedule="regular",
        start_dow=MMWR_WEEK,
    )


@pytest.fixture
def weekly_irregular_subpop_counts(simple_delay_pmf):
    """
    SubpopulationCounts with weekly aggregation and irregular (sparse) reporting.

    Uses MMWR Sunday-Saturday epiweeks (``start_dow=6``).

    Returns
    -------
    SubpopulationCounts
        A weekly-irregular SubpopulationCounts process.
    """
    return SubpopulationCounts(
        name="ed",
        ascertainment_rate_rv=DeterministicVariable("iedr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=PoissonNoise(),
        aggregation="weekly",
        reporting_schedule="irregular",
        start_dow=MMWR_WEEK,
    )


@pytest.fixture
def daily_irregular_subpop_counts(simple_delay_pmf):
    """
    SubpopulationCounts with daily scale and irregular (sparse) reporting.

    Returns
    -------
    SubpopulationCounts
        A daily-irregular SubpopulationCounts process.
    """
    return SubpopulationCounts(
        name="ed",
        ascertainment_rate_rv=DeterministicVariable("iedr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=PoissonNoise(),
        reporting_schedule="irregular",
    )


# =============================================================================
# Infection Fixtures
# =============================================================================


@pytest.fixture
def subpop_infections_28d():
    """
    Four-week ``(28, 3)`` infections array at 100/day/subpop.

    Used by weekly-aggregation tests on ``SubpopulationCounts``.

    Returns
    -------
    jnp.ndarray
        Shape ``(28, 3)`` filled with 100.0.
    """
    return jnp.ones((28, 3)) * 100.0


@pytest.fixture
def subpop_infections_30d():
    """
    Thirty-day ``(30, 3)`` infections array at 100/day/subpop.

    Used by daily-scale tests on ``SubpopulationCounts``.

    Returns
    -------
    jnp.ndarray
        Shape ``(30, 3)`` filled with 100.0.
    """
    return jnp.ones((30, 3)) * 100.0


@pytest.fixture
def mmwr_saturday_indices_first_three():
    """
    Daily-axis indices of the first three MMWR Saturdays for a Sunday-start axis.

    When element 0 of the daily axis is a Sunday (``first_day_dow=6``),
    Saturdays fall on indices 6, 13, 20, ...

    Returns
    -------
    jnp.ndarray
        Shape ``(3,)`` containing ``[6, 13, 20]``.
    """
    return jnp.array([6, 13, 20])


# =============================================================================
# Temporal Process Stubs
# =============================================================================


class WrongShapeTemporalProcess:
    """Temporal process stub that returns a fixed wrong-shaped array."""

    step_size = 1

    def __init__(self, value):
        """
        Store the wrong-shaped value to return from ``sample``.

        Parameters
        ----------
        value
            Array returned by ``sample`` regardless of requested shape.
        """
        self.value = value

    def sample(self, **kwargs):
        """
        Return the configured wrong-shaped value.

        Returns
        -------
        ArrayLike
            The array passed to ``__init__``.
        """
        return self.value


class ConstantTemporalProcess:
    """Temporal process stub that returns zeros with the requested shape."""

    step_size = 1

    def sample(self, n_timepoints, n_processes=1, **kwargs):
        """
        Return a correctly shaped zero trajectory.

        Parameters
        ----------
        n_timepoints
            Number of time points.
        n_processes
            Number of parallel processes.

        Returns
        -------
        jnp.ndarray
            Zeros of shape ``(n_timepoints, n_processes)``.
        """
        return jnp.zeros((n_timepoints, n_processes))


@pytest.fixture
def wrong_shape_temporal_process_cls():
    """
    Class producing a temporal-process stub that returns a fixed wrong shape.

    Returns
    -------
    type
        The ``WrongShapeTemporalProcess`` class. Instantiate with the array
        the stub should return from ``sample``.
    """
    return WrongShapeTemporalProcess


@pytest.fixture
def constant_temporal_process():
    """
    Temporal-process stub that returns a correctly shaped zero trajectory.

    Returns
    -------
    ConstantTemporalProcess
        Instance whose ``sample`` returns zeros of the requested shape.
    """
    return ConstantTemporalProcess()
