"""
Shared pytest fixtures for PyRenew tests.

This module provides reusable fixtures for creating observation processes,
test data, and common configurations used across multiple test files.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import AR1, HierarchicalInfections, RandomWalk
from pyrenew.observation import (
    Counts,
    HierarchicalNormalNoise,
    NegativeBinomialNoise,
    VectorizedRV,
)
from pyrenew.randomvariable import DistributionalVariable

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
    Standard HierarchicalNormalNoise with VectorizedRV wrappers.

    Returns
    -------
    HierarchicalNormalNoise
        Noise model for continuous measurements.
    """
    sensor_mode_rv = VectorizedRV(
        name="sensor_mode_rv",
        rv=DistributionalVariable("ww_sensor_mode", dist.Normal(0, 0.5)),
        plate_name="sensor_mode",
    )
    sensor_sd_rv = VectorizedRV(
        name="sensor_sd_rv",
        rv=DistributionalVariable(
            "ww_sensor_sd", dist.TruncatedNormal(0.3, 0.15, low=0.10)
        ),
        plate_name="sensor_sd",
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
    sensor_mode_rv = VectorizedRV(
        name="sensor_mode_rv",
        rv=DistributionalVariable("ww_sensor_mode", dist.Normal(0, 0.01)),
        plate_name="sensor_mode",
    )
    sensor_sd_rv = VectorizedRV(
        name="sensor_sd_rv",
        rv=DistributionalVariable(
            "ww_sensor_sd", dist.TruncatedNormal(0.01, 0.005, low=0.001)
        ),
        plate_name="sensor_sd",
    )
    return HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)


# =============================================================================
# Hierarchical Infections Fixture
# =============================================================================


@pytest.fixture
def hierarchical_infections(gen_int_rv):
    """
    Pre-configured HierarchicalInfections instance.

    Returns
    -------
    HierarchicalInfections
        Configured infection process with realistic parameters.
    """
    return HierarchicalInfections(
        gen_int_rv=gen_int_rv,
        I0_rv=DeterministicVariable("I0", 0.001),
        initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
        baseline_rt_process=AR1(autoreg=0.9, innovation_sd=0.05),
        subpop_rt_deviation_process=RandomWalk(innovation_sd=0.025),
        n_initialization_points=7,
    )


# =============================================================================
# Counts Process Fixtures
# =============================================================================


@pytest.fixture
def counts_process(simple_delay_pmf):
    """
    Standard Counts observation process with simple delay.

    Returns
    -------
    Counts
        A Counts observation process with no delay.
    """
    return Counts(
        name="test_counts",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", simple_delay_pmf),
        noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
    )


class CountsProcessFactory:
    """Factory for creating Counts processes with custom parameters."""

    @staticmethod
    def create(
        name="test_counts",
        delay_pmf=None,
        ascertainment_rate=0.01,
        concentration=10.0,
    ):
        """
        Create a Counts process with specified parameters.

        Returns
        -------
        Counts
            A Counts observation process with the specified parameters.
        """
        if delay_pmf is None:
            delay_pmf = jnp.array([1.0])
        return Counts(
            name=name,
            ascertainment_rate_rv=DeterministicVariable("ihr", ascertainment_rate),
            delay_distribution_rv=DeterministicPMF("delay", delay_pmf),
            noise=NegativeBinomialNoise(DeterministicVariable("conc", concentration)),
        )


@pytest.fixture
def counts_factory():
    """
    Factory fixture for creating custom Counts processes.

    Returns
    -------
    CountsProcessFactory
        A factory for creating Counts processes.
    """
    return CountsProcessFactory()


# =============================================================================
# Infection Fixtures
# =============================================================================
