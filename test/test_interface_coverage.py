"""
Interface contract tests for coverage recovery.

These parametrized tests exercise __repr__, validate(), infection_resolution(),
and get_required_lookback() across all classes that implement them, ensuring
the interface contracts are covered without per-class boilerplate tests.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    AR1,
    DifferencedAR1,
    GammaGroupSdPrior,
    HierarchicalInfections,
    HierarchicalNormalPrior,
    RandomWalk,
    StudentTGroupModePrior,
)
from pyrenew.observation import (
    Counts,
    CountsBySubpop,
    HierarchicalNormalNoise,
    NegativeBinomialNoise,
    PoissonNoise,
    VectorizedRV,
)
from pyrenew.randomvariable import DistributionalVariable
from test.test_helpers import ConcreteMeasurements

# =============================================================================
# Shared instance builders
# =============================================================================


def _make_counts():
    """
    Build a Counts instance.

    Returns
    -------
    instantiated object
    """
    return Counts(
        name="test",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", jnp.array([1.0])),
        noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
    )


def _make_counts_by_subpop():
    """
    Build a CountsBySubpop instance.

    Returns
    -------
    instantiated object
    """
    return CountsBySubpop(
        name="test_subpop",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", jnp.array([1.0])),
        noise=PoissonNoise(),
    )


def _make_measurements():
    """
    Build a ConcreteMeasurements instance.

    Returns
    -------
    instantiated object
    """
    sensor_mode_rv = VectorizedRV(
        DistributionalVariable("mode", dist.Normal(0, 0.5)),
        plate_name="sensor_mode",
    )
    sensor_sd_rv = VectorizedRV(
        DistributionalVariable("sd", dist.TruncatedNormal(0.3, 0.15, low=0.1)),
        plate_name="sensor_sd",
    )
    return ConcreteMeasurements(
        name="test_ww",
        temporal_pmf_rv=DeterministicPMF("shed", jnp.array([0.3, 0.4, 0.3])),
        noise=HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv),
    )


def _make_hierarchical_normal_noise():
    """
    Build a HierarchicalNormalNoise instance.

    Returns
    -------
    instantiated object
    """
    sensor_mode_rv = VectorizedRV(
        DistributionalVariable("mode", dist.Normal(0, 0.5)),
        plate_name="sensor_mode",
    )
    sensor_sd_rv = VectorizedRV(
        DistributionalVariable("sd", dist.TruncatedNormal(0.3, 0.15, low=0.1)),
        plate_name="sensor_sd",
    )
    return HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)


# =============================================================================
# __repr__ coverage
# =============================================================================


@pytest.mark.parametrize(
    "instance",
    [
        pytest.param(AR1(autoreg=0.9, innovation_sd=0.1), id="AR1"),
        pytest.param(
            DifferencedAR1(autoreg=0.8, innovation_sd=0.2), id="DifferencedAR1"
        ),
        pytest.param(RandomWalk(innovation_sd=0.5), id="RandomWalk"),
        pytest.param(_make_counts(), id="Counts"),
        pytest.param(_make_counts_by_subpop(), id="CountsBySubpop"),
        pytest.param(_make_measurements(), id="Measurements"),
        pytest.param(PoissonNoise(), id="PoissonNoise"),
        pytest.param(
            NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
            id="NegativeBinomialNoise",
        ),
        pytest.param(_make_hierarchical_normal_noise(), id="HierarchicalNormalNoise"),
    ],
)
def test_repr_returns_nonempty_string(instance):
    """All classes with __repr__ return a non-empty string containing the class name."""
    result = repr(instance)
    assert isinstance(result, str)
    assert len(result) > 0
    assert type(instance).__name__ in result


# =============================================================================
# validate() coverage (no-op and real)
# =============================================================================


@pytest.mark.parametrize(
    "instance",
    [
        pytest.param(
            HierarchicalNormalPrior(
                name="test", sd_rv=DeterministicVariable("sd", 1.0)
            ),
            id="HierarchicalNormalPrior",
        ),
        pytest.param(
            GammaGroupSdPrior(
                name="test",
                sd_mean_rv=DeterministicVariable("mean", 0.5),
                sd_concentration_rv=DeterministicVariable("conc", 10.0),
            ),
            id="GammaGroupSdPrior",
        ),
        pytest.param(
            StudentTGroupModePrior(
                name="test",
                sd_rv=DeterministicVariable("sd", 1.0),
                df_rv=DeterministicVariable("df", 5.0),
            ),
            id="StudentTGroupModePrior",
        ),
        pytest.param(PoissonNoise(), id="PoissonNoise"),
        pytest.param(_make_hierarchical_normal_noise(), id="HierarchicalNormalNoise"),
        pytest.param(_make_counts(), id="Counts"),
    ],
)
def test_validate_does_not_raise(instance):
    """validate() completes without error on well-formed instances."""
    instance.validate()


# =============================================================================
# infection_resolution() coverage
# =============================================================================


def test_counts_by_subpop_infection_resolution():
    """CountsBySubpop.infection_resolution() returns 'subpop'."""
    counts = _make_counts_by_subpop()
    assert counts.infection_resolution() == "subpop"


def test_measurements_infection_resolution():
    """ConcreteMeasurements.infection_resolution() returns 'subpop'."""
    m = _make_measurements()
    assert m.infection_resolution() == "subpop"


def test_base_count_observation_infection_resolution_raises():
    """Base _CountBase.infection_resolution() raises NotImplementedError."""
    from pyrenew.observation.count_observations import _CountBase

    class _MinimalCounts(_CountBase):
        """Minimal subclass that inherits infection_resolution unchanged."""

        def sample(self, *args, **kwargs):  # numpydoc ignore=GL08
            pass

    obs = _MinimalCounts(
        name="test_base",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=DeterministicPMF("delay", jnp.array([1.0])),
        noise=PoissonNoise(),
    )
    with pytest.raises(NotImplementedError):
        obs.infection_resolution()


# =============================================================================
# get_required_lookback() coverage
# =============================================================================


def test_get_required_lookback(gen_int_rv):
    """get_required_lookback returns generation interval PMF length."""
    infections = HierarchicalInfections(
        gen_int_rv=gen_int_rv,
        I0_rv=DeterministicVariable("I0", 0.001),
        initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
        baseline_rt_process=AR1(autoreg=0.9, innovation_sd=0.05),
        subpop_rt_deviation_process=RandomWalk(innovation_sd=0.025),
        n_initialization_points=7,
    )
    expected_length = len(gen_int_rv())
    assert infections.get_required_lookback() == expected_length


# =============================================================================
# HierarchicalInfections.validate() coverage
# =============================================================================


def test_hierarchical_infections_validate(gen_int_rv):
    """HierarchicalInfections.validate() runs without error on valid PMF."""
    infections = HierarchicalInfections(
        gen_int_rv=gen_int_rv,
        I0_rv=DeterministicVariable("I0", 0.001),
        initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
        baseline_rt_process=AR1(autoreg=0.9, innovation_sd=0.05),
        subpop_rt_deviation_process=RandomWalk(innovation_sd=0.025),
        n_initialization_points=7,
    )
    infections.validate()


# =============================================================================
# MultiSignalModel._validate_observation_resolutions error path
# =============================================================================


def test_multisignal_model_rejects_invalid_resolution():
    """MultiSignalModel rejects observation with invalid infection_resolution."""
    from pyrenew.model.multisignal_model import MultiSignalModel

    class BadObservation:  # numpydoc ignore=GL08
        def infection_resolution(self):  # numpydoc ignore=GL08
            return "invalid"

    with pytest.raises(ValueError, match="invalid infection_resolution"):
        MultiSignalModel(
            latent_process=None,
            observations={"bad": BadObservation()},
        )
