"""
Interface contract tests for coverage recovery.

These parametrized tests exercise __repr__, validate(), infection_resolution(),
get_required_lookback(), and RandomVariable.name across all classes that
implement them, ensuring the interface contracts are covered without per-class
boilerplate tests.
"""

import operator

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pyrenew.deterministic import (
    DeterministicPMF,
    DeterministicVariable,
    NullObservation,
    NullVariable,
)
from pyrenew.latent import (
    AR1,
    DifferencedAR1,
    GammaGroupSdPrior,
    HierarchicalInfections,
    HierarchicalNormalPrior,
    Infections,
    InfectionsWithFeedback,
    RandomWalk,
    StudentTGroupModePrior,
)
from pyrenew.metaclass import RandomVariable
from pyrenew.observation import (
    Counts,
    CountsBySubpop,
    HierarchicalNormalNoise,
    NegativeBinomialNoise,
    NegativeBinomialObservation,
    PoissonNoise,
    VectorizedRV,
)
from pyrenew.process import ARProcess, DifferencedProcess
from pyrenew.process.iidrandomsequence import IIDRandomSequence, StandardNormalSequence
from pyrenew.process.periodiceffect import DayOfWeekEffect, PeriodicEffect
from pyrenew.process.randomwalk import RandomWalk as ProcessRandomWalk
from pyrenew.process.randomwalk import StandardNormalRandomWalk
from pyrenew.process.rtperiodicdiffar import RtPeriodicDiffARProcess
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable
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
        name="sensor_mode_rv",
        rv=DistributionalVariable("mode", dist.Normal(0, 0.5)),
        plate_name="sensor_mode",
    )
    sensor_sd_rv = VectorizedRV(
        name="sensor_sd_rv",
        rv=DistributionalVariable("sd", dist.TruncatedNormal(0.3, 0.15, low=0.1)),
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
        name="sensor_mode_rv",
        rv=DistributionalVariable("mode", dist.Normal(0, 0.5)),
        plate_name="sensor_mode",
    )
    sensor_sd_rv = VectorizedRV(
        name="sensor_sd_rv",
        rv=DistributionalVariable("sd", dist.TruncatedNormal(0.3, 0.15, low=0.1)),
        plate_name="sensor_sd",
    )
    return HierarchicalNormalNoise(sensor_mode_rv, sensor_sd_rv)


def _make_rt_periodic():
    """
    Build an RtPeriodicDiffARProcess instance with name "test_rt_periodic".

    Returns
    -------
    RtPeriodicDiffARProcess
    """
    return RtPeriodicDiffARProcess(
        name="test_rt_periodic",
        offset=0,
        period_size=7,
        log_rt_rv=DeterministicVariable("log_rt", jnp.array([0.1, 0.2])),
        autoreg_rv=DeterministicVariable("autoreg", jnp.array([0.7])),
        periodic_diff_sd_rv=DeterministicVariable("sd", jnp.array([0.1])),
    )


def _make_infections_with_feedback():
    """
    Build an InfectionsWithFeedback instance with name "test_inf_feedback".

    Returns
    -------
    InfectionsWithFeedback
    """
    return InfectionsWithFeedback(
        name="test_inf_feedback",
        infection_feedback_strength=DeterministicVariable("fb_strength", 0.5),
        infection_feedback_pmf=DeterministicPMF(
            "fb_pmf", jnp.array([0.4, 0.3, 0.2, 0.1])
        ),
    )


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

        def validate_data(self, n_total, n_subpops, **obs_data):  # numpydoc ignore=GL08
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


# =============================================================================
# RandomVariable.name validation (ABC-level)
# =============================================================================


class _MinimalRV(RandomVariable):
    """Minimal concrete RandomVariable for testing ABC behavior."""

    def sample(self, **kwargs):  # numpydoc ignore=GL08
        pass

    @staticmethod
    def validate():  # numpydoc ignore=GL08
        pass


@pytest.mark.parametrize(
    "bad_name",
    [
        pytest.param("", id="empty_string"),
        pytest.param(None, id="none"),
        pytest.param(123, id="integer"),
    ],
)
def test_random_variable_rejects_invalid_name(bad_name):
    """RandomVariable.__init__ rejects non-string and empty names."""
    with pytest.raises(TypeError, match="name must be a non-empty string"):
        _MinimalRV(name=bad_name)


# =============================================================================
# RandomVariable.name attribute (functional checks)
# =============================================================================


@pytest.mark.parametrize(
    "instance, expected_name",
    [
        pytest.param(
            DeterministicVariable("det_var", 1.0),
            "det_var",
            id="DeterministicVariable",
        ),
        pytest.param(
            DeterministicPMF("det_pmf", jnp.array([1.0])),
            "det_pmf",
            id="DeterministicPMF",
        ),
        pytest.param(NullVariable(), "null", id="NullVariable"),
        pytest.param(NullObservation(), "null_observation", id="NullObservation"),
        pytest.param(
            DistributionalVariable("dist_var", dist.Normal(0, 1)),
            "dist_var",
            id="DistributionalVariable",
        ),
        pytest.param(
            TransformedVariable(
                "trans_var",
                DeterministicVariable("base", 1.0),
                lambda x: x,
            ),
            "trans_var",
            id="TransformedVariable",
        ),
        pytest.param(ARProcess(name="test_ar"), "test_ar", id="ARProcess"),
        pytest.param(
            DifferencedProcess(
                name="test_diff",
                fundamental_process=ARProcess(name="inner"),
                differencing_order=1,
            ),
            "test_diff",
            id="DifferencedProcess",
        ),
        pytest.param(
            IIDRandomSequence(
                name="test_iid",
                element_rv=DistributionalVariable("el", dist.Normal(0, 1)),
            ),
            "test_iid",
            id="IIDRandomSequence",
        ),
        pytest.param(
            StandardNormalSequence(name="test_sns", element_rv_name="el"),
            "test_sns",
            id="StandardNormalSequence",
        ),
        pytest.param(
            ProcessRandomWalk(
                name="test_rw",
                step_rv=DistributionalVariable("step", dist.Normal(0, 1)),
            ),
            "test_rw",
            id="ProcessRandomWalk",
        ),
        pytest.param(
            StandardNormalRandomWalk(name="test_snrw", step_rv_name="step"),
            "test_snrw",
            id="StandardNormalRandomWalk",
        ),
        pytest.param(
            PeriodicEffect(
                name="test_pe",
                offset=0,
                quantity_to_broadcast=DeterministicVariable(
                    "qty", jnp.array([1.0, 2.0])
                ),
            ),
            "test_pe",
            id="PeriodicEffect",
        ),
        pytest.param(
            DayOfWeekEffect(
                name="test_dow",
                offset=0,
                quantity_to_broadcast=DeterministicVariable("qty", jnp.ones(7)),
            ),
            "test_dow",
            id="DayOfWeekEffect",
        ),
        pytest.param(
            _make_rt_periodic(),
            "test_rt_periodic",
            id="RtPeriodicDiffARProcess",
        ),
        pytest.param(Infections(name="test_inf"), "test_inf", id="Infections"),
        pytest.param(
            _make_infections_with_feedback(),
            "test_inf_feedback",
            id="InfectionsWithFeedback",
        ),
        pytest.param(
            NegativeBinomialObservation(
                name="test_nb",
                concentration_rv=DeterministicVariable("conc", 10.0),
            ),
            "test_nb",
            id="NegativeBinomialObservation",
        ),
        pytest.param(
            HierarchicalNormalPrior(
                name="test_hnp", sd_rv=DeterministicVariable("sd", 1.0)
            ),
            "test_hnp",
            id="HierarchicalNormalPrior",
        ),
        pytest.param(
            GammaGroupSdPrior(
                name="test_gamma",
                sd_mean_rv=DeterministicVariable("mean", 0.5),
                sd_concentration_rv=DeterministicVariable("conc", 10.0),
            ),
            "test_gamma",
            id="GammaGroupSdPrior",
        ),
        pytest.param(
            StudentTGroupModePrior(
                name="test_student",
                sd_rv=DeterministicVariable("sd", 1.0),
                df_rv=DeterministicVariable("df", 5.0),
            ),
            "test_student",
            id="StudentTGroupModePrior",
        ),
        pytest.param(_make_counts(), "test", id="Counts"),
        pytest.param(_make_counts_by_subpop(), "test_subpop", id="CountsBySubpop"),
        pytest.param(_make_measurements(), "test_ww", id="ConcreteMeasurements"),
        pytest.param(
            VectorizedRV(
                name="test_vec",
                rv=DistributionalVariable("inner", dist.Normal(0, 1)),
                plate_name="plate",
            ),
            "test_vec",
            id="VectorizedRV",
        ),
    ],
)
def test_name_attribute_matches_expected(instance, expected_name):
    """RandomVariable.name is correctly set during construction."""
    assert instance.name == expected_name


def test_hierarchical_infections_name(gen_int_rv):
    """HierarchicalInfections.name is correctly set during construction."""
    infections = HierarchicalInfections(
        name="test_hi",
        gen_int_rv=gen_int_rv,
        I0_rv=DeterministicVariable("I0", 0.001),
        initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
        baseline_rt_process=AR1(autoreg=0.9, innovation_sd=0.05),
        subpop_rt_deviation_process=RandomWalk(innovation_sd=0.025),
        n_initialization_points=7,
    )
    assert infections.name == "test_hi"


# =============================================================================
# Internal auto-generated RV names (composition checks)
# =============================================================================


@pytest.mark.parametrize(
    "instance, expected_names",
    [
        pytest.param(
            AR1(autoreg=0.9, innovation_sd=0.1),
            [("ar_process", "ar1")],
            id="AR1",
        ),
        pytest.param(
            DifferencedAR1(autoreg=0.8, innovation_sd=0.2),
            [
                ("process", "diff_ar1"),
                ("process.fundamental_process", "diff_ar1_fundamental"),
            ],
            id="DifferencedAR1",
        ),
        pytest.param(
            _make_rt_periodic(),
            [
                ("ar_diff", "test_rt_periodic_diff"),
                (
                    "ar_diff.fundamental_process",
                    "test_rt_periodic_diff_fundamental",
                ),
            ],
            id="RtPeriodicDiffARProcess",
        ),
        pytest.param(
            ProcessRandomWalk(
                name="test_rw",
                step_rv=DistributionalVariable("step", dist.Normal(0, 1)),
            ),
            [("fundamental_process", "test_rw_iid_seq")],
            id="ProcessRandomWalk",
        ),
        pytest.param(
            StandardNormalRandomWalk(name="test_snrw", step_rv_name="step"),
            [("fundamental_process", "test_snrw_iid_seq")],
            id="StandardNormalRandomWalk",
        ),
    ],
)
def test_internal_rv_names(instance, expected_names):
    """Composite classes generate correct internal RandomVariable names."""
    for attr_path, expected_name in expected_names:
        obj = operator.attrgetter(attr_path)(instance)
        assert obj.name == expected_name
