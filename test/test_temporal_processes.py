"""
Unit tests for temporal processes.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import Predictive

from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    AR1,
    DifferencedAR1,
    RandomWalk,
    StepwiseTemporalProcess,
    WeeklyTemporalProcess,
)
from pyrenew.latent.state_centered_distributions import (
    StateAR1,
    StateDifferencedAR1,
    StateRandomWalk,
)
from pyrenew.randomvariable import DistributionalVariable
from pyrenew.time import MMWR_WEEK


def fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05):
    """
    Build constructor kwargs for an AR1-like process with fixed parameters.

    Returns
    -------
    dict
        Keyword arguments containing deterministic parameter random variables.
    """
    return {
        "autoreg_rv": DeterministicVariable("autoreg", autoreg),
        "innovation_sd_rv": DeterministicVariable("innovation_sd", innovation_sd),
    }


def fixed_rw_kwargs(innovation_sd=0.05):
    """
    Build constructor kwargs for a RandomWalk with a fixed innovation scale.

    Returns
    -------
    dict
        Keyword arguments containing a deterministic innovation scale RV.
    """
    return {
        "innovation_sd_rv": DeterministicVariable("innovation_sd", innovation_sd),
    }


INNER_PROCESS_PARAMS = [
    (AR1, fixed_ar1_kwargs()),
    (DifferencedAR1, fixed_ar1_kwargs()),
    (RandomWalk, fixed_rw_kwargs()),
]


class TestStateCenteredDistributionLogProb:
    """Exact density checks for state-centered temporal-process distributions."""

    def test_state_random_walk_log_prob_matches_manual_transition_sum(self):
        """Batched StateRandomWalk log_prob equals the explicit RW transition density."""
        scale = jnp.array([0.3, 0.7])
        initial_loc = jnp.array([1.0, -0.5])
        value = jnp.array(
            [
                [1.2, 0.6, 0.1, -0.2],
                [-0.3, 0.4, 0.0, 0.2],
            ]
        )

        distribution = StateRandomWalk(
            scale=scale,
            initial_loc=initial_loc,
            num_steps=value.shape[-1],
        )

        full_path = jnp.concatenate([initial_loc[:, None], value], axis=-1)
        expected = dist.Normal(full_path[:, :-1], scale[:, None]).log_prob(
            full_path[:, 1:]
        )
        expected = expected.sum(axis=-1)

        assert jnp.allclose(distribution.log_prob(value), expected)

    def test_state_ar1_log_prob_matches_manual_transition_sum(self):
        """Batched StateAR1 log_prob equals the explicit AR1 transition density."""
        autoreg = jnp.array([0.4, -0.2])
        scale = jnp.array([0.3, 0.7])
        initial_loc = jnp.array([1.0, -0.5])
        value = jnp.array(
            [
                [1.2, 0.6, 0.1, -0.2],
                [-0.3, 0.4, 0.0, 0.2],
            ]
        )

        distribution = StateAR1(
            autoreg=autoreg,
            scale=scale,
            initial_loc=initial_loc,
            num_steps=value.shape[-1],
        )

        stationary_sd = scale / jnp.sqrt(1 - autoreg**2)
        init_prob = dist.Normal(initial_loc, stationary_sd).log_prob(value[:, 0])
        transition_locs = autoreg[:, None] * value[:, :-1]
        transition_probs = dist.Normal(transition_locs, scale[:, None]).log_prob(
            value[:, 1:]
        )
        expected = init_prob + transition_probs.sum(axis=-1)

        assert jnp.allclose(distribution.log_prob(value), expected)

    def test_state_differenced_ar1_log_prob_matches_manual_transition_sum(self):
        """Batched StateDifferencedAR1 log_prob equals the explicit transition density."""
        autoreg = jnp.array([0.6, -0.3])
        scale = jnp.array([0.2, 0.5])
        initial_loc = jnp.array([1.0, -0.5])
        value = jnp.array(
            [
                [1.1, 1.4, 1.45, 1.7],
                [-0.6, -0.4, -0.1, -0.2],
            ]
        )

        distribution = StateDifferencedAR1(
            autoreg=autoreg,
            scale=scale,
            initial_loc=initial_loc,
            num_steps=value.shape[-1],
        )

        stationary_sd = scale / jnp.sqrt(1 - autoreg**2)
        init_prob = dist.Normal(initial_loc, stationary_sd).log_prob(value[:, 0])
        full_path = jnp.concatenate([initial_loc[:, None], value], axis=-1)
        previous_delta = full_path[:, 1:-1] - full_path[:, :-2]
        transition_locs = full_path[:, 1:-1] + autoreg[:, None] * previous_delta
        transition_probs = dist.Normal(transition_locs, scale[:, None]).log_prob(
            full_path[:, 2:]
        )
        expected = init_prob + transition_probs.sum(axis=-1)

        assert jnp.allclose(distribution.log_prob(value), expected)


class TestStateCenteredDistributionValidationAndSampling:
    """Focused coverage for state-centered distribution validation branches."""

    @pytest.mark.parametrize(
        "distribution_cls,kwargs",
        [
            (StateRandomWalk, {"scale": 1.0}),
            (StateAR1, {"autoreg": 0.5, "scale": 1.0}),
            (StateDifferencedAR1, {"autoreg": 0.5, "scale": 1.0}),
        ],
    )
    @pytest.mark.parametrize("invalid_num_steps", [0, 1.5])
    def test_num_steps_must_be_positive_integer(
        self, distribution_cls, kwargs, invalid_num_steps
    ):
        """Constructors reject non-positive and non-integer step counts."""
        with pytest.raises(ValueError, match="num_steps must be a positive integer"):
            distribution_cls(**kwargs, num_steps=invalid_num_steps)

    def test_state_differenced_ar1_single_step_sample_matches_initial_transition(self):
        """Single-step differenced AR(1) sampling returns only the first transition."""
        key = jax.random.PRNGKey(43)
        autoreg = jnp.array([0.2, -0.4])
        scale = jnp.array([0.5, 0.25])
        initial_loc = jnp.array([1.0, -2.0])

        distribution = StateDifferencedAR1(
            autoreg=autoreg,
            scale=scale,
            initial_loc=initial_loc,
            num_steps=1,
        )

        sample = distribution.sample(key)
        stationary_sd = scale / jnp.sqrt(1 - autoreg**2)
        expected_noise = jax.random.normal(key, shape=(1, 2))[0]
        expected = initial_loc + stationary_sd * expected_noise

        assert sample.shape == (2, 1)
        assert jnp.allclose(sample[:, 0], expected)


class TestTemporalProcessVectorizedSampling:
    """Test vectorized sampling across all temporal process types."""

    @pytest.mark.parametrize(
        "process_cls,kwargs",
        [
            (AR1, fixed_ar1_kwargs()),
            (DifferencedAR1, fixed_ar1_kwargs()),
            (RandomWalk, fixed_rw_kwargs()),
        ],
    )
    def test_vectorized_shape_and_initial_values_array(self, process_cls, kwargs):
        """Test shape and initial value handling with array initial values."""
        n_timepoints = 30
        n_processes = 4
        initial_values = jnp.array([0.0, 1.0, -1.0, 2.0])

        process = process_cls(**kwargs)

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = process.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=initial_values,
            )

        assert trajectories.shape == (n_timepoints, n_processes)

    @pytest.mark.parametrize(
        "process_cls,kwargs",
        [
            (AR1, fixed_ar1_kwargs()),
            (DifferencedAR1, fixed_ar1_kwargs()),
            (RandomWalk, fixed_rw_kwargs()),
        ],
    )
    def test_vectorized_shape_with_scalar_initial_value(self, process_cls, kwargs):
        """Test shape with scalar initial value broadcast."""
        n_timepoints = 30
        n_processes = 3

        process = process_cls(**kwargs)

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = process.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=1.0,
            )

        assert trajectories.shape == (n_timepoints, n_processes)


class TestRandomWalkInitialValues:
    """Test that RandomWalk preserves initial values."""

    def test_random_walk_vectorized_with_initial_values_array(self):
        """Test RandomWalk first row equals initial values."""
        n_timepoints = 30
        n_processes = 4
        initial_values = jnp.array([0.0, 1.0, -1.0, 2.0])

        rw = RandomWalk(**fixed_rw_kwargs(innovation_sd=0.05))

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = rw.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=initial_values,
            )

        assert trajectories.shape == (n_timepoints, n_processes)
        assert jnp.allclose(trajectories[0, :], initial_values)

    def test_random_walk_vectorized_with_scalar_initial_value(self):
        """Test RandomWalk first row equals broadcast scalar."""
        n_timepoints = 30
        n_processes = 3

        rw = RandomWalk(**fixed_rw_kwargs(innovation_sd=0.05))

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = rw.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=1.0,
            )

        assert trajectories.shape == (n_timepoints, n_processes)
        assert jnp.allclose(trajectories[0, :], 1.0)


class TestTemporalProcessRandomVariableParameters:
    """Focused tests for RandomVariable-backed temporal process parameters."""

    @pytest.mark.parametrize(
        "process_cls,kwargs,error_match",
        [
            (
                AR1,
                {
                    "autoreg_rv": 0.9,
                    "innovation_sd_rv": DeterministicVariable("innovation_sd", 0.05),
                },
                "autoreg_rv must be a RandomVariable",
            ),
            (
                AR1,
                {
                    "autoreg_rv": DeterministicVariable("autoreg", 0.9),
                    "innovation_sd_rv": 0.05,
                },
                "innovation_sd_rv must be a RandomVariable",
            ),
            (
                DifferencedAR1,
                {
                    "autoreg_rv": 0.9,
                    "innovation_sd_rv": DeterministicVariable("innovation_sd", 0.05),
                },
                "autoreg_rv must be a RandomVariable",
            ),
            (
                DifferencedAR1,
                {
                    "autoreg_rv": DeterministicVariable("autoreg", 0.9),
                    "innovation_sd_rv": 0.05,
                },
                "innovation_sd_rv must be a RandomVariable",
            ),
            (
                RandomWalk,
                {"innovation_sd_rv": 0.05},
                "innovation_sd_rv must be a RandomVariable",
            ),
        ],
    )
    def test_constructor_rejects_non_random_variable_args(
        self, process_cls, kwargs, error_match
    ):
        """Reject constructor parameters that are not RandomVariables."""
        with pytest.raises(TypeError, match=error_match):
            process_cls(**kwargs)

    @pytest.mark.parametrize(
        "process,expected_sites",
        [
            (
                AR1(
                    autoreg_rv=DistributionalVariable("autoreg", dist.Beta(9, 1)),
                    innovation_sd_rv=DistributionalVariable(
                        "innovation_sd", dist.HalfNormal(0.1)
                    ),
                ),
                {"autoreg", "innovation_sd", "ar1_init", "ar1_noise_decentered"},
            ),
            (
                DifferencedAR1(
                    autoreg_rv=DistributionalVariable("autoreg", dist.Beta(9, 1)),
                    innovation_sd_rv=DistributionalVariable(
                        "innovation_sd", dist.HalfNormal(0.1)
                    ),
                ),
                {
                    "autoreg",
                    "innovation_sd",
                    "diff_ar1_init_rate",
                    "diff_ar1_noise_decentered",
                },
            ),
            (
                RandomWalk(
                    innovation_sd_rv=DistributionalVariable(
                        "innovation_sd", dist.HalfNormal(0.1)
                    )
                ),
                {"innovation_sd", "rw_step"},
            ),
        ],
    )
    def test_distributional_parameter_rvs_create_expected_sample_sites(
        self, process, expected_sites
    ):
        """Distributional parameter RVs create parameter and process sample sites."""
        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(process.sample, rng_seed=42)
        ).get_trace(n_timepoints=10, n_processes=2)

        for site in expected_sites:
            assert site in traced
            assert traced[site]["type"] == "sample"

    @pytest.mark.parametrize(
        "process",
        [
            AR1(
                autoreg_rv=DistributionalVariable("autoreg", dist.Beta(9, 1)),
                innovation_sd_rv=DistributionalVariable(
                    "innovation_sd", dist.HalfNormal(0.1)
                ),
            ),
            DifferencedAR1(
                autoreg_rv=DistributionalVariable("autoreg", dist.Beta(9, 1)),
                innovation_sd_rv=DistributionalVariable(
                    "innovation_sd", dist.HalfNormal(0.1)
                ),
            ),
            RandomWalk(
                innovation_sd_rv=DistributionalVariable(
                    "innovation_sd", dist.HalfNormal(0.1)
                )
            ),
        ],
    )
    def test_distributional_parameter_rvs_preserve_vectorized_shape(self, process):
        """Distributional parameter RVs preserve the temporal process output shape."""
        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(n_timepoints=10, n_processes=2)

        assert result.shape == (10, 2)


class TestTemporalProcessInnovationSD:
    """Test that temporal processes correctly use innovation_sd parameter."""

    def test_random_walk_smaller_innovation_sd_produces_smoother_trajectory(
        self,
    ):
        """Verify that smaller innovation_sd produces less volatile trajectories."""
        n_timepoints = 100

        with numpyro.handlers.seed(rng_seed=42):
            rw_small = RandomWalk(**fixed_rw_kwargs(innovation_sd=0.1))
            trajectory_small = rw_small.sample(n_timepoints=n_timepoints)

        with numpyro.handlers.seed(rng_seed=42):
            rw_large = RandomWalk(**fixed_rw_kwargs(innovation_sd=1.0))
            trajectory_large = rw_large.sample(n_timepoints=n_timepoints)

        steps_small = jnp.abs(jnp.diff(trajectory_small[:, 0]))
        steps_large = jnp.abs(jnp.diff(trajectory_large[:, 0]))

        assert jnp.mean(steps_small) < jnp.mean(steps_large)
        assert jnp.max(steps_small) < jnp.max(steps_large)

    def test_ar1_smaller_innovation_sd_produces_lower_variance(self):
        """Verify AR1 with smaller innovation_sd produces lower variance trajectories."""
        n_timepoints = 100
        autoreg = 0.7

        with numpyro.handlers.seed(rng_seed=42):
            ar_small = AR1(**fixed_ar1_kwargs(autoreg=autoreg, innovation_sd=0.2))
            trajectory_small = ar_small.sample(n_timepoints=n_timepoints)

        with numpyro.handlers.seed(rng_seed=42):
            ar_large = AR1(**fixed_ar1_kwargs(autoreg=autoreg, innovation_sd=1.0))
            trajectory_large = ar_large.sample(n_timepoints=n_timepoints)

        burn_in = 20
        var_small = jnp.var(trajectory_small[burn_in:, 0])
        var_large = jnp.var(trajectory_large[burn_in:, 0])

        assert var_small < var_large

    def test_differenced_ar1_smaller_innovation_sd_produces_smoother_changes(
        self,
    ):
        """Verify DifferencedAR1 with smaller innovation_sd produces smoother changes."""
        n_timepoints = 100
        autoreg = 0.6

        with numpyro.handlers.seed(rng_seed=42):
            dar_small = DifferencedAR1(
                **fixed_ar1_kwargs(autoreg=autoreg, innovation_sd=0.15)
            )
            trajectory_small = dar_small.sample(n_timepoints=n_timepoints)

        with numpyro.handlers.seed(rng_seed=42):
            dar_large = DifferencedAR1(
                **fixed_ar1_kwargs(autoreg=autoreg, innovation_sd=0.8)
            )
            trajectory_large = dar_large.sample(n_timepoints=n_timepoints)

        diffs_small = jnp.diff(trajectory_small[:, 0])
        diffs_large = jnp.diff(trajectory_large[:, 0])

        assert jnp.std(diffs_small) < jnp.std(diffs_large)

    def test_vectorized_sampling_respects_innovation_sd(self):
        """Verify vectorized sampling uses the innovation_sd parameter."""
        n_processes = 5
        n_timepoints = 50

        with numpyro.handlers.seed(rng_seed=42):
            rw_small = RandomWalk(**fixed_rw_kwargs(innovation_sd=0.2))
            trajs_small = rw_small.sample(
                n_timepoints=n_timepoints, n_processes=n_processes
            )

        with numpyro.handlers.seed(rng_seed=42):
            rw_large = RandomWalk(**fixed_rw_kwargs(innovation_sd=1.0))
            trajs_large = rw_large.sample(
                n_timepoints=n_timepoints, n_processes=n_processes
            )

        steps_small = jnp.abs(jnp.diff(trajs_small, axis=0))
        steps_large = jnp.abs(jnp.diff(trajs_large, axis=0))

        assert jnp.mean(steps_small) < jnp.mean(steps_large)

    @pytest.mark.parametrize(
        "process_factory",
        [
            lambda innovation_sd: AR1(
                autoreg_rv=DeterministicVariable("autoreg", 0.5),
                innovation_sd_rv=DeterministicVariable("innovation_sd", innovation_sd),
            ),
            lambda innovation_sd: DifferencedAR1(
                autoreg_rv=DeterministicVariable("autoreg", 0.5),
                innovation_sd_rv=DeterministicVariable("innovation_sd", innovation_sd),
            ),
            lambda innovation_sd: RandomWalk(
                innovation_sd_rv=DeterministicVariable("innovation_sd", innovation_sd),
            ),
        ],
    )
    @pytest.mark.parametrize("innovation_sd", [0.0, -0.1, jnp.array([0.1, 0.0])])
    def test_validation_rejects_non_positive_deterministic_innovation_sd(
        self, process_factory, innovation_sd
    ):
        """Deterministic innovation scales must be positive."""
        with pytest.raises(ValueError, match="innovation_sd_rv must return positive"):
            process_factory(innovation_sd)

    @pytest.mark.parametrize("process_cls", [AR1, DifferencedAR1])
    def test_ar_processes_accept_distributional_parameter_rvs(self, process_cls):
        """Distributional parameter RVs create sample sites and preserve shape."""
        process = process_cls(
            autoreg_rv=DistributionalVariable("autoreg", dist.Beta(9, 1)),
            innovation_sd_rv=DistributionalVariable(
                "innovation_sd", dist.HalfNormal(0.1)
            ),
        )

        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(process.sample, rng_seed=42)
        ).get_trace(n_timepoints=10, n_processes=2)
        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(n_timepoints=10, n_processes=2)

        assert "autoreg" in traced
        assert "innovation_sd" in traced
        assert traced["autoreg"]["type"] == "sample"
        assert traced["innovation_sd"]["type"] == "sample"
        assert result.shape == (10, 2)

    def test_random_walk_accepts_distributional_innovation_sd_rv(self):
        """RandomWalk samples innovation_sd_rv before constructing step noise."""
        process = RandomWalk(
            innovation_sd_rv=DistributionalVariable(
                "innovation_sd", dist.HalfNormal(0.1)
            )
        )

        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(process.sample, rng_seed=42)
        ).get_trace(n_timepoints=10, n_processes=2)
        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(n_timepoints=10, n_processes=2)

        assert "innovation_sd" in traced
        assert traced["innovation_sd"]["type"] == "sample"
        assert result.shape == (10, 2)


class TestTemporalProcessBehavior:
    """Test behavioral properties of temporal processes."""

    def test_ar1_mean_reversion(self):
        """Test that AR1 reverts toward zero from a displaced initial value."""
        ar1 = AR1(**fixed_ar1_kwargs(autoreg=0.95, innovation_sd=0.05))

        with numpyro.handlers.seed(rng_seed=42):
            trajectory = ar1.sample(
                n_timepoints=100,
                initial_value=2.0,
            )

        # The trajectory mean should be closer to 0 than the initial value
        # due to mean-reverting dynamics
        trajectory_mean = jnp.mean(trajectory[:, 0])
        assert jnp.abs(trajectory_mean) < jnp.abs(2.0)

    def test_differenced_ar1_trend_persistence(self):
        """Test that DifferencedAR1 produces persistent trends."""
        dar1 = DifferencedAR1(**fixed_ar1_kwargs(autoreg=0.95, innovation_sd=0.01))

        with numpyro.handlers.seed(rng_seed=42):
            trajectory = dar1.sample(
                n_timepoints=50,
                initial_value=0.1,
            )

        # With positive initial rate and high autoreg, the differences
        # (growth rates) should remain predominantly positive,
        # producing a persistent upward trend
        diffs = jnp.diff(trajectory[:, 0])
        fraction_positive = jnp.mean(diffs > 0)
        assert fraction_positive > 0.5


class TestTemporalProcessStepSizeDefault:
    """Standard temporal processes expose step_size=1 as a class attribute."""

    @pytest.mark.parametrize(
        "process_cls",
        [AR1, DifferencedAR1, RandomWalk],
    )
    def test_class_attribute_step_size_is_one(self, process_cls):
        """Each concrete class has step_size=1 as a class attribute."""
        assert process_cls.step_size == 1


class TestTemporalProcessRepr:
    """String representations show RandomVariable constructor arguments."""

    @pytest.mark.parametrize(
        "process,expected",
        [
            (
                AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)),
                ("AR1(", "autoreg_rv=", "innovation_sd_rv="),
            ),
            (
                DifferencedAR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)),
                ("DifferencedAR1(", "autoreg_rv=", "innovation_sd_rv="),
            ),
            (
                RandomWalk(**fixed_rw_kwargs(innovation_sd=0.05)),
                ("RandomWalk(", "innovation_sd_rv="),
            ),
        ],
    )
    def test_repr_uses_random_variable_argument_names(self, process, expected):
        """Representations use *_rv constructor argument names."""
        rendered = repr(process)
        for text in expected:
            assert text in rendered


PARAMETERIZATION_FLAG_CASES = [
    (AR1, fixed_ar1_kwargs()),
    (DifferencedAR1, fixed_ar1_kwargs()),
    (RandomWalk, fixed_rw_kwargs()),
]


class TestTemporalProcessParameterizationFlag:
    """Constructor validates and exposes the ``parameterization`` flag."""

    @pytest.mark.parametrize("process_cls,kwargs", PARAMETERIZATION_FLAG_CASES)
    def test_invalid_parameterization_raises(self, process_cls, kwargs):
        """Unknown parameterization strings are rejected at construction."""
        with pytest.raises(ValueError, match="parameterization"):
            process_cls(**kwargs, parameterization="bogus")

    @pytest.mark.parametrize("process_cls,kwargs", PARAMETERIZATION_FLAG_CASES)
    def test_default_parameterization_is_innovation(self, process_cls, kwargs):
        """Constructor default preserves historical innovation behavior."""
        process = process_cls(**kwargs)
        assert process.parameterization == "innovation"

    @pytest.mark.parametrize("process_cls,kwargs", PARAMETERIZATION_FLAG_CASES)
    def test_state_parameterization_stored(self, process_cls, kwargs):
        """``parameterization='state'`` is accepted and stored as attribute."""
        process = process_cls(**kwargs, parameterization="state")
        assert process.parameterization == "state"

    @pytest.mark.parametrize("process_cls,kwargs", PARAMETERIZATION_FLAG_CASES)
    def test_repr_shows_parameterization(self, process_cls, kwargs):
        """``__repr__`` exposes the current parameterization for diagnostics."""
        process = process_cls(**kwargs, parameterization="state")
        assert "parameterization='state'" in repr(process)


class TestStateCenteredRandomWalk:
    """``RandomWalk`` in state mode draws the post-initial path from a single ``rw_state`` site backed by ``StateRandomWalk``."""

    def test_return_shape(self):
        """Return value has shape ``(n_timepoints, n_processes)``."""
        rw = RandomWalk(**fixed_rw_kwargs(innovation_sd=0.1), parameterization="state")
        with numpyro.handlers.seed(rng_seed=0):
            path = rw.sample(n_timepoints=15, n_processes=4, name_prefix="rw")
        assert path.shape == (15, 4)

    def test_initial_row_equals_initial_value(self):
        """``x[0]`` is deterministic and equal to ``initial_value`` for every draw."""
        rw = RandomWalk(**fixed_rw_kwargs(innovation_sd=0.1), parameterization="state")
        init = jnp.array([0.5, -1.0, 2.0])
        with numpyro.handlers.seed(rng_seed=0):
            path = rw.sample(
                n_timepoints=10,
                n_processes=3,
                initial_value=init,
                name_prefix="rw",
            )
        assert jnp.allclose(path[0], init)

    def test_n_timepoints_one_returns_initial_value(self):
        """``n_timepoints=1`` returns just the initial value as shape ``(1, n_processes)``."""
        rw = RandomWalk(**fixed_rw_kwargs(innovation_sd=0.1), parameterization="state")
        init = jnp.array([0.3, 0.7])
        with numpyro.handlers.seed(rng_seed=0):
            path = rw.sample(
                n_timepoints=1,
                n_processes=2,
                initial_value=init,
                name_prefix="rw",
            )
        assert path.shape == (1, 2)
        assert jnp.allclose(path[0], init)

    def test_trace_has_state_site_not_step_site(self):
        """State-mode trace records ``_state``; innovation-mode ``_step`` is absent."""
        rw = RandomWalk(**fixed_rw_kwargs(innovation_sd=0.1), parameterization="state")
        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(rw.sample, rng_seed=0)
        ).get_trace(n_timepoints=8, n_processes=2, name_prefix="rw")
        assert "rw_state" in traced
        assert "rw_step" not in traced

    def test_state_site_contains_actual_post_initial_states(self):
        """The ``_state`` site stores shifted states, not zero-origin offsets."""
        rw = RandomWalk(**fixed_rw_kwargs(innovation_sd=0.1), parameterization="state")
        init = jnp.array([10.0, -10.0])

        def model():
            """Record the sampled path for comparison with the latent state site."""
            path = rw.sample(
                n_timepoints=6,
                n_processes=2,
                initial_value=init,
                name_prefix="rw",
            )
            numpyro.deterministic("path", path)

        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(model, rng_seed=0)
        ).get_trace()
        state_site = traced["rw_state"]["value"]
        path = traced["path"]["value"]
        assert state_site.shape == (2, 5)
        assert jnp.allclose(state_site, path[1:].T)

    @pytest.mark.parametrize(
        "innovation_sd",
        [0.05, jnp.array([0.05, 0.1, 0.07])],
    )
    def test_prior_moments_match_innovation_parameterization(self, innovation_sd):
        """State and innovation parameterizations produce the same per-timepoint moments."""
        n_timepoints = 25
        n_processes = 3
        init = jnp.array([0.0, 0.5, -0.3])

        sd_rv = DeterministicVariable("sigma", innovation_sd)
        rw_state = RandomWalk(sd_rv, parameterization="state")
        rw_innov = RandomWalk(sd_rv, parameterization="innovation")

        def model_state():
            """Record state-centered path as deterministic for Predictive readout."""
            path = rw_state.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=init,
                name_prefix="rw",
            )
            numpyro.deterministic("path", path)

        def model_innov():
            """Record innovation-form path as deterministic for Predictive readout."""
            path = rw_innov.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=init,
                name_prefix="rw",
            )
            numpyro.deterministic("path", path)

        n_samples = 10000
        s_state = Predictive(model_state, num_samples=n_samples)(jax.random.PRNGKey(0))[
            "path"
        ]
        s_innov = Predictive(model_innov, num_samples=n_samples)(jax.random.PRNGKey(1))[
            "path"
        ]

        sigma_max = float(jnp.max(jnp.atleast_1d(jnp.asarray(innovation_sd))))
        terminal_sd = sigma_max * jnp.sqrt(n_timepoints - 1)
        mean_atol = 5.0 * terminal_sd / jnp.sqrt(n_samples)
        assert jnp.allclose(s_state.mean(axis=0), init[jnp.newaxis, :], atol=mean_atol)
        assert jnp.allclose(s_innov.mean(axis=0), init[jnp.newaxis, :], atol=mean_atol)

        assert jnp.allclose(
            s_state.var(axis=0), s_innov.var(axis=0), rtol=0.10, atol=1e-4
        )


class TestStateCenteredAR1:
    """``AR1`` in state mode draws the full path from a single ``ar1_state`` site backed by ``StateAR1``."""

    def test_return_shape(self):
        """Return value has shape ``(n_timepoints, n_processes)``."""
        ar1 = AR1(**fixed_ar1_kwargs(), parameterization="state")
        with numpyro.handlers.seed(rng_seed=0):
            path = ar1.sample(n_timepoints=15, n_processes=4, name_prefix="ar1")
        assert path.shape == (15, 4)

    def test_trace_has_state_site_not_init_or_noise(self):
        """State-mode AR1 trace contains a single ``_state`` site only."""
        ar1 = AR1(**fixed_ar1_kwargs(), parameterization="state")
        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(ar1.sample, rng_seed=0)
        ).get_trace(n_timepoints=8, n_processes=2, name_prefix="ar1")
        assert "ar1_state" in traced
        assert "ar1_init" not in traced
        assert "ar1_noise" not in traced

    def test_state_site_shape(self):
        """The state site holds the full path of shape ``(n_processes, n_timepoints)``."""
        ar1 = AR1(**fixed_ar1_kwargs(), parameterization="state")
        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(ar1.sample, rng_seed=0)
        ).get_trace(n_timepoints=12, n_processes=3, name_prefix="ar1")
        assert traced["ar1_state"]["value"].shape == (3, 12)

    def test_n_timepoints_one_returns_initial_distribution_draw(self):
        """``n_timepoints=1`` returns a single stationary-prior draw per process."""
        ar1 = AR1(**fixed_ar1_kwargs(), parameterization="state")
        with numpyro.handlers.seed(rng_seed=0):
            path = ar1.sample(
                n_timepoints=1,
                n_processes=2,
                initial_value=jnp.array([0.0, 1.0]),
                name_prefix="ar1",
            )
        assert path.shape == (1, 2)

    @pytest.mark.parametrize("autoreg,innovation_sd", [(0.5, 0.05), (0.9, 0.1)])
    def test_prior_moments_match_innovation_parameterization(
        self, autoreg, innovation_sd
    ):
        """State and innovation AR1 produce the same per-timepoint moments."""
        n_timepoints = 30
        n_processes = 3
        init = jnp.array([0.0, 0.4, -0.2])

        kwargs = fixed_ar1_kwargs(autoreg=autoreg, innovation_sd=innovation_sd)
        ar1_state = AR1(**kwargs, parameterization="state")
        ar1_innov = AR1(**kwargs, parameterization="innovation")

        def model_state():
            """Record state-centered AR1 path as a deterministic for Predictive readout."""
            path = ar1_state.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=init,
                name_prefix="ar1",
            )
            numpyro.deterministic("path", path)

        def model_innov():
            """Record innovation-form AR1 path as a deterministic for Predictive readout."""
            path = ar1_innov.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=init,
                name_prefix="ar1",
            )
            numpyro.deterministic("path", path)

        n_samples = 10000
        s_state = Predictive(model_state, num_samples=n_samples)(jax.random.PRNGKey(0))[
            "path"
        ]
        s_innov = Predictive(model_innov, num_samples=n_samples)(jax.random.PRNGKey(1))[
            "path"
        ]

        stationary_sd = innovation_sd / jnp.sqrt(1 - autoreg**2)
        mean_atol = 5.0 * float(stationary_sd) / jnp.sqrt(n_samples)
        expected_mean = autoreg ** jnp.arange(n_timepoints)[:, None] * init[None, :]
        assert jnp.allclose(s_state.mean(axis=0), expected_mean, atol=mean_atol)
        assert jnp.allclose(s_innov.mean(axis=0), expected_mean, atol=mean_atol)

        assert jnp.allclose(
            s_state.var(axis=0), s_innov.var(axis=0), rtol=0.10, atol=1e-4
        )


class TestStateCenteredDifferencedAR1:
    """``DifferencedAR1`` in state mode draws the post-initial path from a single ``diff_state`` site backed by ``StateDifferencedAR1``."""

    def test_return_shape(self):
        """Return value has shape ``(n_timepoints, n_processes)``."""
        d = DifferencedAR1(**fixed_ar1_kwargs(), parameterization="state")
        with numpyro.handlers.seed(rng_seed=0):
            path = d.sample(n_timepoints=15, n_processes=4, name_prefix="diff")
        assert path.shape == (15, 4)

    def test_initial_row_equals_initial_value(self):
        """``x[0]`` is deterministic and equal to ``initial_value`` for every draw."""
        d = DifferencedAR1(**fixed_ar1_kwargs(), parameterization="state")
        init = jnp.array([0.5, -1.0, 2.0])
        with numpyro.handlers.seed(rng_seed=0):
            path = d.sample(
                n_timepoints=10,
                n_processes=3,
                initial_value=init,
                name_prefix="diff",
            )
        assert jnp.allclose(path[0], init)

    def test_n_timepoints_one_returns_initial_value(self):
        """``n_timepoints=1`` returns just the initial value as shape ``(1, n_processes)``."""
        d = DifferencedAR1(**fixed_ar1_kwargs(), parameterization="state")
        init = jnp.array([0.3, 0.7])
        with numpyro.handlers.seed(rng_seed=0):
            path = d.sample(
                n_timepoints=1,
                n_processes=2,
                initial_value=init,
                name_prefix="diff",
            )
        assert path.shape == (1, 2)
        assert jnp.allclose(path[0], init)

    def test_trace_has_state_site_not_innovation_sites(self):
        """State-mode trace contains a single ``_state`` site only."""
        d = DifferencedAR1(**fixed_ar1_kwargs(), parameterization="state")
        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(d.sample, rng_seed=0)
        ).get_trace(n_timepoints=8, n_processes=2, name_prefix="diff")
        assert "diff_state" in traced
        assert "diff_init_rate" not in traced
        assert "diff_noise" not in traced

    def test_state_site_shape(self):
        """The state site holds the post-initial path of shape ``(n_processes, n_timepoints - 1)``."""
        d = DifferencedAR1(**fixed_ar1_kwargs(), parameterization="state")
        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(d.sample, rng_seed=0)
        ).get_trace(n_timepoints=12, n_processes=3, name_prefix="diff")
        assert traced["diff_state"]["value"].shape == (3, 11)

    @pytest.mark.parametrize("autoreg,innovation_sd", [(0.5, 0.05), (0.9, 0.1)])
    def test_prior_moments_match_innovation_parameterization(
        self, autoreg, innovation_sd
    ):
        """State and innovation DifferencedAR1 produce the same per-timepoint moments."""
        n_timepoints = 30
        n_processes = 3
        init = jnp.array([0.0, 0.4, -0.2])

        kwargs = fixed_ar1_kwargs(autoreg=autoreg, innovation_sd=innovation_sd)
        d_state = DifferencedAR1(**kwargs, parameterization="state")
        d_innov = DifferencedAR1(**kwargs, parameterization="innovation")

        def model_state():
            """Record state-centered DifferencedAR1 path for Predictive readout."""
            path = d_state.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=init,
                name_prefix="diff",
            )
            numpyro.deterministic("path", path)

        def model_innov():
            """Record innovation-form DifferencedAR1 path for Predictive readout."""
            path = d_innov.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=init,
                name_prefix="diff",
            )
            numpyro.deterministic("path", path)

        n_samples = 10000
        s_state = Predictive(model_state, num_samples=n_samples)(jax.random.PRNGKey(0))[
            "path"
        ]
        s_innov = Predictive(model_innov, num_samples=n_samples)(jax.random.PRNGKey(1))[
            "path"
        ]

        terminal_var_state = float(s_state[:, -1, :].var())
        mean_atol = 5.0 * jnp.sqrt(terminal_var_state / n_samples)
        assert jnp.allclose(s_state.mean(axis=0), init[jnp.newaxis, :], atol=mean_atol)
        assert jnp.allclose(s_innov.mean(axis=0), init[jnp.newaxis, :], atol=mean_atol)

        assert jnp.allclose(
            s_state.var(axis=0), s_innov.var(axis=0), rtol=0.10, atol=1e-4
        )


class TestStepwiseTemporalProcessConstruction:
    """Construction-time validation for StepwiseTemporalProcess."""

    def test_step_size_attribute(self):
        """step_size is exposed on the instance for builder inspection."""
        wrapper = StepwiseTemporalProcess(
            AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)), step_size=7
        )
        assert wrapper.step_size == 7

    def test_zero_step_size_raises(self):
        """step_size=0 raises."""
        with pytest.raises(ValueError, match="positive integer"):
            StepwiseTemporalProcess(
                AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)),
                step_size=0,
            )

    def test_negative_step_size_raises(self):
        """Negative step_size raises."""
        with pytest.raises(ValueError, match="positive integer"):
            StepwiseTemporalProcess(
                AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)),
                step_size=-1,
            )

    def test_float_step_size_raises(self):
        """Non-integer step_size raises."""
        with pytest.raises(ValueError, match="positive integer"):
            StepwiseTemporalProcess(
                AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)),
                step_size=7.0,
            )

    def test_repr_includes_inner_and_step_size(self):
        """__repr__ shows the inner process and step_size."""
        inner = AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05))
        wrapper = StepwiseTemporalProcess(inner, step_size=7)
        rendered = repr(wrapper)
        assert rendered.startswith("StepwiseTemporalProcess(")
        assert f"inner={inner!r}" in rendered
        assert "step_size=7" in rendered


class TestStepwiseTemporalProcessSample:
    """Sample-time behavior of StepwiseTemporalProcess."""

    @pytest.mark.parametrize("inner_cls,inner_kwargs", INNER_PROCESS_PARAMS)
    def test_output_shape_divisible(self, inner_cls, inner_kwargs):
        """n_timepoints divisible by step_size yields (n_timepoints, n_processes)."""
        wrapper = StepwiseTemporalProcess(inner_cls(**inner_kwargs), step_size=7)
        with numpyro.handlers.seed(rng_seed=42):
            result = wrapper.sample(n_timepoints=28, n_processes=3)
        assert result.shape == (28, 3)

    @pytest.mark.parametrize("inner_cls,inner_kwargs", INNER_PROCESS_PARAMS)
    def test_output_shape_non_divisible(self, inner_cls, inner_kwargs):
        """n_timepoints not divisible by step_size still yields n_timepoints rows."""
        wrapper = StepwiseTemporalProcess(inner_cls(**inner_kwargs), step_size=7)
        with numpyro.handlers.seed(rng_seed=42):
            result = wrapper.sample(n_timepoints=30, n_processes=2)
        assert result.shape == (30, 2)

    @pytest.mark.parametrize("inner_cls,inner_kwargs", INNER_PROCESS_PARAMS)
    def test_broadcast_within_block(self, inner_cls, inner_kwargs):
        """Each block of step_size consecutive timepoints is constant."""
        wrapper = StepwiseTemporalProcess(inner_cls(**inner_kwargs), step_size=7)
        with numpyro.handlers.seed(rng_seed=42):
            result = wrapper.sample(n_timepoints=28, n_processes=1)
        for start in range(0, 28, 7):
            block = result[start : start + 7]
            assert jnp.allclose(block, block[0])

    @pytest.mark.parametrize("inner_cls,inner_kwargs", INNER_PROCESS_PARAMS)
    def test_broadcast_with_partial_final_block(self, inner_cls, inner_kwargs):
        """A partial final block is still constant, just shorter than step_size."""
        wrapper = StepwiseTemporalProcess(inner_cls(**inner_kwargs), step_size=7)
        with numpyro.handlers.seed(rng_seed=42):
            result = wrapper.sample(n_timepoints=30, n_processes=1)
        # The 4th block starts at row 28 and runs through row 29 (2 rows)
        final_block = result[28:30]
        assert jnp.allclose(final_block, final_block[0])

    def test_step_size_one_passthrough_shape(self):
        """step_size=1 yields (n_timepoints, n_processes), same as inner directly."""
        wrapper = StepwiseTemporalProcess(
            AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)), step_size=1
        )
        with numpyro.handlers.seed(rng_seed=42):
            result = wrapper.sample(n_timepoints=20, n_processes=2)
        assert result.shape == (20, 2)

    def test_coarse_trajectory_is_recorded(self):
        """StepwiseTemporalProcess records the coarse trajectory."""
        wrapper = StepwiseTemporalProcess(
            AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)), step_size=7
        )
        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(wrapper.sample, rng_seed=42)
        ).get_trace(n_timepoints=15, n_processes=1, name_prefix="rt")

        assert "rt_coarse" in traced
        assert traced["rt_coarse"]["type"] == "deterministic"
        assert traced["rt_coarse"]["value"].shape == (3, 1)


class TestWeeklyTemporalProcessConstruction:
    """Construction-time validation for WeeklyTemporalProcess."""

    def test_step_size_attribute(self):
        """step_size is exposed on the instance for builder inspection."""
        wrapper = WeeklyTemporalProcess(
            AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)),
            start_dow=MMWR_WEEK,
        )
        assert wrapper.step_size == 7

    def test_requires_calendar_anchor_attribute(self):
        """WeeklyTemporalProcess reports that it needs a calendar anchor."""
        wrapper = WeeklyTemporalProcess(
            AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)),
            start_dow=MMWR_WEEK,
        )
        assert wrapper.requires_calendar_anchor is True

    def test_requires_valid_start_dow(self):
        """WeeklyTemporalProcess requires a valid integer start_dow."""
        with pytest.raises(ValueError, match="Day-of-week"):
            WeeklyTemporalProcess(
                AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)),
                start_dow=None,
            )
        with pytest.raises(ValueError, match="Day-of-week"):
            WeeklyTemporalProcess(
                AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)),
                start_dow=7,
            )

    def test_repr_includes_inner_and_start_dow(self):
        """__repr__ shows the inner process and start_dow."""
        inner = AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05))
        wrapper = WeeklyTemporalProcess(inner, start_dow=MMWR_WEEK)
        rendered = repr(wrapper)
        assert rendered.startswith("WeeklyTemporalProcess(")
        assert f"inner={inner!r}" in rendered
        assert f"start_dow={MMWR_WEEK!r}" in rendered


class TestWeeklyTemporalProcessSample:
    """Sample-time behavior of WeeklyTemporalProcess."""

    def test_requires_first_day_dow_at_sample_time(self):
        """WeeklyTemporalProcess needs the model-axis day of week."""
        wrapper = WeeklyTemporalProcess(
            AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)),
            start_dow=MMWR_WEEK,
        )
        with numpyro.handlers.seed(rng_seed=42):
            with pytest.raises(ValueError, match="first_day_dow"):
                wrapper.sample(n_timepoints=20, n_processes=1)

    def test_alignment_with_leading_partial_week(self):
        """WeeklyTemporalProcess starts full blocks on start_dow."""
        wrapper = WeeklyTemporalProcess(
            AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)),
            start_dow=MMWR_WEEK,
        )
        # first_day_dow=3 means day 0 is Thursday. With Sunday week starts,
        # days 0-2 are a leading partial week, then days 3-9 are the first
        # full Sunday-Saturday block on the model axis.
        with numpyro.handlers.seed(rng_seed=42):
            result = wrapper.sample(
                n_timepoints=17,
                n_processes=1,
                first_day_dow=3,
            )

        assert jnp.allclose(result[:3], result[0])
        assert jnp.allclose(result[3:10], result[3])
        assert jnp.allclose(result[10:17], result[10])
        assert not jnp.allclose(result[2], result[3])

    def test_alignment_without_leading_partial_week(self):
        """WeeklyTemporalProcess handles model axes starting on start_dow."""
        wrapper = WeeklyTemporalProcess(
            AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)),
            start_dow=MMWR_WEEK,
        )
        with numpyro.handlers.seed(rng_seed=42):
            result = wrapper.sample(
                n_timepoints=15,
                n_processes=1,
                first_day_dow=6,
            )

        assert jnp.allclose(result[:7], result[0])
        assert jnp.allclose(result[7:14], result[7])
        assert jnp.allclose(result[14:15], result[14])

    def test_weekly_trajectory_is_recorded(self):
        """WeeklyTemporalProcess records the weekly trajectory."""
        wrapper = WeeklyTemporalProcess(
            AR1(**fixed_ar1_kwargs(autoreg=0.9, innovation_sd=0.05)),
            start_dow=MMWR_WEEK,
        )
        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(wrapper.sample, rng_seed=42)
        ).get_trace(
            n_timepoints=15,
            n_processes=1,
            name_prefix="rt",
            first_day_dow=6,
        )

        assert "rt_weekly" in traced
        assert traced["rt_weekly"]["type"] == "deterministic"
        assert traced["rt_weekly"]["value"].shape == (3, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
