"""Prior, timing, and replay contracts for subpopulation deviations."""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
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
    TemporalProcess,
    WeeklyTemporalProcess,
)
from pyrenew.latent.temporal_processes import Parameterization, _zero_sum_deviations
from pyrenew.metaclass import RandomVariable
from pyrenew.randomvariable import DistributionalVariable

ProcessType = type[AR1] | type[DifferencedAR1] | type[RandomWalk]
PROCESSES = (AR1, DifferencedAR1, RandomWalk)
PARAMETERIZATIONS = ("state", "innovation")


def make_process(
    process_type: ProcessType,
    parameterization: Parameterization,
    sd: RandomVariable | None = None,
    autoreg: RandomVariable | None = None,
) -> TemporalProcess:
    """Build an ordinary temporal process before applying the adapter.

    Returns
    -------
    TemporalProcess
        The unadapted temporal process.
    """
    sd = sd if sd is not None else DeterministicVariable("sd", 0.2)
    if process_type is RandomWalk:
        return RandomWalk(sd, parameterization=parameterization)
    autoreg = autoreg if autoreg is not None else DeterministicVariable("autoreg", 0.6)
    return process_type(autoreg, sd, parameterization=parameterization)


@pytest.mark.parametrize("process_type", PROCESSES)
@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
@pytest.mark.parametrize("n_timepoints", [1, 2, 6])
@pytest.mark.parametrize("k", [1, 2, 6])
def test_coordinates_and_initial_values(
    process_type: ProcessType,
    parameterization: Parameterization,
    n_timepoints: int,
    k: int,
) -> None:
    """Sample K - 1 coordinates, preserve starts, and skip all draws for K = 1."""
    process = _zero_sum_deviations(
        make_process(
            process_type,
            parameterization,
            sd=DistributionalVariable("sd", dist.LogNormal(-2, 0.1)),
            autoreg=DistributionalVariable("autoreg", dist.Beta(4, 2)),
        )
    )
    initial = jnp.arange(k, dtype=float)
    initial -= initial.mean()
    with numpyro.handlers.seed(rng_seed=1), numpyro.handlers.trace() as trace:
        values = process.sample(n_timepoints, initial, k)
    assert values.shape == (n_timepoints, k)
    np.testing.assert_allclose(values[0], initial, atol=1e-12)
    np.testing.assert_allclose(values.sum(axis=-1), 0, atol=1e-12)
    sampled = {name: site for name, site in trace.items() if site["type"] == "sample"}
    if k == 1:
        assert not sampled
        np.testing.assert_array_equal(values, 0)
    else:
        assert "sd" in sampled
        assert ("autoreg" in sampled) == (process_type is not RandomWalk)
        for name, site in sampled.items():
            if name.startswith("subpop_deviations"):
                assert name.startswith("subpop_deviations_coordinates_")
                assert site["value"].shape[-1] == k - 1


@pytest.mark.parametrize("process_type", PROCESSES)
@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_prior_mean_and_full_covariance(
    process_type: ProcessType, parameterization: Parameterization
) -> None:
    """Old and new priors match analytic means and space-time covariance."""
    n_timepoints, k, n_draws = 5, 3, 12000
    initial = jnp.array([-0.3, 0.1, 0.2])
    original = make_process(process_type, parameterization)
    adapted = _zero_sum_deviations(original)

    def model() -> None:
        """Record independent draws from the old and new constructions."""
        raw = original.sample(n_timepoints, initial, k, name_prefix="raw")
        numpyro.deterministic("old", raw - raw.mean(axis=-1, keepdims=True))
        numpyro.deterministic("new", adapted.sample(n_timepoints, initial, k))

    paths = Predictive(model, num_samples=n_draws)(jax.random.key(34))
    times = np.arange(n_timepoints)
    lag = np.abs(times[:, None] - times[None, :])
    mean = np.broadcast_to(initial, (n_timepoints, k))
    if process_type is AR1:
        mean = 0.6 ** times[:, None] * initial
        temporal_cov = (
            0.2**2
            / (1 - 0.6**2)
            * (0.6**lag - 0.6 ** (times[:, None] + times[None, :]))
        )
    elif process_type is DifferencedAR1:
        rate_cov = 0.2**2 / (1 - 0.6**2) * 0.6 ** lag[:-1, :-1]
        temporal_cov = np.pad(rate_cov.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    else:
        temporal_cov = 0.2**2 * np.minimum(times[:, None], times[None, :])
    expected_cov = np.kron(temporal_cov, np.eye(k) - np.ones((k, k)) / k)
    variance = np.diag(expected_cov)
    mean_tolerance = 6 * np.sqrt(variance / n_draws) + 1e-12
    covariance_tolerance = (
        6
        * np.sqrt(
            (variance[:, None] * variance[None, :] + expected_cov**2) / (n_draws - 1)
        )
        + 1e-12
    )
    for name in ("old", "new"):
        flat = np.asarray(paths[name]).reshape(n_draws, -1)
        assert np.all(np.abs(flat.mean(0) - mean.ravel()) < mean_tolerance), name
        assert np.all(
            np.abs(np.cov(flat, rowvar=False) - expected_cov) < covariance_tolerance
        ), name


@pytest.mark.parametrize("process_type", PROCESSES)
@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
@pytest.mark.parametrize("cadence", ["daily", "weekly", "stepwise"])
def test_timing_and_forecast_replay(
    process_type: ProcessType,
    parameterization: Parameterization,
    cadence: Literal["daily", "weekly", "stepwise"],
) -> None:
    """Replay coordinate sites and extend zero-sum daily and coarse records."""
    original = make_process(process_type, parameterization)
    if cadence == "weekly":
        original = WeeklyTemporalProcess(original, start_dow=6)
    elif cadence == "stepwise":
        original = StepwiseTemporalProcess(original, step_size=3)
    process = _zero_sum_deviations(original)

    def model(n_timepoints: int) -> None:
        """Record deviations with a leading partial calendar week."""
        numpyro.deterministic(
            "deviations",
            process.sample(
                n_timepoints,
                jnp.array([-0.3, 0.1, 0.2]),
                3,
                name_prefix="subpop_deviations",
                first_day_dow=2,
            ),
        )

    trace = numpyro.handlers.trace(numpyro.handlers.seed(model, rng_seed=4)).get_trace(
        9
    )
    posterior = {
        name: site["value"][None, ...]
        for name, site in trace.items()
        if site["type"] == "sample"
    }
    predictive = Predictive(model, posterior_samples=posterior)
    for length in (9, 16):
        result = predictive(jax.random.key(5), length)
        values = result["deviations"][0]
        assert values.shape == (length, 3)
        np.testing.assert_allclose(values.sum(-1), 0, atol=1e-12)
        overlap = min(length, 9)
        np.testing.assert_allclose(
            values[:overlap], trace["deviations"]["value"][:overlap]
        )
        if length > 9:
            assert not np.allclose(values[-1], values[8])
        if cadence != "daily":
            suffix = "weekly" if cadence == "weekly" else "coarse"
            coarse = result[f"subpop_deviations_{suffix}"][0]
            indices = (
                (np.arange(length) + 3) // 7
                if cadence == "weekly"
                else np.arange(length) // 3
            )
            np.testing.assert_allclose(values, coarse[indices])
            np.testing.assert_allclose(coarse.sum(-1), 0, atol=1e-12)
            assert coarse.shape[1] == 3


@pytest.mark.parametrize("process_type", PROCESSES)
@pytest.mark.parametrize("parameter", ["sd", "autoreg"])
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("shape", [(1,), (2,), (3,)])
def test_rejects_vector_parameters(
    process_type: ProcessType,
    parameter: str,
    stochastic: bool,
    shape: tuple[int, ...],
) -> None:
    """Reject vectors even when they broadcast to the coordinate dimension."""
    if parameter == "autoreg" and process_type is RandomWalk:
        return
    rv = (
        DistributionalVariable(parameter, dist.Uniform(0.1, 0.5).expand(shape))
        if stochastic
        else DeterministicVariable(parameter, jnp.full(shape, 0.2))
    )
    process = make_process(
        process_type,
        "innovation",
        sd=rv if parameter == "sd" else None,
        autoreg=rv if parameter == "autoreg" else None,
    )
    with (
        numpyro.handlers.seed(rng_seed=1),
        pytest.raises(ValueError, match="shared scalar parameters"),
    ):
        _zero_sum_deviations(process).sample(4, jnp.zeros(3), 3)


def test_adapter_does_not_change_original_process() -> None:
    """Adapting deviations leaves a reused standalone process untouched."""
    original = AR1(
        DeterministicVariable("autoreg", 0.6), DeterministicVariable("sd", 0.2)
    )
    sd = original.innovation_sd_rv
    autoreg = original.autoreg_rv
    _zero_sum_deviations(original)
    assert original.innovation_sd_rv is sd
    assert original.autoreg_rv is autoreg
