"""Fit and forecast a synthetic mobility outbreak with cases and deaths.

The model follows vhf-metapop-renew's construction: a fixed mobility matrix,
weekly baseline and deviation processes, ratio-linked ascertainment, separate
reporting delays, and negative-binomial counts by home region. All populations,
mobility values, Rt trajectories, and counts here are synthetic; this is not a
fit to the 2026 DRC outbreak. Counts are generated with a separate NumPy renewal
calculation so the test can check recovery against known infections and Rt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.diagnostics import summary

from pyrenew.ascertainment import RatioLinkedAscertainment
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import AR1, MobilitySubpopulationInfections, WeeklyTemporalProcess
from pyrenew.math import r_approx_from_R
from pyrenew.model import MultiSignalModel, PyrenewBuilder
from pyrenew.observation import NegativeBinomialNoise, SubpopulationCounts
from pyrenew.randomvariable import DistributionalVariable
from pyrenew.time import MMWR_WEEK

pytestmark = pytest.mark.integration

Parameterization = Literal["state", "innovation"]
POPULATIONS = np.array([300_000.0, 500_000.0, 200_000.0])
FRACTIONS = POPULATIONS / POPULATIONS.sum()
MOBILITY = np.array([[0.90, 0.06, 0.02], [0.08, 0.90, 0.08], [0.02, 0.04, 0.90]])
I0 = np.array([2e-5, 7e-6, 1e-6])
START_DATE = date(2026, 5, 17)
N_DAYS = 56
N_FORECAST = 14
NUM_WARMUP = 500
NUM_SAMPLES = 500
NUM_CHAINS = 4
ASCERTAINMENT = 0.25
DEATH_RATIO = 0.44
CONCENTRATION = 30.0
SIGNALS = ("case_confirmations", "confirmed_deaths")


def _delay_pmf(offset: float, scale: float, shape: float) -> np.ndarray:
    """Discretize an offset Weibull delay on days 0 through 49.

    Returns
    -------
    np.ndarray
        Normalized daily delay probabilities.
    """
    elapsed = np.maximum(np.arange(50) - offset, 0.0)
    cdf = -np.expm1(-((elapsed / scale) ** shape))
    pmf = np.diff(cdf, prepend=0.0)
    return pmf / pmf.sum()


GEN_INT = _delay_pmf(offset=3.0, scale=10.14, shape=2.0)
DELAYS = {
    "case_confirmations": _delay_pmf(offset=5.0, scale=5.0, shape=1.0),
    "confirmed_deaths": _delay_pmf(offset=12.0, scale=1.0, shape=10.0),
}
N_INIT = len(GEN_INT)


@dataclass
class SyntheticOutbreak:
    """Known daily Rt and infections, with noisy weekly observations."""

    rt: np.ndarray
    infections: np.ndarray
    means: dict[str, np.ndarray]
    observations: dict[str, np.ndarray]


def _renewal_infections(rt: np.ndarray, mobility: np.ndarray) -> np.ndarray:
    """Calculate infections per resident with mobility and susceptible depletion.

    Returns
    -------
    np.ndarray
        Daily infections by home region, including initialization.
    """
    infections = np.zeros((N_INIT + len(rt), len(POPULATIONS)))
    growth = np.asarray(
        r_approx_from_R(jnp.asarray(rt[0]), jnp.asarray(GEN_INT), n_newton_steps=4)
    )
    infections[:N_INIT] = I0 * np.exp(np.arange(N_INIT)[:, None] * growth)
    present = mobility @ FRACTIONS
    resident_share = mobility * FRACTIONS[None, :] / present[:, None]
    cumulative = infections[:N_INIT].sum(axis=0)
    for t, reproduction in enumerate(rt, start=N_INIT):
        infectiousness = GEN_INT[::-1] @ infections[t - len(GEN_INT) : t]
        susceptible = 1.0 - cumulative
        exposure = mobility.T @ (reproduction * (resident_share @ infectiousness))
        infections[t] = np.minimum(susceptible * exposure, susceptible)
        cumulative += infections[t]
    return infections


def _weekly_counts(infections: np.ndarray, signal: str) -> np.ndarray:
    """Delay and ascertain infection counts, then sum complete reporting weeks.

    Returns
    -------
    np.ndarray
        Expected weekly counts by region, with initialization weeks masked.
    """
    counts = infections * POPULATIONS
    delayed = np.column_stack(
        [
            np.convolve(counts[:, k], DELAYS[signal])[: len(counts)]
            for k in range(len(POPULATIONS))
        ]
    )
    rate = ASCERTAINMENT * (DEATH_RATIO if signal == "confirmed_deaths" else 1.0)
    history_dow = (START_DATE.weekday() - N_INIT) % 7
    offset = (MMWR_WEEK - history_dow) % 7
    n_weeks = (len(counts) - offset) // 7
    weekly = (
        (rate * delayed[offset : offset + 7 * n_weeks])
        .reshape(n_weeks, 7, len(POPULATIONS))
        .sum(axis=1)
    )
    week_starts = offset + 7 * np.arange(n_weeks)
    weekly[week_starts < N_INIT] = np.nan
    return weekly


@pytest.fixture(scope="module")
def outbreak() -> SyntheticOutbreak:
    """Generate eight weeks of regional case and death counts with known truth.

    Returns
    -------
    SyntheticOutbreak
        True trajectories, expected counts, and noisy observations.
    """
    baseline = np.log(np.array([1.25, 1.28, 1.22, 1.15, 1.10, 1.0, 0.95, 0.92]))
    departures = np.array(
        [
            [0.12, -0.02, -0.10],
            [0.10, 0.00, -0.10],
            [0.06, 0.04, -0.10],
            [0.02, 0.08, -0.10],
            [-0.02, 0.10, -0.08],
            [-0.06, 0.10, -0.04],
            [-0.08, 0.08, 0.00],
            [-0.06, 0.02, 0.04],
        ]
    )
    rt = np.repeat(np.exp(baseline[:, None] + departures), 7, axis=0)
    infections = _renewal_infections(rt, MOBILITY)
    means = {signal: _weekly_counts(infections, signal) for signal in SIGNALS}
    rng = np.random.default_rng(935)
    observations = {}
    for signal, expected in means.items():
        valid = np.isfinite(expected)
        observed = np.full_like(expected, np.nan)
        observed[valid] = rng.negative_binomial(
            CONCENTRATION, CONCENTRATION / (CONCENTRATION + expected[valid])
        )
        observations[signal] = observed
    # Exercise missing observations independently of initialization masking.
    observations["confirmed_deaths"][-3, 1] = np.nan
    return SyntheticOutbreak(rt, infections, means, observations)


def _build_model(parameterization: Parameterization) -> MultiSignalModel:
    """Build the weekly mobility model used to infer the synthetic outbreak.

    Returns
    -------
    MultiSignalModel
        Mobility infections with linked case and death observations.
    """
    builder = PyrenewBuilder()
    builder.configure_latent(
        MobilitySubpopulationInfections,
        name="latent",
        gen_int_rv=DeterministicPMF("gen_int", jnp.asarray(GEN_INT)),
        I0_rv=DeterministicVariable("I0", jnp.asarray(I0)),
        log_rt_time_0_rv=DistributionalVariable(
            "log_rt_time_0", dist.Normal(np.log(1.2), 0.2).expand((len(POPULATIONS),))
        ),
        baseline_rt_process=WeeklyTemporalProcess(
            AR1(
                autoreg_rv=DeterministicVariable("baseline_autoreg", 0.9),
                innovation_sd_rv=DistributionalVariable(
                    "baseline_innovation_sd", dist.LogNormal(np.log(0.08), 0.35)
                ),
                parameterization=parameterization,
            ),
            start_dow=MMWR_WEEK,
        ),
        subpop_rt_deviation_process=WeeklyTemporalProcess(
            AR1(
                autoreg_rv=DistributionalVariable("deviation_autoreg", dist.Beta(8, 2)),
                innovation_sd_rv=DistributionalVariable(
                    "deviation_innovation_sd", dist.LogNormal(np.log(0.07), 0.35)
                ),
                parameterization=parameterization,
            ),
            start_dow=MMWR_WEEK,
        ),
        mobility_matrix_rv=DeterministicVariable("mobility", jnp.asarray(MOBILITY)),
        susceptible_depletion=True,
    )
    ascertainment = RatioLinkedAscertainment(
        name="ascertainment",
        base_signal="case_confirmations",
        linked_signal="confirmed_deaths",
        base_rate_rv=DeterministicVariable("case_ascertainment", ASCERTAINMENT),
        ratio_rv=DeterministicVariable("death_ratio", DEATH_RATIO),
    )
    builder.add_ascertainment(ascertainment)
    for signal in SIGNALS:
        builder.add_observation(
            SubpopulationCounts(
                name=signal,
                ascertainment_rate_rv=ascertainment.for_signal(signal),
                delay_distribution_rv=DeterministicPMF(
                    signal + "_delay", DELAYS[signal]
                ),
                noise=NegativeBinomialNoise(
                    DeterministicVariable(signal + "_concentration", CONCENTRATION)
                ),
                aggregation="weekly",
                start_dow=MMWR_WEEK,
            )
        )
    return builder.build()


@pytest.fixture(scope="module", params=["state", "innovation"])
def fitted_model(
    request: pytest.FixtureRequest, outbreak: SyntheticOutbreak
) -> MultiSignalModel:
    """Fit each parameterization once and share its samples across assertions.

    Returns
    -------
    MultiSignalModel
        Fitted model with four chains of posterior samples.
    """
    model = _build_model(request.param)
    assert model.latent.n_initialization_points == N_INIT
    model.run(
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        rng_key=jax.random.key(42),
        nuts_args={"target_accept_prob": 0.9},
        mcmc_args={"num_chains": NUM_CHAINS, "progress_bar": False},
        n_days_post_init=N_DAYS,
        population_size=float(POPULATIONS.sum()),
        subpop_fractions=jnp.asarray(FRACTIONS),
        obs_start_date=START_DATE,
        **{
            signal: {
                "obs": jnp.asarray(outbreak.observations[signal]),
                "subpop_indices": jnp.arange(len(POPULATIONS)),
            }
            for signal in SIGNALS
        },
    )
    jax.block_until_ready(model.mcmc.get_samples())
    return model


def test_fixture_requires_mobility(outbreak: SyntheticOutbreak) -> None:
    """The asymmetric matrix changes infections appreciably from an identity matrix."""
    np.testing.assert_allclose(MOBILITY.sum(axis=0), 1.0)
    assert not np.allclose(MOBILITY.sum(axis=1), 1.0)
    isolated = _renewal_infections(outbreak.rt, np.eye(len(POPULATIONS)))
    relative_change = np.linalg.norm(outbreak.infections - isolated) / np.linalg.norm(
        isolated
    )
    assert relative_change > 0.1


def test_mobility_posterior_geometry(fitted_model: MultiSignalModel) -> None:
    """Both representations retain zero-sum deviations and weekly Rt states."""
    samples = fitted_model.mcmc.get_samples()
    rt = np.asarray(samples["latent::rt_subpop"])
    deviations = np.asarray(samples["latent::subpop_deviations"])
    coordinate_sites = {
        name: values
        for name, values in samples.items()
        if name.startswith("subpop_deviations_coordinates_")
    }
    assert coordinate_sites
    for values in coordinate_sites.values():
        assert values.shape[-1] == len(POPULATIONS) - 1
    assert rt.shape == (NUM_CHAINS * NUM_SAMPLES, N_DAYS, len(POPULATIONS))
    np.testing.assert_allclose(deviations.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.log(rt), samples["latent::log_rt_baseline"] + deviations, atol=1e-12
    )
    np.testing.assert_allclose(rt, np.repeat(rt[:, ::7, :], 7, axis=1), atol=1e-12)
    np.testing.assert_allclose(
        samples["latent_infections_by_subpop"].sum(axis=-1),
        samples["latent_infections"],
        rtol=1e-10,
    )


def test_sampled_infections_and_counts_match_numpy(
    fitted_model: MultiSignalModel,
) -> None:
    """Posterior infections and delayed counts agree with independent calculations."""
    samples = fitted_model.mcmc.get_samples()
    for draw in (0, NUM_SAMPLES, 2 * NUM_SAMPLES):
        expected = _renewal_infections(
            np.asarray(samples["latent::rt_subpop"][draw]), MOBILITY
        )
        actual = np.asarray(samples["latent_infections_by_subpop"][draw]) / POPULATIONS
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
        for signal in SIGNALS:
            np.testing.assert_allclose(
                samples[signal + "_predicted"][draw],
                _weekly_counts(expected, signal),
                rtol=1e-8,
                atol=1e-10,
                equal_nan=True,
            )


def test_mobility_mcmc_diagnostics(fitted_model: MultiSignalModel) -> None:
    """Initial Rt and temporal hyperparameters mix without divergent transitions."""
    samples = fitted_model.mcmc.get_samples(group_by_chain=True)
    names = (
        "log_rt_time_0",
        "baseline_innovation_sd",
        "deviation_autoreg",
        "deviation_innovation_sd",
    )
    diagnostics = summary({name: samples[name] for name in names})
    for name, diagnostic in diagnostics.items():
        assert np.all(diagnostic["r_hat"] < 1.05), (name, diagnostic)
        assert np.all(diagnostic["n_eff"] > 100), (name, diagnostic)
    assert not np.any(fitted_model.mcmc.get_extra_fields()["diverging"])


def test_mobility_recovers_rt_and_signals(
    fitted_model: MultiSignalModel, outbreak: SyntheticOutbreak
) -> None:
    """Posterior intervals cover known Rt and expected counts in each region."""
    samples = fitted_model.mcmc.get_samples()
    for name, truth in (
        ("latent::rt_subpop", outbreak.rt),
        ("latent_infections_by_subpop", outbreak.infections * POPULATIONS),
        *((signal + "_predicted", outbreak.means[signal]) for signal in SIGNALS),
    ):
        low, high = np.quantile(np.asarray(samples[name]), [0.05, 0.95], axis=0)
        for k in range(len(POPULATIONS)):
            valid = np.isfinite(truth[:, k])
            covered = (low[valid, k] <= truth[valid, k]) & (
                truth[valid, k] <= high[valid, k]
            )
            assert covered.mean() >= 0.75, (name, k, covered.mean())


def test_mobility_forecast_replays_fitted_trajectories(
    fitted_model: MultiSignalModel,
) -> None:
    """Forecasts preserve fitted states and produce both weekly observation signals."""
    forecast = fitted_model.posterior_predictive(
        rng_key=jax.random.key(43),
        n_days_post_init=N_DAYS + N_FORECAST,
        population_size=float(POPULATIONS.sum()),
        subpop_fractions=jnp.asarray(FRACTIONS),
        obs_start_date=START_DATE,
        **{
            signal: {"subpop_indices": jnp.arange(len(POPULATIONS))}
            for signal in SIGNALS
        },
    )
    samples = fitted_model.mcmc.get_samples()
    for name, fitted_length in (
        ("latent::rt_subpop", N_DAYS),
        ("latent_infections_by_subpop", N_INIT + N_DAYS),
    ):
        np.testing.assert_allclose(
            forecast[name][:, :fitted_length], samples[name], rtol=1e-10
        )
        assert forecast[name].shape[1] == fitted_length + N_FORECAST
        assert np.all(np.isfinite(forecast[name]))
    np.testing.assert_allclose(
        forecast["latent::subpop_deviations"].sum(axis=-1), 0.0, atol=1e-12
    )
    for signal in SIGNALS:
        predicted = forecast[signal + "_predicted"]
        assert (
            predicted.shape[1]
            == samples[signal + "_predicted"].shape[1] + N_FORECAST // 7
        )
        assert np.all(np.isfinite(predicted[:, -N_FORECAST // 7 :]))
        assert np.all(np.asarray(forecast[signal + "_obs"][:, -N_FORECAST // 7 :]) >= 0)
