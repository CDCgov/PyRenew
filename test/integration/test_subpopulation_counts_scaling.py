"""
Integration test: subpopulation count scaling in MultiSignalModel (issue 903).

Four subpopulations share the same true Rt and the same initial infections
per resident, so they differ only in population size. Counts are simulated
as infections per resident times the subpopulation population. A
SubpopulationInfections model fitted to these counts should recover Rt in
every subpopulation, with no dependence on subpopulation size, and its
recorded subpopulation counts should sum to the aggregate count.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.distributions as dist
import pytest

from pyrenew.ascertainment import IndependentAscertainment
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import RandomWalk, SubpopulationInfections
from pyrenew.math import r_approx_from_R
from pyrenew.model import MultiSignalModel, PyrenewBuilder
from pyrenew.observation import PoissonNoise, SubpopulationCounts
from pyrenew.randomvariable import DistributionalVariable

pytestmark = pytest.mark.integration

POPULATION_SIZE = 1_000_000.0
SUBPOP_FRACTIONS = np.array([0.5, 0.25, 0.15, 0.10])
N_SUBPOPS = len(SUBPOP_FRACTIONS)
GEN_INT = np.array([0.2, 0.35, 0.25, 0.15, 0.05])
DELAY = np.array([0.1, 0.3, 0.3, 0.2, 0.1])
I0_PER_RESIDENT = 2e-5
ASCERTAINMENT = 0.3
N_DAYS_FIT = 70
N_DAYS_EARLY = 21
DATA_SEED = 20261011
NUM_WARMUP = 500
NUM_SAMPLES = 500
NUM_CHAINS = 4


def _true_rt(n_total: int) -> np.ndarray:
    """
    Return the true Rt shared by all subpopulations on the padded time axis.

    Parameters
    ----------
    n_total
        Number of days, including the initialization period.

    Returns
    -------
    np.ndarray
        Rt for each day, falling from about 1.4 to about 0.9.
    """
    t = np.arange(n_total)
    return 0.9 + 0.5 / (1.0 + np.exp((t - 35.0) / 4.0))


def _simulate(n_init: int) -> dict[str, np.ndarray]:
    """
    Simulate infections per resident and observed counts for each subpopulation.

    Initialization-period infections grow exponentially from
    ``I0_PER_RESIDENT`` at the rate implied by the first day's Rt, as in
    ``SubpopulationInfections``. Counts in subpopulation ``k`` are
    infections per resident times ``POPULATION_SIZE * SUBPOP_FRACTIONS[k]``,
    delayed by ``DELAY``, scaled by ``ASCERTAINMENT``, and drawn from a
    Poisson distribution.

    Parameters
    ----------
    n_init
        Number of initialization days.

    Returns
    -------
    dict[str, np.ndarray]
        ``"rt"``: true Rt on the padded axis, shape ``(n_total,)``.
        ``"observed"``: observed counts after initialization,
        shape ``(N_DAYS_FIT, N_SUBPOPS)``.
    """
    n_total = n_init + N_DAYS_FIT
    rt = _true_rt(n_total)
    initial_growth_rate = float(
        r_approx_from_R(R=jnp.asarray(rt[0]), g=jnp.asarray(GEN_INT), n_newton_steps=4)
    )
    infections = np.zeros(n_total)
    infections[:n_init] = I0_PER_RESIDENT * np.exp(
        initial_growth_rate * np.arange(n_init)
    )
    reversed_gen_int = GEN_INT[::-1]
    for t in range(n_init, n_total):
        infections[t] = rt[t] * reversed_gen_int @ infections[t - GEN_INT.size : t]
    counts = POPULATION_SIZE * np.outer(infections, SUBPOP_FRACTIONS)
    predicted = np.zeros_like(counts)
    for d, weight in enumerate(DELAY):
        predicted[d:] += ASCERTAINMENT * weight * counts[: n_total - d]
    rng = np.random.default_rng(DATA_SEED)
    observed = rng.poisson(predicted[n_init:])
    return {"rt": rt, "observed": observed}


@pytest.fixture(scope="module")
def scaling_model() -> MultiSignalModel:
    """
    Build a SubpopulationInfections model with subpopulation case counts.

    Initial infections per resident are fixed at the true value, and one
    ascertainment rate is shared across subpopulations.

    Returns
    -------
    MultiSignalModel
        The built model.
    """
    ascertainment = IndependentAscertainment(
        name="case_ascertainment",
        rate_rvs={
            "cases": DistributionalVariable("ascertainment_rate", dist.Beta(3.0, 7.0))
        },
    )
    cases = SubpopulationCounts(
        name="cases",
        ascertainment_rate_rv=ascertainment.for_signal("cases"),
        delay_distribution_rv=DeterministicPMF("delay", jnp.asarray(DELAY)),
        noise=PoissonNoise(),
    )
    builder = PyrenewBuilder()
    builder.configure_latent(
        SubpopulationInfections,
        name="latent",
        gen_int_rv=DeterministicPMF("gen_int", jnp.asarray(GEN_INT)),
        I0_rv=DeterministicVariable("I0", jnp.full(N_SUBPOPS, I0_PER_RESIDENT)),
        log_rt_time_0_rv=DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5)),
        baseline_rt_process=RandomWalk(DeterministicVariable("baseline_sd", 0.05)),
        subpop_rt_deviation_process=RandomWalk(
            DeterministicVariable("deviation_sd", 0.03)
        ),
    )
    builder.add_observation(cases)
    builder.add_ascertainment(ascertainment)
    return builder.build()


@pytest.fixture(scope="module")
def simulated(scaling_model: MultiSignalModel) -> dict[str, np.ndarray]:
    """
    Simulate data on the model's padded time axis.

    Parameters
    ----------
    scaling_model
        Built model, used for its number of initialization days.

    Returns
    -------
    dict[str, np.ndarray]
        True Rt and observed counts.
    """
    return _simulate(scaling_model.latent.n_initialization_points)


@pytest.fixture(scope="module")
def posterior(
    scaling_model: MultiSignalModel,
    simulated: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Fit the model to the simulated counts and return posterior samples.

    Parameters
    ----------
    scaling_model
        Built model.
    simulated
        Simulated data.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples, keyed by site name.
    """
    scaling_model.run(
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        rng_key=random.PRNGKey(1),
        mcmc_args={"num_chains": NUM_CHAINS, "progress_bar": False},
        n_days_post_init=N_DAYS_FIT,
        population_size=POPULATION_SIZE,
        subpop_fractions=jnp.asarray(SUBPOP_FRACTIONS),
        cases={
            "obs": scaling_model.pad_observations(
                jnp.asarray(simulated["observed"], dtype=float)
            ),
            "subpop_indices": jnp.arange(N_SUBPOPS),
        },
    )
    samples = scaling_model.mcmc.get_samples()
    jax.block_until_ready(samples)
    return {name: np.asarray(value) for name, value in samples.items()}


class TestSubpopulationCountScaling:
    """Fitted subpopulation Rt and infections do not depend on subpopulation size."""

    def test_fit_uses_float64(
        self,
        posterior: dict[str, np.ndarray],
    ) -> None:
        """
        The model is fitted in 64-bit floats.

        Parameters
        ----------
        posterior
            Posterior samples.
        """
        assert posterior["latent::rt_subpop"].dtype == np.float64
        assert posterior["latent_infections_by_subpop"].dtype == np.float64

    def test_subpop_counts_sum_to_aggregate(
        self,
        posterior: dict[str, np.ndarray],
    ) -> None:
        """
        Recorded subpopulation infection counts sum to the aggregate count.

        Parameters
        ----------
        posterior
            Posterior samples.
        """
        np.testing.assert_allclose(
            posterior["latent_infections_by_subpop"].sum(axis=-1),
            posterior["latent_infections"],
            rtol=1e-10,
        )

    def test_rt_recovered_in_every_subpop(
        self,
        scaling_model: MultiSignalModel,
        simulated: dict[str, np.ndarray],
        posterior: dict[str, np.ndarray],
    ) -> None:
        """
        Posterior median Rt is close to the true Rt in every subpopulation.

        Checks the mean of log(fitted Rt / true Rt) over the first three
        weeks after initialization, where errors in the infection level
        show up most strongly.

        Parameters
        ----------
        scaling_model
            Fitted model.
        simulated
            Simulated data.
        posterior
            Posterior samples.
        """
        n_init = scaling_model.latent.n_initialization_points
        rt_fit = np.median(posterior["latent::rt_subpop"], axis=0)[n_init:]
        log_ratio = np.log(rt_fit / simulated["rt"][n_init:, np.newaxis])
        early_bias = log_ratio[:N_DAYS_EARLY].mean(axis=0)
        assert np.all(np.abs(early_bias) < 0.06), early_bias

    def test_no_rt_trend_with_subpop_size(
        self,
        scaling_model: MultiSignalModel,
        posterior: dict[str, np.ndarray],
    ) -> None:
        """
        Subpopulation Rt deviations show no trend with subpopulation size.

        All subpopulations share the same true Rt, so the slope of each
        subpopulation's mean deviation against the log of its population
        fraction should be close to zero.

        Parameters
        ----------
        scaling_model
            Fitted model.
        posterior
            Posterior samples.
        """
        n_init = scaling_model.latent.n_initialization_points
        deviations = np.median(posterior["latent::subpop_deviations"], axis=0)
        mean_deviation = deviations[n_init:].mean(axis=0)
        slope = np.polyfit(np.log(SUBPOP_FRACTIONS), mean_deviation, 1)[0]
        assert abs(slope) < 0.01, (slope, mean_deviation)
