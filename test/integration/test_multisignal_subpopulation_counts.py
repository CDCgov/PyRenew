"""Integration coverage for subpopulation count observations."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as random
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import SubpopulationInfections
from pyrenew.model import MultiSignalModel, PyrenewBuilder
from pyrenew.observation import PoissonNoise, SubpopulationCounts
from test.test_helpers import fixed_random_walk

pytestmark = pytest.mark.integration


def _build_subpopulation_counts_model(  # numpydoc ignore=RT01
) -> MultiSignalModel:
    """Build a minimal model with a subpopulation count signal."""
    builder = PyrenewBuilder()
    builder.configure_latent(
        SubpopulationInfections,
        gen_int_rv=DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3])),
        I0_rv=DeterministicVariable("I0", 0.001),
        log_rt_time_0_rv=DeterministicVariable("initial_log_rt", 0.0),
        baseline_rt_process=fixed_random_walk(innovation_sd=0.05),
        subpop_rt_deviation_process=fixed_random_walk(innovation_sd=0.025),
    )
    builder.add_observation(
        SubpopulationCounts(
            name="regional_counts",
            ascertainment_rate_rv=DeterministicVariable("regional_ascertainment", 0.01),
            delay_distribution_rv=DeterministicPMF("regional_delay", jnp.array([1.0])),
            noise=PoissonNoise(),
        )
    )
    return builder.build()


def test_run_forwards_nested_subpopulation_indices() -> None:
    """Fit subpopulation counts with indices nested under the signal name."""
    model = _build_subpopulation_counts_model()
    n_days_post_init = 3
    num_samples = 2
    subpop_fractions = jnp.array([0.4, 0.6])
    subpop_indices = jnp.array([0, 1])
    counts = model.pad_observations(
        jnp.array(
            [
                [9.0, 11.0],
                [10.0, 12.0],
                [11.0, 13.0],
            ]
        )
    )

    model.run(
        num_warmup=2,
        num_samples=num_samples,
        rng_key=random.PRNGKey(42),
        mcmc_args={"progress_bar": False},
        n_days_post_init=n_days_post_init,
        population_size=1_000_000,
        subpop_fractions=subpop_fractions,
        regional_counts={
            "obs": counts,
            "subpop_indices": subpop_indices,
        },
    )

    assert model.mcmc is not None
    samples = model.mcmc.get_samples()
    assert samples
    assert "regional_counts_predicted" in samples
    assert samples["latent_infections"].shape[0] == num_samples
