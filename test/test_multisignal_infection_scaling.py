"""
Unit tests for how MultiSignalModel scales latent infections for each
infection resolution.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from jax.typing import ArrayLike

from pyrenew.deterministic import DeterministicPMF
from pyrenew.latent import (
    LatentSample,
    PopulationInfections,
    SubpopulationInfections,
)
from pyrenew.model import MultiSignalModel
from pyrenew.observation import HierarchicalNormalNoise
from pyrenew.observation.base import BaseObservationProcess
from test.test_helpers import StubMeasurementObservation

POPULATION_SIZE = 1_000_000.0
SUBPOP_FRACTIONS = jnp.array([0.6, 0.3, 0.1])
N_DAYS_POST_INIT = 10


class RecordingObservation(BaseObservationProcess):
    """Observation process that records the infections it receives."""

    def __init__(self, name: str, resolution: str) -> None:
        """
        Initialize the recording observation.

        Parameters
        ----------
        name
            Observation name; the recorded site is ``{name}_infections``.
        resolution
            Infection resolution this observation requests.
        """
        super().__init__(
            name=name,
            temporal_pmf_rv=DeterministicPMF(f"{name}_pmf", jnp.array([1.0])),
        )
        self.resolution = resolution

    def lookback_days(self) -> int:
        """
        Return the number of days of infection history required.

        Returns
        -------
        int
            Zero.
        """
        return 0

    def infection_resolution(self) -> str:
        """
        Return the infection resolution given at construction.

        Returns
        -------
        str
            The requested infection resolution.
        """
        return self.resolution

    def _predicted_obs(self, infections: ArrayLike) -> ArrayLike:
        """
        Return infections unchanged.

        Parameters
        ----------
        infections
            Infections received from the model.

        Returns
        -------
        ArrayLike
            Same as input.
        """
        return infections

    def validate_data(
        self,
        n_total: int,
        n_subpops: int,
        **obs_data: dict[str, object],
    ) -> None:
        """
        Accept any data.

        Parameters
        ----------
        n_total
            Total number of time steps.
        n_subpops
            Number of subpopulations.
        **obs_data
            Ignored.
        """

    def sample(self, infections: ArrayLike, **kwargs: object) -> ArrayLike:
        """
        Record the infections received as a deterministic site.

        Parameters
        ----------
        infections
            Infections routed to this observation by the model.
        **kwargs
            Ignored.

        Returns
        -------
        ArrayLike
            The infections received.
        """
        self._deterministic("infections", infections)
        return infections


def _trace_model(
    model: MultiSignalModel,
    subpop_fractions: ArrayLike | None,
    **observation_data: dict[str, object],
) -> tuple[dict[str, np.ndarray], LatentSample]:
    """
    Sample the model once and recover the latent sample it used.

    The latent process is re-run with its sampled sites substituted from
    the model trace, so the returned ``LatentSample`` is the one the
    observations received.

    Parameters
    ----------
    model
        Model to sample.
    subpop_fractions
        Subpopulation fractions passed to ``model.sample``.
    **observation_data
        Per-observation data passed to ``model.sample``.

    Returns
    -------
    tuple[dict[str, np.ndarray], LatentSample]
        Site values from the model trace and the latent sample.
    """
    with numpyro.handlers.trace() as trace, numpyro.handlers.seed(rng_seed=0):
        model.sample(
            n_days_post_init=N_DAYS_POST_INIT,
            population_size=POPULATION_SIZE,
            subpop_fractions=subpop_fractions,
            **observation_data,
        )
    sites = {name: np.asarray(site["value"]) for name, site in trace.items()}
    sampled = {
        name: site["value"] for name, site in trace.items() if site["type"] == "sample"
    }
    with (
        numpyro.handlers.substitute(data=sampled),
        numpyro.handlers.seed(rng_seed=1),
    ):
        latent_sample = model.latent.sample(
            n_days_post_init=N_DAYS_POST_INIT,
            subpop_fractions=subpop_fractions,
        )
    return sites, latent_sample


@pytest.fixture
def subpop_trace(
    subpopulation_infections: SubpopulationInfections,
    hierarchical_normal_noise: HierarchicalNormalNoise,
) -> tuple[dict[str, np.ndarray], LatentSample]:
    """
    Trace a SubpopulationInfections model with one observation per resolution.

    Parameters
    ----------
    subpopulation_infections
        Latent process with three subpopulations supplied at sample time.
    hierarchical_normal_noise
        Noise model for the measurement observation.

    Returns
    -------
    tuple[dict[str, np.ndarray], LatentSample]
        Site values from the model trace and the latent sample.
    """
    model = MultiSignalModel(
        latent_process=subpopulation_infections,
        observations={
            "agg": RecordingObservation("agg", "aggregate"),
            "counts": RecordingObservation("counts", "subpop"),
            "ww": StubMeasurementObservation(
                name="ww",
                temporal_pmf_rv=DeterministicPMF("ww_pmf", jnp.array([1.0])),
                noise=hierarchical_normal_noise,
            ),
        },
    )
    n_subpops = SUBPOP_FRACTIONS.size
    return _trace_model(
        model,
        SUBPOP_FRACTIONS,
        ww={
            "times": jnp.full(
                n_subpops, subpopulation_infections.n_initialization_points
            ),
            "subpop_indices": jnp.arange(n_subpops),
            "sensor_indices": jnp.arange(n_subpops),
            "n_sensors": n_subpops,
        },
    )


def test_subpop_counts_sum_to_aggregate(
    subpop_trace: tuple[dict[str, np.ndarray], LatentSample],
) -> None:
    """
    Recorded subpopulation infection counts sum to the aggregate count.

    Parameters
    ----------
    subpop_trace
        Traced SubpopulationInfections model.
    """
    sites, _ = subpop_trace
    np.testing.assert_allclose(
        sites["latent_infections_by_subpop"].sum(axis=1),
        sites["latent_infections"],
        rtol=1e-12,
    )


def test_subpop_counts_scaled_by_subpop_population(
    subpop_trace: tuple[dict[str, np.ndarray], LatentSample],
) -> None:
    """
    A "subpop" observation receives infections per resident times the
    population of each subpopulation.

    Parameters
    ----------
    subpop_trace
        Traced SubpopulationInfections model.
    """
    sites, latent_sample = subpop_trace
    expected = (
        np.asarray(latent_sample.all_subpops)
        * POPULATION_SIZE
        * np.asarray(SUBPOP_FRACTIONS)
    )
    np.testing.assert_allclose(sites["counts_infections"], expected, rtol=1e-12)
    np.testing.assert_allclose(
        sites["latent_infections_by_subpop"], expected, rtol=1e-12
    )


def test_measurement_receives_per_capita(
    subpop_trace: tuple[dict[str, np.ndarray], LatentSample],
) -> None:
    """
    A MeasurementObservation receives infections per resident.

    Parameters
    ----------
    subpop_trace
        Traced SubpopulationInfections model.
    """
    sites, latent_sample = subpop_trace
    np.testing.assert_allclose(
        sites["ww_predicted"],
        np.asarray(latent_sample.all_subpops),
        rtol=1e-12,
    )


def test_population_infections_unchanged(
    population_infections: PopulationInfections,
) -> None:
    """
    With PopulationInfections and no subpopulation fractions, aggregate
    and subpopulation counts are both infections per resident times the
    population size.

    Parameters
    ----------
    population_infections
        Latent process with a single population.
    """
    model = MultiSignalModel(
        latent_process=population_infections,
        observations={
            "agg": RecordingObservation("agg", "aggregate"),
            "counts": RecordingObservation("counts", "subpop"),
        },
    )
    sites, latent_sample = _trace_model(model, None)
    expected = np.asarray(latent_sample.aggregate) * POPULATION_SIZE
    np.testing.assert_allclose(sites["agg_infections"], expected, rtol=1e-12)
    assert sites["counts_infections"].shape == (expected.size, 1)
    np.testing.assert_allclose(sites["counts_infections"][:, 0], expected, rtol=1e-12)


def test_validate_accepts_subpop_per_capita(
    subpopulation_infections: SubpopulationInfections,
) -> None:
    """
    MultiSignalModel accepts an observation requesting "subpop_per_capita".

    Parameters
    ----------
    subpopulation_infections
        Latent process with subpopulations.
    """
    model = MultiSignalModel(
        latent_process=subpopulation_infections,
        observations={"ww": RecordingObservation("ww", "subpop_per_capita")},
    )
    model.validate()
