"""Check weekly synthetic data against the infection and observation models."""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import polars as pl
import pytest

import pyrenew.datasets.datagen_he_CA_126 as datagen
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import PopulationInfections, RandomWalk, WeeklyTemporalProcess
from pyrenew.model import PyrenewBuilder
from pyrenew.observation import NegativeBinomialNoise, PopulationCounts


@pytest.fixture(scope="module")
def weekly_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Generate a weekly-Rt dataset in a temporary directory.

    Returns
    -------
    Path
        Directory containing observations and their generating parameters.
    """
    output = tmp_path_factory.mktemp("weekly_rt")
    datagen.generate(rt_cadence="weekly", output_dir=output)
    return output


def test_weekly_truth_and_dates(weekly_dataset: Path) -> None:
    """Daily Rt repeats the saved weekly truth on Sunday-Saturday weeks."""
    weekly = pl.read_csv(weekly_dataset / "weekly_rt.csv", try_parse_dates=True)
    daily = pl.read_csv(weekly_dataset / "daily_infections.csv", try_parse_dates=True)
    initial = pl.read_csv(
        weekly_dataset / "initial_infections.csv", try_parse_dates=True
    )
    hospital = pl.read_csv(
        weekly_dataset / "weekly_hospital_admissions.csv", try_parse_dates=True
    )
    params = json.loads((weekly_dataset / "true_parameters.json").read_text())

    assert len(weekly) == 18
    assert len(daily) == params["n_days"] == 126
    assert len(initial) == params["n_init"]
    np.testing.assert_array_equal(
        daily["true_rt"].to_numpy(), np.repeat(weekly["true_rt"].to_numpy(), 7)
    )
    np.testing.assert_array_equal(
        weekly["true_rt"].to_numpy(), params["rt_trajectory"]["weekly_values"]
    )
    assert weekly["week_start"].to_list() == daily["date"].to_list()[::7]
    assert weekly["week_end"].to_list() == daily["date"].to_list()[6::7]
    assert hospital["week_end"].to_list() == weekly["week_end"].to_list()
    assert all(day.weekday() == 6 for day in weekly["week_start"])
    assert all(day.weekday() == 5 for day in weekly["week_end"])
    assert initial["date"][-1] + timedelta(days=1) == daily["date"][0]
    assert initial["date"][0] == date.fromisoformat(params["initialization_start_date"])
    assert (
        initial["true_infections"][0] == params["i0_per_capita"] * params["population"]
    )


def test_known_parameters_reproduce_model_predictions(weekly_dataset: Path) -> None:
    """Known parameters reproduce infections, delays, weekday effects, and weekly means."""
    params = json.loads((weekly_dataset / "true_parameters.json").read_text())
    hospital = params["hospitalizations"]
    ed = params["ed_visits"]
    builder = PyrenewBuilder()
    builder.configure_latent(
        PopulationInfections,
        gen_int_rv=DeterministicPMF(
            "gen_int", jnp.asarray(params["generation_interval_pmf"])
        ),
        I0_rv=DeterministicVariable("I0", params["i0_per_capita"]),
        log_rt_time_0_rv=DeterministicVariable(
            "log_rt_time_0", params["log_rt_time_0"]
        ),
        single_rt_process=WeeklyTemporalProcess(
            RandomWalk(
                innovation_sd_rv=DeterministicVariable("sd", 0.05),
                parameterization="state",
            ),
            start_dow=params["week_start_dow"],
        ),
    )
    aggregations: tuple[Literal["daily", "weekly"], ...] = ("daily", "weekly")
    for aggregation in aggregations:
        builder.add_observation(
            PopulationCounts(
                name=f"hospital_{aggregation}",
                ascertainment_rate_rv=DeterministicVariable("ihr", hospital["ihr"]),
                delay_distribution_rv=DeterministicPMF(
                    "hospital_delay", jnp.asarray(hospital["delay_pmf"])
                ),
                noise=NegativeBinomialNoise(
                    DeterministicVariable(
                        "hospital_conc",
                        hospital[f"negbinom_concentration_{aggregation}"],
                    )
                ),
                aggregation=aggregation,
                start_dow=params["week_start_dow"] if aggregation == "weekly" else None,
            )
        )
    builder.add_observation(
        PopulationCounts(
            name="ed",
            ascertainment_rate_rv=DeterministicVariable("iedr", ed["iedr"]),
            delay_distribution_rv=DeterministicPMF(
                "ed_delay", jnp.asarray(ed["delay_pmf"])
            ),
            noise=NegativeBinomialNoise(
                DeterministicVariable("ed_conc", ed["negbinom_concentration"])
            ),
            day_of_week_rv=DeterministicVariable(
                "ed_dow", jnp.asarray(ed["day_of_week_effects"])
            ),
        )
    )
    model = builder.build()
    assert model.latent.n_initialization_points == params["n_init"]
    conditioned = numpyro.handlers.condition(
        model.sample,
        data={
            "log_rt_single_state": jnp.log(
                jnp.asarray(params["rt_trajectory"]["weekly_values"])[1:, None]
            )
        },
    )
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(conditioned, rng_seed=0)
    ).get_trace(
        n_days_post_init=params["n_days"],
        population_size=params["population"],
        obs_start_date=date.fromisoformat(params["start_date"]),
    )

    initial = pl.read_csv(weekly_dataset / "initial_infections.csv")
    daily = pl.read_csv(weekly_dataset / "daily_infections.csv")
    expected = pl.read_csv(weekly_dataset / "expected_observations.csv")
    weekly = pl.read_csv(weekly_dataset / "weekly_hospital_admissions.csv")
    full_infections = np.concatenate(
        [initial["true_infections"].to_numpy(), daily["true_infections"].to_numpy()]
    )
    n_init = params["n_init"]
    np.testing.assert_allclose(
        trace["latent_infections"]["value"], full_infections, rtol=1e-10
    )
    np.testing.assert_allclose(
        trace["PopulationInfections::rt_single"]["value"][:, 0],
        daily["true_rt"].to_numpy(),
        rtol=1e-12,
    )
    for site, column in (
        ("hospital_daily_predicted", "expected_hosp_admits"),
        ("ed_predicted", "expected_ed_visits"),
    ):
        np.testing.assert_allclose(
            trace[site]["value"][n_init:], expected[column].to_numpy(), rtol=1e-10
        )
    np.testing.assert_allclose(
        trace["hospital_weekly_predicted"]["value"][-len(weekly) :],
        weekly["expected_weekly_hosp_admits"].to_numpy(),
        rtol=1e-10,
    )


def test_weekly_reproducibility_and_seed(weekly_dataset: Path, tmp_path: Path) -> None:
    """The seed controls observations; Rt and expected values stay fixed."""
    with jax.enable_x64(False):
        datagen.generate(rt_cadence="weekly", output_dir=tmp_path)
        assert not jax.config.x64_enabled
    for original in weekly_dataset.iterdir():
        assert (tmp_path / original.name).read_bytes() == original.read_bytes()

    datagen.generate(rt_cadence="weekly", output_dir=tmp_path, seed=20240102)
    for filename in (
        "weekly_rt.csv",
        "initial_infections.csv",
        "daily_infections.csv",
        "expected_observations.csv",
    ):
        assert (tmp_path / filename).read_bytes() == (
            weekly_dataset / filename
        ).read_bytes()
    for filename in (
        "daily_hospital_admissions.csv",
        "weekly_hospital_admissions.csv",
        "daily_ed_visits.csv",
    ):
        assert (tmp_path / filename).read_bytes() != (
            weekly_dataset / filename
        ).read_bytes()


def test_weekly_observation_means_are_not_floored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Low-incidence simulations retain observation means below one."""
    monkeypatch.setattr(datagen, "WEEKLY_I0_PER_CAPITA", 1e-10)
    datagen.generate(rt_cadence="weekly", output_dir=tmp_path)
    expected = pl.read_csv(tmp_path / "expected_observations.csv")
    weekly = pl.read_csv(tmp_path / "weekly_hospital_admissions.csv")
    for values in (
        expected["expected_hosp_admits"],
        expected["expected_ed_visits"],
        weekly["expected_weekly_hosp_admits"],
    ):
        assert 0 < values.min() < values.max() < 1
