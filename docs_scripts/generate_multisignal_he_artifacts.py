# numpydoc ignore=GL08

"""Generate precomputed artifacts for the multisignal H+E tutorial.

Run this script from the repository root on a machine that can run both the
PyRenew tutorial model and the custom NumPyro H+E model.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import runpy
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import xarray as xr

import pyrenew.transformation as transformation
from pyrenew.ascertainment import AscertainmentModel, JointAscertainment
from pyrenew.datasets import (
    load_example_infection_admission_interval,
    load_synthetic_daily_ed_visits,
    load_synthetic_true_parameters,
    load_synthetic_weekly_hospital_admissions,
    write_synthetic_hew_model_dir,
)
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import DifferencedAR1, PopulationInfections, WeeklyTemporalProcess
from pyrenew.metaclass import RandomVariable
from pyrenew.model import PyrenewBuilder
from pyrenew.observation import NegativeBinomialNoise, PopulationCounts
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable
from pyrenew.time import MMWR_WEEK

N_DAYS_FIT = 126
OBS_START_DATE = date(2023, 11, 5)


class RatioLinkedAscertainment(AscertainmentModel):
    """Two ascertainment rates expressed as a base rate and a ratio."""

    def __init__(
        self,
        name: str,
        base_signal: str,
        linked_signal: str,
        base_rate_rv: RandomVariable,
        ratio_rv: RandomVariable,
    ) -> None:  # numpydoc ignore=GL08
        super().__init__(name=name, signals=(base_signal, linked_signal))
        self.base_signal = base_signal
        self.linked_signal = linked_signal
        self.base_rate_rv = base_rate_rv
        self.ratio_rv = ratio_rv

    def sample(
        self, **kwargs: object
    ) -> Mapping[str, jax.Array]:  # numpydoc ignore=RT01
        """Sample the base rate and ratio, then compute the linked rate."""
        base_rate = self.base_rate_rv()
        ratio = self.ratio_rv()
        linked_rate = base_rate * ratio
        numpyro.deterministic(f"{self.name}_{self.base_signal}", base_rate)
        numpyro.deterministic(f"{self.name}_{self.linked_signal}", linked_rate)
        return {self.base_signal: base_rate, self.linked_signal: linked_rate}


def git_sha(path: Path) -> str | None:  # numpydoc ignore=RT01
    """Return the git SHA for a repo path, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def build_pyrenew_model(
    ascertainment_parameterization: str = "joint",
) -> tuple[object, float, jax.Array, jax.Array]:  # numpydoc ignore=RT01
    """Build the PyRenew H+E tutorial model and aligned observations."""
    true_params = load_synthetic_true_parameters()
    weekly_hosp = load_synthetic_weekly_hospital_admissions()
    daily_ed = load_synthetic_daily_ed_visits()

    hosp_delay_pmf = jnp.array(
        load_example_infection_admission_interval()["probability_mass"].to_numpy()
    )
    ed_delay_pmf = jnp.array(true_params["ed_visits"]["delay_pmf"])
    population_size = float(weekly_hosp["pop"][0])

    gen_int_pmf = jnp.array(
        [
            0.6326975,
            0.2327564,
            0.0856263,
            0.03150015,
            0.01158826,
            0.00426308,
            0.0015683,
        ]
    )
    gen_int_rv = DeterministicPMF("gen_int", gen_int_pmf)
    i0_rv = TransformedVariable(
        name="I0",
        base_rv=DistributionalVariable(
            name="logit_I0",
            distribution=dist.Normal(
                transformation.SigmoidTransform().inv(true_params["i0_per_capita"]),
                0.25,
            ),
        ),
        transforms=transformation.SigmoidTransform(),
    )
    log_rt_time_0_rv = DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5))

    weekly_rt_process = WeeklyTemporalProcess(
        DifferencedAR1(
            autoreg_rv=DeterministicVariable("rt_diff_autoreg", 0.5),
            innovation_sd_rv=DeterministicVariable("rt_diff_innovation_sd", 0.03),
        ),
        start_dow=MMWR_WEEK,
    )

    if ascertainment_parameterization == "joint":
        ascertainment_center = 0.004
        ascertainment_sd = 0.3
        ascertainment_corr = 0.5
        ascertainment_cov = jnp.array(
            [
                [ascertainment_sd**2, ascertainment_corr * ascertainment_sd**2],
                [ascertainment_corr * ascertainment_sd**2, ascertainment_sd**2],
            ]
        )
        ascertainment = JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed_visits"),
            baseline_rates=jnp.array([ascertainment_center, ascertainment_center]),
            covariance_matrix=ascertainment_cov,
        )
    elif ascertainment_parameterization == "linked":
        iedr_rv = TransformedVariable(
            name="iedr",
            base_rv=DistributionalVariable(
                name="logit_iedr",
                distribution=dist.Normal(
                    transformation.SigmoidTransform().inv(0.004),
                    0.3,
                ),
            ),
            transforms=transformation.SigmoidTransform(),
        )
        ratio_rv = DistributionalVariable(
            "ihr_rel_iedr",
            dist.LogNormal(0.0, jnp.log(jnp.sqrt(2.0))),
        )
        ascertainment = RatioLinkedAscertainment(
            name="he_ascertainment",
            base_signal="ed_visits",
            linked_signal="hospital",
            base_rate_rv=iedr_rv,
            ratio_rv=ratio_rv,
        )
    else:
        raise ValueError(
            "ascertainment_parameterization must be either 'joint' or 'linked'."
        )

    hospital_obs = PopulationCounts(
        name="hospital",
        ascertainment_rate_rv=ascertainment.for_signal("hospital"),
        delay_distribution_rv=DeterministicPMF("hosp_delay", hosp_delay_pmf),
        noise=NegativeBinomialNoise(
            DistributionalVariable("hosp_conc", dist.LogNormal(5.0, 1.0))
        ),
        aggregation="weekly",
        reporting_schedule="regular",
        start_dow=MMWR_WEEK,
    )
    ed_day_of_week_rv = DeterministicVariable(
        "ed_day_of_week_effect",
        jnp.array(true_params["ed_visits"]["day_of_week_effects"]),
    )
    ed_obs = PopulationCounts(
        name="ed_visits",
        ascertainment_rate_rv=ascertainment.for_signal("ed_visits"),
        delay_distribution_rv=DeterministicPMF("ed_delay", ed_delay_pmf),
        noise=NegativeBinomialNoise(
            DistributionalVariable("ed_conc", dist.LogNormal(4.0, 1.0))
        ),
        day_of_week_rv=ed_day_of_week_rv,
    )

    builder = PyrenewBuilder()
    builder.configure_latent(
        PopulationInfections,
        gen_int_rv=gen_int_rv,
        I0_rv=i0_rv,
        log_rt_time_0_rv=log_rt_time_0_rv,
        single_rt_process=weekly_rt_process,
    )
    builder.add_ascertainment(ascertainment)
    builder.add_observation(hospital_obs)
    builder.add_observation(ed_obs)
    model = builder.build()

    ed_observed = model.pad_observations(
        jnp.array(daily_ed["ed_visits"].to_numpy(), dtype=jnp.float32)
    )
    hospital_observed = model.pad_aggregated_observations(
        jnp.array(weekly_hosp["weekly_hosp_admits"].to_numpy(), dtype=jnp.float32),
        observation_name="hospital",
        n_days_post_init=N_DAYS_FIT,
        obs_start_date=OBS_START_DATE,
    )
    return model, population_size, hospital_observed, ed_observed


def trim_time(ds: xr.Dataset, n_init: int) -> xr.Dataset:  # numpydoc ignore=RT01
    """Trim initialization points from an ArviZ dataset."""
    if "time" in ds.dims:
        ds = ds.isel(time=slice(n_init, None))
        ds = ds.assign_coords(time=range(ds.sizes["time"]))
    return ds


def run_pyrenew_fit(output_dir: Path) -> float:  # numpydoc ignore=RT01
    """Fit pyrenew-H-E and write posterior artifacts."""
    model, population_size, hospital_observed, ed_observed = build_pyrenew_model()

    jax.clear_caches()
    start_time = time.time()
    model.run(
        num_warmup=500,
        num_samples=500,
        rng_key=random.PRNGKey(42),
        mcmc_args={"num_chains": 4, "progress_bar": True},
        n_days_post_init=N_DAYS_FIT,
        population_size=population_size,
        obs_start_date=OBS_START_DATE,
        hospital={"obs": hospital_observed},
        ed_visits={"obs": ed_observed},
    )
    samples = model.mcmc.get_samples()
    jax.block_until_ready(samples)
    elapsed = time.time() - start_time

    n_init = model.latent.n_initialization_points
    idata = az.from_numpyro(
        model.mcmc,
        dims={
            "latent_infections": ["time"],
            "PopulationInfections::infections_aggregate": ["time"],
            "PopulationInfections::log_rt_single": ["time", "dummy"],
            "PopulationInfections::rt_single": ["time", "dummy"],
            "log_rt_single_weekly": ["rt_week", "dummy"],
            "hospital_predicted_daily": ["time"],
            "hospital_predicted": ["week"],
            "ed_visits_predicted": ["time"],
        },
    )
    idata_trimmed = idata.map_over_datasets(lambda ds: trim_time(ds, n_init))
    idata_trimmed.posterior["I0"] = 1 / (
        1 + np.exp(-idata_trimmed.posterior["logit_I0"])
    )

    posterior = idata_trimmed.posterior
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "pyrenew_fit_artifacts.npz",
        n_init=np.array(n_init, dtype=np.int64),
        I0=posterior["I0"].values,
        log_rt_time_0=posterior["log_rt_time_0"].values,
        he_ascertainment_hospital=posterior["he_ascertainment_hospital"].values,
        he_ascertainment_ed_visits=posterior["he_ascertainment_ed_visits"].values,
        rt_single_full=idata.posterior["PopulationInfections::rt_single"].values,
        rt_single_trimmed=posterior["PopulationInfections::rt_single"].values,
        latent_infections_trimmed=posterior["latent_infections"].values,
    )
    return elapsed


def run_linked_pyrenew_fit(output_dir: Path) -> float:  # numpydoc ignore=RT01
    """Fit linked-pyrenew-H-E and write posterior artifacts."""
    model, population_size, hospital_observed, ed_observed = build_pyrenew_model(
        ascertainment_parameterization="linked"
    )

    jax.clear_caches()
    start_time = time.time()
    model.run(
        num_warmup=500,
        num_samples=500,
        rng_key=random.PRNGKey(42),
        mcmc_args={"num_chains": 4, "progress_bar": True},
        n_days_post_init=N_DAYS_FIT,
        population_size=population_size,
        obs_start_date=OBS_START_DATE,
        hospital={"obs": hospital_observed},
        ed_visits={"obs": ed_observed},
    )
    samples = model.mcmc.get_samples()
    jax.block_until_ready(samples)
    elapsed = time.time() - start_time

    n_init = model.latent.n_initialization_points
    idata = az.from_numpyro(
        model.mcmc,
        dims={
            "latent_infections": ["time"],
            "PopulationInfections::infections_aggregate": ["time"],
            "PopulationInfections::log_rt_single": ["time", "dummy"],
            "PopulationInfections::rt_single": ["time", "dummy"],
            "log_rt_single_weekly": ["rt_week", "dummy"],
            "hospital_predicted_daily": ["time"],
            "hospital_predicted": ["week"],
            "ed_visits_predicted": ["time"],
        },
    )
    idata_trimmed = idata.map_over_datasets(lambda ds: trim_time(ds, n_init))
    idata_trimmed.posterior["I0"] = 1 / (
        1 + np.exp(-idata_trimmed.posterior["logit_I0"])
    )

    posterior = idata_trimmed.posterior
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "linked_pyrenew_fit_artifacts.npz",
        n_init=np.array(n_init, dtype=np.int64),
        I0=posterior["I0"].values,
        log_rt_time_0=posterior["log_rt_time_0"].values,
        he_ascertainment_hospital=posterior["he_ascertainment_hospital"].values,
        he_ascertainment_ed_visits=posterior["he_ascertainment_ed_visits"].values,
        ihr_rel_iedr=posterior["ihr_rel_iedr"].values,
        rt_single_full=idata.posterior["PopulationInfections::rt_single"].values,
        rt_single_trimmed=posterior["PopulationInfections::rt_single"].values,
        latent_infections_trimmed=posterior["latent_infections"].values,
    )
    return elapsed


def find_sample(  # numpydoc ignore=RT01
    samples: dict[str, np.ndarray], candidates: list[str]
) -> np.ndarray:
    """Get a posterior sample array from candidate variable names."""
    for name in candidates:
        if name in samples:
            return np.asarray(samples[name])
    raise KeyError(f"Could not find any of {candidates} in custom posterior samples.")


def reduce_to_chain_draw(  # numpydoc ignore=RT01
    values: np.ndarray, reducer: str = "mean"
) -> np.ndarray:
    """Reduce non-sample dimensions in a chain/draw posterior array."""
    values = np.asarray(values, dtype=float)
    if values.ndim <= 2:
        return values
    axes = tuple(range(2, values.ndim))
    if reducer == "first":
        index = (slice(None), slice(None), *([0] * (values.ndim - 2)))
        return values[index]
    if reducer == "mean":
        return values.mean(axis=axes)
    raise ValueError(f"Unknown reducer: {reducer}")


def reduce_to_chain_draw_time(values: np.ndarray) -> np.ndarray:  # numpydoc ignore=RT01
    """Normalize a posterior trajectory array to chain/draw/time shape."""
    values = np.asarray(values, dtype=float)
    values = np.squeeze(values)
    if values.ndim == 2:
        values = values[None, :, :]
    if values.ndim != 3:
        raise ValueError(
            "Expected custom latent infections to have chain/draw/time shape "
            f"after squeezing; got shape {values.shape}."
        )
    if values.shape[-1] < N_DAYS_FIT:
        raise ValueError(
            "Custom latent infections are shorter than the fit period: "
            f"{values.shape[-1]} < {N_DAYS_FIT}."
        )
    return values[..., -N_DAYS_FIT:]


def run_custom_fit(
    output_dir: Path,
    pyrenew_multisignal_dir: Path,
    cfa_stf_dir: Path,
    model_dir: Path,
) -> float:  # numpydoc ignore=RT01
    """Fit custom-H-E and write scalar posterior artifacts."""
    sys.path.insert(0, str(pyrenew_multisignal_dir / "src"))
    sys.path.insert(0, str(cfa_stf_dir))

    from pipelines.pyrenew_hew.fit_pyrenew_model import fit_and_save_model

    write_synthetic_hew_model_dir(model_dir, overwrite=True)
    with open(model_dir / "data" / "model_params.json") as file:
        model_params = json.load(file)
    population_size = float(model_params["population_size"])

    jax.clear_caches()
    stdout = io.StringIO()
    stderr = io.StringIO()
    start_time = time.time()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            fit_and_save_model(
                model_dir=model_dir,
                fit_ed_visits=True,
                fit_hospital_admissions=True,
                fit_wastewater=False,
                n_warmup=500,
                n_samples=500,
                n_chains=4,
                rng_key=42,
            )
        if hasattr(jax, "effects_barrier"):
            jax.effects_barrier()
    except Exception:
        print("fit_and_save_model failed. Last captured stdout:")
        print(stdout.getvalue()[-4000:])
        print("Last captured stderr:")
        print(stderr.getvalue()[-4000:])
        raise
    elapsed = time.time() - start_time

    with open(model_dir / "posterior_samples.pickle", "rb") as file:
        custom_mcmc = pickle.load(file)

    samples = custom_mcmc.get_samples(group_by_chain=True)
    ihr = reduce_to_chain_draw(find_sample(samples, ["IHR", "ihr"]))
    iedr = find_sample(samples, ["IEDR", "iedr"])
    latent_infections = population_size * reduce_to_chain_draw_time(
        find_sample(
            samples,
            [
                "latent_infections",
                "infections",
                "infections_aggregate",
                "PopulationInfections::infections_aggregate",
                "i_t",
                "I_t",
            ],
        )
    )
    rt = reduce_to_chain_draw_time(
        find_sample(samples, ["rt", "rtu_subpop", "Rt", "R_t"])
    )
    iedr_first_day = reduce_to_chain_draw(iedr, reducer="first")
    iedr_mean = reduce_to_chain_draw(iedr, reducer="mean")

    try:
        i0 = reduce_to_chain_draw(
            find_sample(samples, ["I0", "i0_first_obs_n_rv", "i0_first_obs_n"])
        )
    except KeyError:
        custom_priors = runpy.run_path(str(model_dir / "priors.py"))
        i0_value = float(custom_priors["i0_first_obs_n_rv"]())
        i0 = np.full_like(ihr, i0_value, dtype=float)

    scalar_draws = pd.DataFrame(
        {
            "chain": np.repeat(np.arange(ihr.shape[0]), ihr.shape[1]),
            "draw": np.tile(np.arange(ihr.shape[1]), ihr.shape[0]),
            "I0": i0.ravel(),
            "IHR": ihr.ravel(),
            "IEDR_first_day": iedr_first_day.ravel(),
            "IEDR_mean": iedr_mean.ravel(),
        }
    )
    scalar_draws["ratio"] = scalar_draws["IHR"] / scalar_draws["IEDR_mean"]
    output_dir.mkdir(parents=True, exist_ok=True)
    scalar_draws.to_csv(output_dir / "custom_he_scalar_draws.csv", index=False)
    np.savez_compressed(
        output_dir / "custom_he_fit_artifacts.npz",
        latent_infections_trimmed=latent_infections,
        rt_trimmed=rt,
    )
    return elapsed


def parse_args() -> argparse.Namespace:  # numpydoc ignore=GL08
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/tutorials/data/multisignal_he"),
    )
    parser.add_argument(
        "--pyrenew-multisignal-dir",
        type=Path,
        default=Path("~/github/CDC/pyrenew-multisignal").expanduser(),
    )
    parser.add_argument(
        "--cfa-stf-dir",
        type=Path,
        default=Path("~/github/CDC/cfa-stf-routine-forecasting").expanduser(),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("docs/tutorials/scratch/synthetic_pyrenew_hew_model"),
    )
    return parser.parse_args()


def main() -> None:  # numpydoc ignore=GL08
    args = parse_args()
    numpyro.set_host_device_count(4)
    numpyro.enable_x64()

    pyrenew_elapsed = run_pyrenew_fit(args.output_dir)
    linked_pyrenew_elapsed = run_linked_pyrenew_fit(args.output_dir)
    custom_elapsed = run_custom_fit(
        output_dir=args.output_dir,
        pyrenew_multisignal_dir=args.pyrenew_multisignal_dir,
        cfa_stf_dir=args.cfa_stf_dir,
        model_dir=args.model_dir,
    )

    timings = {
        "pyrenew-H-E": {
            "elapsed_seconds": pyrenew_elapsed,
            "num_warmup": 500,
            "num_samples": 500,
            "num_chains": 4,
            "rng_key": 42,
        },
        "custom-H-E": {
            "elapsed_seconds": custom_elapsed,
            "num_warmup": 500,
            "num_samples": 500,
            "num_chains": 4,
            "rng_key": 42,
        },
        "linked-pyrenew-H-E": {
            "elapsed_seconds": linked_pyrenew_elapsed,
            "num_warmup": 500,
            "num_samples": 500,
            "num_chains": 4,
            "rng_key": 42,
        },
    }
    with open(args.output_dir / "timings.json", "w") as file:
        json.dump(timings, file, indent=2)
        file.write("\n")

    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "pyrenew_sha": git_sha(Path.cwd()),
        "pyrenew_multisignal_sha": git_sha(args.pyrenew_multisignal_dir),
        "cfa_stf_sha": git_sha(args.cfa_stf_dir),
    }
    with open(args.output_dir / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print(f"Wrote multisignal H+E artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
