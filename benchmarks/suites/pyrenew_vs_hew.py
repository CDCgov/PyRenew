# ruff: noqa: E402

"""pyrenew_vs_hew benchmark suite.

Compare the production ``pyrenew-multisignal`` HEW model against a PyRenew
``MultiSignalModel`` on the same synthetic H+E data. The PyRenew candidate
uses weekly-aggregated hospital admissions plus daily ED visits, a joint
Gaussian ascertainment, and a weekly state-centered $\\mathcal{R}(t)$ process,
and deliberately omits the daily ED day-of-week effect (which prior fits found
poorly identified). The production HEW model retains its ED weekday effect, so
the comparison is intentionally asymmetric on that axis.

The HEW candidate requires ``pyrenew-multisignal`` and
``cfa-stf-routine-forecasting`` to be importable; pass their checkout paths
with ``--pyrenew-multisignal-dir`` and ``--cfa-stf-dir``.

Per benchmark convention, the model-construction code lives in this suite;
``benchmarks.core`` provides only the machinery. Run from the repository root:

    python -m benchmarks.suites.pyrenew_vs_hew --quick

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_AVAILABLE_CPUS: int = os.cpu_count() or 1
_DEFAULT_DEVICE_COUNT: int = min(8, _AVAILABLE_CPUS)
_DEFAULT_NUM_CHAINS: int = min(4, _AVAILABLE_CPUS)
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault(
    "XLA_FLAGS", f"--xla_force_host_platform_device_count={_DEFAULT_DEVICE_COUNT}"
)

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import pyrenew.transformation as transformation
from benchmarks.core.comparison import DEFAULT_METRICS, ComparisonSpec
from benchmarks.core.datasets import (
    SYNTHETIC_HE_WEEKLY_HOSPITAL,
    SyntheticProvider,
)
from benchmarks.core.hew_model import (
    DEFAULT_CFA_STF_DIR,
    DEFAULT_PYRENEW_MULTISIGNAL_DIR,
    HEW_RT_SITE_NAMES,
    build_hew_model,
)
from benchmarks.core.models import BuiltFit, align_weekly_observations
from benchmarks.core.priors import real_he_i0_prior
from benchmarks.core.reporting import (
    print_comparison_tables,
    print_fit_progress,
    write_results,
)
from benchmarks.core.runner import (
    Candidate,
    FitResult,
    McmcSettings,
    fit_candidate,
)
from benchmarks.core.signals import DatasetBundle
from pyrenew.ascertainment import JointAscertainment
from pyrenew.datasets import write_synthetic_hew_model_dir
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    DifferencedAR1,
    PopulationInfections,
    WeeklyTemporalProcess,
)
from pyrenew.model import PyrenewBuilder
from pyrenew.observation import NegativeBinomialNoise, PopulationCounts
from pyrenew.randomvariable import (
    DistributionalVariable,
    TransformedVariable,
)
from pyrenew.time import MMWR_WEEK

SUITE_NAME = "pyrenew_vs_hew"
DEFAULT_OUTPUT_DIR = Path("benchmarks/results")
DATASET_NAME = SYNTHETIC_HE_WEEKLY_HOSPITAL
PYRENEW_ARM = "pyrenew-state-nodow"
HEW_ARM = "hew"

COMPARISON_SPEC: ComparisonSpec = ComparisonSpec(
    name=SUITE_NAME,
    arms=(HEW_ARM, PYRENEW_ARM),
    baseline=HEW_ARM,
    match_keys=("dataset",),
    metrics=DEFAULT_METRICS,
)


def _build_pyrenew_state_nodow(bundle: DatasetBundle) -> BuiltFit:
    """Build the PyRenew H+E candidate without an ED day-of-week effect.

    Joint Gaussian ascertainment over hospital and ED visit rates, weekly
    state-centered $\\mathcal{R}(t)$, weekly-aggregated hospital admissions,
    and daily ED visits with no day-of-week multiplier.

    Returns
    -------
    BuiltFit
        Model and run kwargs ready for the runner.
    """
    hospital_signal = bundle.signals["hospital"]
    ed_signal = bundle.signals["ed_visits"]

    if "i0_per_capita" in bundle.fixed_params:
        i0_per_capita = float(bundle.fixed_params["i0_per_capita"])
        i0_rv = TransformedVariable(
            name="I0",
            base_rv=DistributionalVariable(
                name="logit_I0",
                distribution=dist.Normal(
                    transformation.SigmoidTransform().inv(i0_per_capita), 0.25
                ),
            ),
            transforms=transformation.SigmoidTransform(),
        )
    else:
        i0_rv = real_he_i0_prior()

    sd = 0.3
    corr = 0.5
    cov = jnp.array([[sd**2, corr * sd**2], [corr * sd**2, sd**2]])
    ascertainment = JointAscertainment(
        name="he_ascertainment",
        signals=("hospital", "ed_visits"),
        baseline_rates=jnp.array([0.004, 0.004]),
        covariance_matrix=cov,
    )

    rt_process = WeeklyTemporalProcess(
        DifferencedAR1(
            autoreg_rv=DeterministicVariable("rt_diff_autoreg", 0.9),
            innovation_sd_rv=DeterministicVariable("rt_diff_innovation_sd", 0.05),
            parameterization="state",
        ),
        start_dow=MMWR_WEEK,
    )

    builder = PyrenewBuilder()
    builder.configure_latent(
        PopulationInfections,
        gen_int_rv=DeterministicPMF("gen_int", bundle.gen_int_pmf),
        I0_rv=i0_rv,
        log_rt_time_0_rv=DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5)),
        single_rt_process=rt_process,
    )
    builder.add_ascertainment(ascertainment)
    builder.add_observation(
        PopulationCounts(
            name="hospital",
            ascertainment_rate_rv=ascertainment.for_signal("hospital"),
            delay_distribution_rv=DeterministicPMF(
                "hosp_delay", hospital_signal.extras["delay_pmf"]
            ),
            noise=NegativeBinomialNoise(
                DistributionalVariable("hosp_conc", dist.LogNormal(5.0, 1.0))
            ),
            aggregation="weekly",
            reporting_schedule="regular",
            start_dow=MMWR_WEEK,
        )
    )
    builder.add_observation(
        PopulationCounts(
            name="ed_visits",
            ascertainment_rate_rv=ascertainment.for_signal("ed_visits"),
            delay_distribution_rv=DeterministicPMF(
                "ed_delay", ed_signal.extras["delay_pmf"]
            ),
            noise=NegativeBinomialNoise(
                DistributionalVariable("ed_conc", dist.LogNormal(4.0, 1.0))
            ),
        )
    )
    model = builder.build()

    hospital_obs = align_weekly_observations(
        model,
        "hospital",
        hospital_signal.values,
        bundle.obs_start_date,
        bundle.n_days_post_init,
    )
    ed_obs = model.pad_observations(ed_signal.values)
    return BuiltFit(
        model=model,
        run_kwargs={
            "n_days_post_init": bundle.n_days_post_init,
            "population_size": bundle.population_size,
            "obs_start_date": bundle.obs_start_date,
            "hospital": {"obs": hospital_obs},
            "ed_visits": {"obs": ed_obs},
        },
        dataset_name=bundle.name,
    )


def _build_candidates(
    args: argparse.Namespace, bundle: DatasetBundle
) -> list[Candidate]:
    """Assemble the HEW and PyRenew candidates for the suite.

    Returns
    -------
    list[Candidate]
        The two comparison candidates, HEW first.
    """
    write_synthetic_hew_model_dir(args.model_dir, overwrite=True)

    def build_hew() -> BuiltFit:
        """Build the production HEW model from the synthetic model directory.

        Returns
        -------
        BuiltFit
            Assembled HEW model and run kwargs.
        """
        return build_hew_model(
            args.model_dir,
            dataset_name=DATASET_NAME,
            pyrenew_multisignal_dir=args.pyrenew_multisignal_dir,
            cfa_stf_dir=args.cfa_stf_dir,
        )

    return [
        Candidate(
            name=HEW_ARM,
            arm=HEW_ARM,
            config_fields={
                "model_family": "hew",
                "ascertainment": "linked",
                "rt": "weekly-state",
                "ed_day_of_week": True,
            },
            build=build_hew,
            rt_site_names=HEW_RT_SITE_NAMES,
        ),
        Candidate(
            name=PYRENEW_ARM,
            arm=PYRENEW_ARM,
            config_fields={
                "model_family": "pyrenew",
                "ascertainment": "joint",
                "rt": "weekly-state",
                "ed_day_of_week": False,
            },
            build=lambda: _build_pyrenew_state_nodow(bundle),
        ),
    ]


def _parse_args() -> argparse.Namespace:
    """Parse the pyrenew_vs_hew CLI.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-chains", type=int, default=_DEFAULT_NUM_CHAINS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pyrenew-multisignal-dir",
        type=Path,
        default=DEFAULT_PYRENEW_MULTISIGNAL_DIR,
        help="Path to the pyrenew-multisignal checkout (for the HEW candidate).",
    )
    parser.add_argument(
        "--cfa-stf-dir",
        type=Path,
        default=DEFAULT_CFA_STF_DIR,
        help="Path to the cfa-stf-routine-forecasting checkout.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("benchmarks/results/synthetic_hew_model"),
        help="Directory to write the synthetic HEW model inputs into.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing result files; print summary tables only.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show per-chain progress bars during MCMC.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke run: 50 warmup, 50 samples, 1 chain. Overrides "
            "--num-warmup / --num-samples / --num-chains."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the pyrenew_vs_hew suite from the command line."""
    args = _parse_args()
    if args.quick:
        args.num_warmup = 50
        args.num_samples = 50
        args.num_chains = 1

    numpyro.set_host_device_count(args.num_chains)
    numpyro.enable_x64()

    bundle = SyntheticProvider().get(SYNTHETIC_HE_WEEKLY_HOSPITAL)
    settings = McmcSettings(
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        progress_bar=args.progress_bar,
    )
    candidates = _build_candidates(args, bundle)

    n_fits = len(candidates) * args.repeats
    print(
        f"{SUITE_NAME} suite: {len(candidates)} candidate(s) x "
        f"{args.repeats} repeat(s) = {n_fits} fits",
        flush=True,
    )

    results: list[FitResult] = []
    for candidate in candidates:
        for repeat in range(args.repeats):
            print(
                f">> fitting {candidate.name} (repeat {repeat + 1}/{args.repeats}) ...",
                flush=True,
            )
            result = fit_candidate(candidate, settings, repeat)
            results.append(result)
            print_fit_progress(candidate.name, repeat, args.repeats, result)

    print_comparison_tables(results, COMPARISON_SPEC)
    if not args.no_write:
        write_results(
            args.output_dir,
            suite_name=SUITE_NAME,
            results=results,
            spec=COMPARISON_SPEC,
        )
        print(f"\nWrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
