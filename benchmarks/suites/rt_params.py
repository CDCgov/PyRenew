"""rt_params benchmark suite.

Compare ``innovation`` and ``state`` parameterizations of the weekly Rt
temporal process. Each candidate name encodes the model family and
parameterization.

Run as a module from the repository root:

    python -m benchmarks.suites.rt_params --quick

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

_AVAILABLE_CPUS: int = os.cpu_count() or 1
_DEFAULT_DEVICE_COUNT: int = min(8, _AVAILABLE_CPUS)
_DEFAULT_NUM_CHAINS: int = min(4, _AVAILABLE_CPUS)
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault(
    "XLA_FLAGS", f"--xla_force_host_platform_device_count={_DEFAULT_DEVICE_COUNT}"
)

import numpyro  # noqa: E402

from benchmarks.core.datasets import (  # noqa: E402
    SYNTHETIC_HE_WEEKLY_HOSPITAL,
    SyntheticProvider,
)
from benchmarks.core.models import BuildConfig, build_he_model  # noqa: E402
from benchmarks.core.real_data import RealDataProvider, RealDataSpec  # noqa: E402
from benchmarks.core.reporting import (  # noqa: E402
    print_fit_progress,
    print_pairwise_tables,
    write_results,
)
from benchmarks.core.runner import (  # noqa: E402
    FitResult,
    McmcSettings,
    fit_and_measure,
)
from benchmarks.core.signals import DatasetBundle  # noqa: E402

SUITE_NAME = "rt_params"
DEFAULT_OUTPUT_DIR = Path("benchmarks/results")
DEFAULT_TIGHT_SD = 0.01
DEFAULT_LOOSE_SD = 0.10
DEFAULT_TIGHT_AUTOREG = 0.9
DEFAULT_LOOSE_AUTOREG = 0.5
TIGHT_PRIOR: tuple[float, float] = (DEFAULT_TIGHT_SD, DEFAULT_TIGHT_AUTOREG)
LOOSE_PRIOR: tuple[float, float] = (DEFAULT_LOOSE_SD, DEFAULT_LOOSE_AUTOREG)
DEFAULT_REAL_DISEASE = "COVID-19"
DEFAULT_REAL_LOCATION = "US"
DEFAULT_REAL_TRAINING_DAYS = 150
DEFAULT_REAL_OMIT_DAYS = 2
REAL_HE_DATASET = "real_he"
Disease = str


PARAMETERIZATIONS: tuple[str, ...] = ("innovation", "state")


def _load_bundles(args: argparse.Namespace) -> dict[str, DatasetBundle]:
    """Load the H+E dataset bundle for the suite.

    Returns
    -------
    dict[str, DatasetBundle]
        Loaded bundle keyed by dataset identifier.
    """
    bundles: dict[str, DatasetBundle] = {}
    if args.data_source == "synthetic":
        bundles[SYNTHETIC_HE_WEEKLY_HOSPITAL] = SyntheticProvider().get(
            SYNTHETIC_HE_WEEKLY_HOSPITAL
        )
        return bundles

    provider = RealDataProvider(
        {
            REAL_HE_DATASET: RealDataSpec(
                disease=args.disease,
                loc_abbr=args.location,
                as_of=args.as_of,
                n_training_days=args.training_days,
                n_days_to_omit=args.omit_last_days,
                signals=("hospital", "ed_visits"),
            )
        }
    )
    bundles[SYNTHETIC_HE_WEEKLY_HOSPITAL] = provider.get(REAL_HE_DATASET)
    return bundles


def _print_data_summary(bundles: dict[str, DatasetBundle]) -> None:
    """Print a compact summary of loaded benchmark data bundles."""
    for bundle in bundles.values():
        print()
        print(f"Dataset: {bundle.name}")
        print(f"  population_size: {bundle.population_size:g}")
        print(f"  obs_start_date: {bundle.obs_start_date}")
        print(f"  n_days_post_init: {bundle.n_days_post_init}")
        print(f"  gen_int_pmf_len: {len(bundle.gen_int_pmf)}")
        fixed_keys = ", ".join(sorted(bundle.fixed_params)) or "none"
        print(f"  fixed_params: {fixed_keys}")

        for signal in bundle.signals.values():
            values = np.asarray(signal.values, dtype=float)
            finite = values[np.isfinite(values)]
            missing = int(values.size - finite.size)
            start_date = signal.start_date
            if signal.times is None:
                step_days = 7 if signal.cadence == "weekly" else 1
                end_date = start_date + dt.timedelta(days=(len(values) - 1) * step_days)
            else:
                times = np.asarray(signal.times)
                end_date = start_date + dt.timedelta(days=int(np.max(times)))

            if finite.size:
                value_summary = (
                    f"min={np.min(finite):.4g}, "
                    f"mean={np.mean(finite):.4g}, "
                    f"max={np.max(finite):.4g}"
                )
            else:
                value_summary = "no finite values"

            print(f"  signal: {signal.name}")
            print(f"    cadence: {signal.cadence}")
            print(f"    n_obs: {len(values)}")
            print(f"    date_range: {start_date} to {end_date}")
            print(f"    missing_or_nan: {missing}")
            print(f"    values: {value_summary}")
            print(f"    extras: {', '.join(sorted(signal.extras)) or 'none'}")


def _parse_pair(arg: str) -> tuple[float, float]:
    """Parse an explicit ``sd,autoreg`` prior pair.

    Returns
    -------
    tuple[float, float]
        ``(innovation_sd, autoreg)``.
    """
    parts = arg.split(",")
    if len(parts) != 2:
        raise ValueError(
            f"Prior pair must be 'sd,autoreg' (e.g. '0.05,0.7'); got {arg!r}"
        )
    try:
        sd = float(parts[0])
        ar = float(parts[1])
    except ValueError as exc:
        raise ValueError(f"Could not parse prior pair {arg!r}: {exc}") from exc
    return sd, ar


def _parse_date(arg: str) -> dt.date:
    """Parse a CLI date in YYYY-MM-DD format.

    Returns
    -------
    datetime.date
        Parsed calendar date.
    """
    try:
        return dt.date.fromisoformat(arg)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected date in YYYY-MM-DD format; got {arg!r}"
        ) from exc


def _resolve_priors(args: Sequence[str]) -> list[tuple[float, float]]:
    """Resolve CLI ``--prior`` arguments to ``(innovation_sd, autoreg)`` pairs.

    Returns
    -------
    list[tuple[float, float]]
        Prior regimes to fit each candidate under.
    """
    if not args:
        return [TIGHT_PRIOR]
    out: list[tuple[float, float]] = []
    for a in args:
        if a == "tight":
            out.append(TIGHT_PRIOR)
        elif a == "loose":
            out.append(LOOSE_PRIOR)
        elif a == "both":
            out.extend([TIGHT_PRIOR, LOOSE_PRIOR])
        else:
            out.append(_parse_pair(a))
    return list(dict.fromkeys(out))


def _parse_args() -> argparse.Namespace:
    """Parse the rt_params CLI.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source",
        choices=("synthetic", "real"),
        default="synthetic",
        help=(
            "Data source for H+E candidates. 'real' requires CDC-internal "
            "cfa-stf-routine-forecasting data access."
        ),
    )
    parser.add_argument(
        "--disease",
        choices=("COVID-19", "Influenza", "RSV"),
        default=DEFAULT_REAL_DISEASE,
        help="Disease for --data-source real.",
    )
    parser.add_argument(
        "--location",
        default=DEFAULT_REAL_LOCATION,
        help="Location abbreviation for --data-source real, e.g. US or CA.",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_date,
        default=None,
        help="Vintage date for --data-source real, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--training-days",
        type=int,
        default=DEFAULT_REAL_TRAINING_DAYS,
        help="Training window length for --data-source real.",
    )
    parser.add_argument(
        "--omit-last-days",
        type=int,
        default=DEFAULT_REAL_OMIT_DAYS,
        help="Trailing days to omit from --data-source real.",
    )
    parser.add_argument(
        "--dry-run-data",
        action="store_true",
        help="Load and summarize selected data, then exit before model fitting.",
    )
    parser.add_argument(
        "--prior",
        action="append",
        default=[],
        help=(
            "Prior regime: 'tight' "
            f"(sd={DEFAULT_TIGHT_SD:g}, autoreg={DEFAULT_TIGHT_AUTOREG:g}), "
            "'loose' "
            f"(sd={DEFAULT_LOOSE_SD:g}, autoreg={DEFAULT_LOOSE_AUTOREG:g}), "
            "'both', or an explicit 'sd,autoreg' pair (e.g. '0.05,0.7'). "
            "Repeat to fit under multiple regimes."
        ),
    )
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-chains", type=int, default=_DEFAULT_NUM_CHAINS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write CSV / JSON / Markdown results.",
    )
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
    args = parser.parse_args()
    if args.data_source == "real" and args.as_of is None:
        parser.error("--as-of is required when --data-source real")
    if args.training_days <= 0:
        parser.error("--training-days must be positive")
    if args.omit_last_days < 0:
        parser.error("--omit-last-days must be non-negative")
    return args


def _fit_label(
    parameterization: str, innovation_sd: float, autoreg: float, n_priors: int
) -> str:
    """Compose a per-fit display label.

    Returns
    -------
    str
        Parameterization name, extended with the prior regime when more than
        one is fit.
    """
    if n_priors > 1:
        return f"{parameterization}@sd={innovation_sd:g},ar={autoreg:g}"
    return parameterization


def main() -> None:
    """Run the rt_params suite from the command line."""
    args = _parse_args()
    if args.quick:
        args.num_warmup = 50
        args.num_samples = 50
        args.num_chains = 1

    numpyro.set_host_device_count(args.num_chains)
    numpyro.enable_x64()

    priors = _resolve_priors(args.prior)
    bundles = _load_bundles(args)
    if args.dry_run_data:
        _print_data_summary(bundles)
        return
    settings = McmcSettings(
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        progress_bar=args.progress_bar,
    )
    bundle = bundles[SYNTHETIC_HE_WEEKLY_HOSPITAL]

    n_fits = len(PARAMETERIZATIONS) * len(priors) * args.repeats
    print(
        f"rt_params suite: {len(PARAMETERIZATIONS)} parameterization(s) x "
        f"{len(priors)} prior(s) x {args.repeats} repeat(s) = {n_fits} fits",
        flush=True,
    )

    results: list[FitResult] = []
    for innovation_sd, autoreg in priors:
        for parameterization in PARAMETERIZATIONS:
            config = BuildConfig(
                parameterization=parameterization,
                innovation_sd=innovation_sd,
                autoreg=autoreg,
            )
            for repeat in range(args.repeats):
                label = _fit_label(
                    parameterization, innovation_sd, autoreg, len(priors)
                )
                print(
                    f">> fitting {label} (repeat {repeat + 1}/{args.repeats}) ...",
                    flush=True,
                )
                built = build_he_model(config, bundle)
                result = fit_and_measure(
                    candidate=label,
                    built=built,
                    config=config,
                    settings=settings,
                    repeat=repeat,
                )
                results.append(result)
                print_fit_progress(label, repeat, args.repeats, result)

    print_pairwise_tables(results)
    if not args.no_write:
        write_results(args.output_dir, suite_name=SUITE_NAME, results=results)
        print(f"\nWrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
