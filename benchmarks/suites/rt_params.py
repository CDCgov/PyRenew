"""rt_params benchmark suite.

Compare ``innovation`` and ``state`` parameterizations of the Rt temporal
process across a configurable design matrix. Each candidate name encodes the
model family, Rt cadence, and parameterization.

Run as a module from the repository root:

    python -m benchmarks.suites.rt_params --quick

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

_AVAILABLE_CPUS: int = os.cpu_count() or 1
_DEFAULT_DEVICE_COUNT: int = min(8, _AVAILABLE_CPUS)
_DEFAULT_NUM_CHAINS: int = min(4, _AVAILABLE_CPUS)
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault(
    "XLA_FLAGS", f"--xla_force_host_platform_device_count={_DEFAULT_DEVICE_COUNT}"
)

import numpyro

from benchmarks.core.models import (
    BuildConfig,
    build_he_model,
    build_subpop_hospital_wastewater_model,
)
from benchmarks.core.reporting import (
    print_fit_progress,
    print_pairwise_tables,
    write_results,
)
from benchmarks.core.runner import FitResult, McmcSettings, fit_and_measure

SUITE_NAME = "rt_params"
DEFAULT_OUTPUT_DIR = Path("benchmarks/results")
DEFAULT_TIGHT_SD = 0.01
DEFAULT_LOOSE_SD = 0.10
DEFAULT_TIGHT_AUTOREG = 0.9
DEFAULT_LOOSE_AUTOREG = 0.5
TIGHT_PRIOR: tuple[float, float] = (DEFAULT_TIGHT_SD, DEFAULT_TIGHT_AUTOREG)
LOOSE_PRIOR: tuple[float, float] = (DEFAULT_LOOSE_SD, DEFAULT_LOOSE_AUTOREG)

HE_CANDIDATES = (
    "he_daily_innovation",
    "he_daily_state",
    "he_weekly_innovation",
    "he_weekly_state",
)

SUBPOP_CANDIDATES = (
    "subpop_hw_innovation",
    "subpop_hw_state",
)

ALL_CANDIDATES = HE_CANDIDATES + SUBPOP_CANDIDATES
DEFAULT_CANDIDATES = HE_CANDIDATES


def _parse_he_candidate(name: str, innovation_sd: float, autoreg: float) -> BuildConfig:
    """Parse an ``he_<cadence>_<param>`` candidate name.

    Returns
    -------
    BuildConfig
        Build configuration for the H+E model.
    """
    parts = name.split("_")
    if len(parts) != 3 or parts[0] != "he":
        raise ValueError(f"Expected 'he_<cadence>_<param>', got {name!r}")
    _, cadence, parameterization = parts
    if cadence not in ("daily", "weekly"):
        raise ValueError(f"Unknown cadence in candidate {name!r}")
    if parameterization not in ("innovation", "state"):
        raise ValueError(f"Unknown parameterization in candidate {name!r}")
    return BuildConfig(
        parameterization=parameterization,
        rt_cadence=cadence,
        innovation_sd=innovation_sd,
        autoreg=autoreg,
    )


def _parse_subpop_candidate(
    name: str, innovation_sd: float, autoreg: float
) -> BuildConfig:
    """Parse a ``subpop_hw_<param>`` candidate name.

    Returns
    -------
    BuildConfig
        Build configuration for the hospital+wastewater subpopulation model.
    """
    if name == "subpop_hw_innovation":
        parameterization = "innovation"
    elif name == "subpop_hw_state":
        parameterization = "state"
    else:
        raise ValueError(f"Unknown subpopulation candidate {name!r}")
    return BuildConfig(
        parameterization=parameterization,
        rt_cadence="daily",
        innovation_sd=innovation_sd,
        autoreg=autoreg,
    )


def _build_for_candidate(name: str, config: BuildConfig):
    """Dispatch to the right model builder for ``name``.

    Returns
    -------
    BuiltFit
        Assembled model and run kwargs.
    """
    if name.startswith("he_"):
        return build_he_model(config)
    if name.startswith("subpop_hw_"):
        return build_subpop_hospital_wastewater_model(config)
    raise ValueError(f"No builder is registered for candidate {name!r}")


def _resolve_candidates(args: Sequence[str]) -> list[str]:
    """Resolve CLI ``--candidate`` arguments, expanding ``all``.

    Returns
    -------
    list[str]
        De-duplicated candidate names in declaration order.
    """
    if not args:
        return list(DEFAULT_CANDIDATES)
    names: list[str] = []
    for a in args:
        if a == "all":
            names.extend(ALL_CANDIDATES)
        elif a == "he":
            names.extend(HE_CANDIDATES)
        elif a == "subpop":
            names.extend(SUBPOP_CANDIDATES)
        else:
            names.append(a)
    unknown = sorted(set(names) - set(ALL_CANDIDATES))
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}")
    return list(dict.fromkeys(names))


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
        "--candidate",
        action="append",
        default=[],
        help=(
            "Candidate name, or one of {all, he, subpop}. May be repeated. "
            f"Available: {', '.join(ALL_CANDIDATES)}."
        ),
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
            "Repeat to fit each candidate under multiple regimes."
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
        "--no-x64",
        action="store_true",
        help="Disable NumPyro / JAX 64-bit precision (enabled by default).",
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


def _candidate_label(
    name: str, innovation_sd: float, autoreg: float, n_priors: int
) -> str:
    """Compose a per-fit display label.

    Returns
    -------
    str
        Candidate name extended with the prior regime when more than one is fit.
    """
    if n_priors > 1:
        return f"{name}@sd={innovation_sd:g},ar={autoreg:g}"
    return name


def main() -> None:
    """Run the rt_params suite from the command line."""
    args = _parse_args()
    if args.quick:
        args.num_warmup = 50
        args.num_samples = 50
        args.num_chains = 1

    numpyro.set_host_device_count(args.num_chains)
    if not args.no_x64:
        numpyro.enable_x64()

    candidates = _resolve_candidates(args.candidate)
    priors = _resolve_priors(args.prior)
    settings = McmcSettings(
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        progress_bar=args.progress_bar,
    )

    print(
        f"rt_params suite: {len(candidates)} candidate(s) x "
        f"{len(priors)} prior(s) x {args.repeats} repeat(s) "
        f"= {len(candidates) * len(priors) * args.repeats} fits",
        flush=True,
    )

    results: list[FitResult] = []
    for innovation_sd, autoreg in priors:
        for name in candidates:
            if name.startswith("he_"):
                config = _parse_he_candidate(name, innovation_sd, autoreg)
            else:
                config = _parse_subpop_candidate(name, innovation_sd, autoreg)
            for repeat in range(args.repeats):
                label = _candidate_label(name, innovation_sd, autoreg, len(priors))
                print(
                    f">> fitting {label} (repeat {repeat + 1}/{args.repeats}) ...",
                    flush=True,
                )
                built = _build_for_candidate(name, config)
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
