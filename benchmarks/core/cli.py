"""Shared command-line scaffolding for benchmark drivers.

Every driver exposes the same sampler and output controls. :func:`add_common_args`
registers them on a parser, and :func:`settings_from_args` turns the parsed
namespace into an :class:`benchmarks.core.runner.McmcSettings`, applying the
``--quick`` smoke-run override. Drivers add their own suite-specific arguments
to the same parser before parsing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.core.env import DEFAULT_NUM_CHAINS
from benchmarks.core.runner import McmcSettings

DEFAULT_OUTPUT_DIR = Path("benchmarks/results")
QUICK_NUM_WARMUP = 50
QUICK_NUM_SAMPLES = 50
QUICK_NUM_CHAINS = 1


def add_common_args(
    parser: argparse.ArgumentParser, default_output_dir: Path = DEFAULT_OUTPUT_DIR
) -> None:
    """Register the sampler and output arguments shared by all drivers.

    Parameters
    ----------
    parser
        Parser to add the common arguments to.
    default_output_dir
        Default for ``--output-dir`` when the driver writes to a non-standard
        location.
    """
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-chains", type=int, default=DEFAULT_NUM_CHAINS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
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


def settings_from_args(args: argparse.Namespace) -> McmcSettings:
    """Build :class:`McmcSettings` from parsed common arguments.

    When ``args.quick`` is set, warmup, samples, and chains are replaced with
    the smoke-run values; the seed and progress-bar flag are taken as parsed.

    Returns
    -------
    McmcSettings
        Sampler configuration for the run.
    """
    if args.quick:
        num_warmup = QUICK_NUM_WARMUP
        num_samples = QUICK_NUM_SAMPLES
        num_chains = QUICK_NUM_CHAINS
    else:
        num_warmup = args.num_warmup
        num_samples = args.num_samples
        num_chains = args.num_chains
    return McmcSettings(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=args.seed,
        progress_bar=args.progress_bar,
    )
