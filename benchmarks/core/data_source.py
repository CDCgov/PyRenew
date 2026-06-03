"""Shared data-source selection for benchmark drivers.

Every H+E suite chooses between built-in synthetic fixtures and live CDC
NHSN/NSSP feeds with the same flags and the same loader. :func:`add_data_source_args`
registers the flags, :func:`validate_data_source_args` enforces their
constraints, and :func:`load_he_bundle` returns the selected
:class:`~benchmarks.core.signals.DatasetBundle`. Suites add these to their own
parser alongside :func:`benchmarks.core.cli.add_common_args` so real-data mode
is a property of the core machinery, not of any one suite.

The ``cfa.stf.*`` imports needed for real data live inside
:mod:`benchmarks.core.real_data`; importing this module does not require
``cfa-stf-routine-forecasting``.
"""

from __future__ import annotations

import argparse
import datetime as dt

from benchmarks.core.datasets import (
    SYNTHETIC_HE_WEEKLY_HOSPITAL,
    SyntheticProvider,
)
from benchmarks.core.real_data import RealDataProvider, RealDataSpec
from benchmarks.core.signals import DatasetBundle

DEFAULT_REAL_DISEASE = "COVID-19"
DEFAULT_REAL_LOCATION = "US"
DEFAULT_REAL_TRAINING_DAYS = 150
DEFAULT_REAL_OMIT_DAYS = 2
REAL_HE_DATASET = "real_he"
HE_BUNDLE_KEY = "he"


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


def add_data_source_args(parser: argparse.ArgumentParser) -> None:
    """Register the synthetic / real-data selection arguments.

    Parameters
    ----------
    parser
        Parser to add the data-source arguments to.
    """
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


def validate_data_source_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Enforce constraints on the parsed data-source arguments.

    Parameters
    ----------
    parser
        Parser used to emit usage errors.
    args
        Parsed namespace carrying the data-source arguments.
    """
    if args.data_source == "real" and args.as_of is None:
        parser.error("--as-of is required when --data-source real")
    if args.training_days <= 0:
        parser.error("--training-days must be positive")
    if args.omit_last_days < 0:
        parser.error("--omit-last-days must be non-negative")


def load_he_bundle(args: argparse.Namespace) -> DatasetBundle:
    """Load the H+E dataset bundle selected by ``args.data_source``.

    Synthetic mode returns the built-in weekly-hospital H+E fixture. Real mode
    builds a :class:`~benchmarks.core.real_data.RealDataSpec` from the disease,
    location, vintage, and window flags and pulls live CDC feeds.

    Returns
    -------
    DatasetBundle
        The H+E bundle for the selected data source.
    """
    if args.data_source == "synthetic":
        return SyntheticProvider().get(SYNTHETIC_HE_WEEKLY_HOSPITAL)

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
    return provider.get(REAL_HE_DATASET)
