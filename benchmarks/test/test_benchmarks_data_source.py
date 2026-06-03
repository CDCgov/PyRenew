"""Tests for the shared data-source selection in ``benchmarks.core.data_source``."""

import argparse
from datetime import date

import pytest

from benchmarks.core import data_source
from benchmarks.core.data_source import (
    REAL_HE_DATASET,
    add_data_source_args,
    load_he_bundle,
    validate_data_source_args,
)
from benchmarks.core.datasets import SYNTHETIC_HE_WEEKLY_HOSPITAL


def _parser() -> argparse.ArgumentParser:
    """Build a parser with the data-source arguments registered.

    Returns
    -------
    argparse.ArgumentParser
        Parser exposing only the data-source arguments.
    """
    parser = argparse.ArgumentParser()
    add_data_source_args(parser)
    return parser


def test_defaults_are_synthetic():
    """With no flags, the parser selects synthetic data."""
    args = _parser().parse_args([])
    assert args.data_source == "synthetic"
    assert args.dry_run_data is False


def test_real_options_parse():
    """Real-data flags parse into the expected namespace values."""
    args = _parser().parse_args(
        [
            "--data-source",
            "real",
            "--disease",
            "RSV",
            "--location",
            "CA",
            "--as-of",
            "2025-01-15",
            "--training-days",
            "120",
            "--omit-last-days",
            "3",
        ]
    )
    assert args.data_source == "real"
    assert args.disease == "RSV"
    assert args.location == "CA"
    assert args.as_of == date(2025, 1, 15)
    assert args.training_days == 120
    assert args.omit_last_days == 3


def test_validate_requires_as_of_for_real():
    """Real mode without --as-of is a usage error."""
    parser = _parser()
    args = parser.parse_args(["--data-source", "real"])
    with pytest.raises(SystemExit):
        validate_data_source_args(parser, args)


@pytest.mark.parametrize(
    "flags",
    [
        ["--training-days", "0"],
        ["--omit-last-days", "-1"],
    ],
)
def test_validate_rejects_bad_window(flags):
    """Non-positive training windows and negative omit counts are rejected."""
    parser = _parser()
    args = parser.parse_args(flags)
    with pytest.raises(SystemExit):
        validate_data_source_args(parser, args)


def test_load_he_bundle_synthetic():
    """Synthetic mode returns the built-in weekly-hospital H+E bundle."""
    args = _parser().parse_args([])
    bundle = load_he_bundle(args)
    assert bundle.name == SYNTHETIC_HE_WEEKLY_HOSPITAL
    assert "ed_visits" in bundle.signals
    assert "hospital" in bundle.signals


def test_load_he_bundle_real_builds_spec(monkeypatch):
    """Real mode builds a RealDataSpec from the flags and pulls one bundle."""
    sentinel = object()
    captured_specs: dict = {}

    class FakeRealDataProvider:
        """Provider stub capturing the requested real-data specs."""

        def __init__(self, specs):
            """Store the provided specs for assertion."""
            captured_specs.update(specs)

        def get(self, name):
            """Return the sentinel bundle for the expected dataset name.

            Returns
            -------
            object
                Sentinel bundle supplied by the test.
            """
            assert name == REAL_HE_DATASET
            return sentinel

    monkeypatch.setattr(data_source, "RealDataProvider", FakeRealDataProvider)
    args = _parser().parse_args(
        [
            "--data-source",
            "real",
            "--disease",
            "Influenza",
            "--location",
            "US",
            "--as-of",
            "2025-02-01",
            "--training-days",
            "90",
            "--omit-last-days",
            "2",
        ]
    )

    bundle = load_he_bundle(args)

    assert bundle is sentinel
    spec = captured_specs[REAL_HE_DATASET]
    assert spec.disease == "Influenza"
    assert spec.loc_abbr == "US"
    assert spec.as_of == date(2025, 2, 1)
    assert spec.n_training_days == 90
    assert spec.n_days_to_omit == 2
    assert spec.signals == ("hospital", "ed_visits")
