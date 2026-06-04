# ruff: noqa: E402

"""pyrenew_vs_hew benchmark suite.

Compare the production ``pyrenew-multisignal`` HEW model against a PyRenew
``MultiSignalModel`` on the same H+E data. The PyRenew candidate uses
weekly-aggregated hospital admissions plus daily ED visits, a joint Gaussian
ascertainment, a weekly state-centered $\\mathcal{R}(t)$ process, and no ED
day-of-week effect; the production HEW model retains its ED weekday effect.

Both arms fit either the built-in synthetic H+E fixture or live CDC NHSN/NSSP
feeds, selected with the shared ``--data-source`` flags. On real data the HEW
arm's model directory is materialized from the same bundle the PyRenew arm
consumes.

The HEW candidate requires ``pyrenew-multisignal`` and
``cfa-stf-routine-forecasting`` to be importable; pass their checkout paths
with ``--pyrenew-multisignal-dir`` and ``--cfa-stf-dir``.

Run from the repository root:

    python -m benchmarks.suites.pyrenew_vs_hew --quick
    python -m benchmarks.suites.pyrenew_vs_hew --data-source real \\
        --disease COVID-19 --location US --as-of 2025-01-15 --dry-run-data

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.core.env import configure_jax

configure_jax()

from benchmarks.core.comparison import DEFAULT_METRICS, ComparisonSpec
from benchmarks.core.models import BuiltFit
from benchmarks.core.signals import DatasetBundle
from benchmarks.core.suite import Arm, comparison_suite
from benchmarks.models.he import HEModelConfig, build_he_model, he_arm
from benchmarks.models.hew import (
    DEFAULT_CFA_STF_DIR,
    DEFAULT_PYRENEW_MULTISIGNAL_DIR,
    HEW_RT_SITE_NAMES,
    build_hew_model,
    write_hew_model_dir_from_bundle,
)
from pyrenew.datasets import write_synthetic_hew_model_dir

HEW_ARM = "hew"
PYRENEW_ARM = "pyrenew-state"

SPEC: ComparisonSpec = ComparisonSpec(
    name="pyrenew_vs_hew",
    arms=(HEW_ARM, PYRENEW_ARM),
    baseline=HEW_ARM,
    match_keys=("dataset",),
    metrics=DEFAULT_METRICS,
)


def _add_args(parser: argparse.ArgumentParser) -> None:
    """Register the HEW-candidate checkout paths and model directory."""
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
        default=Path("benchmarks/results/hew_model"),
        help=(
            "Directory to write the HEW arm's model inputs into. Holds whichever "
            "data source is selected (synthetic or real), not synthetic only."
        ),
    )


def _hew_arm(args: argparse.Namespace) -> Arm:
    """Build the production HEW arm.

    The HEW model loads from a directory, so its build callable first
    materializes the model inputs from the bundle (real data) or the synthetic
    fixture, then assembles the model from that directory.

    Returns
    -------
    Arm
        The HEW comparison arm.
    """

    def build(bundle: DatasetBundle) -> BuiltFit:
        """Write the HEW model inputs and assemble the production HEW model.

        Returns
        -------
        BuiltFit
            Assembled HEW model and run kwargs.
        """
        if args.data_source == "real":
            write_hew_model_dir_from_bundle(
                bundle,
                args.model_dir,
                location=args.location,
                disease=args.disease,
                overwrite=True,
            )
        else:
            write_synthetic_hew_model_dir(args.model_dir, overwrite=True)
        return build_hew_model(
            args.model_dir,
            dataset_name=bundle.name,
            pyrenew_multisignal_dir=args.pyrenew_multisignal_dir,
            cfa_stf_dir=args.cfa_stf_dir,
        )

    return Arm(
        name=HEW_ARM,
        config_fields={
            "model_family": "hew",
            "ascertainment": "linked",
            "rt": "weekly-state",
            "ed_day_of_week": True,
        },
        build=build,
        rt_site_names=HEW_RT_SITE_NAMES,
    )


def _arms(args: argparse.Namespace, bundle: DatasetBundle) -> list[Arm]:
    """Assemble the HEW and PyRenew arms, HEW (baseline) first.

    Returns
    -------
    list[Arm]
        The two comparison arms.
    """
    return [
        _hew_arm(args),
        he_arm(PYRENEW_ARM, HEModelConfig(rt="state", day_of_week="none")),
    ]


main = comparison_suite(
    SPEC, _arms, build_he_model, description=__doc__, add_args=_add_args
)

if __name__ == "__main__":
    main()
