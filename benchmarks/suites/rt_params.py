# ruff: noqa: E402

"""rt_params benchmark suite.

Compare the ``innovation`` (non-centered) and ``state`` (centered)
parameterizations of the weekly $\\mathcal{R}(t)$ temporal process on the H+E
model. The two parameterizations are the comparison arms; ``--prior`` steps a
prior regime that fixes the weekly innovation SD and autoregressive coefficient,
held equal across arms via the spec match keys so the parameterizations are
compared within a regime.

The model is the shared :func:`benchmarks.models.he.build_he_model`; this suite
only declares the arms (parameterization x prior regime) and the spec.

Run as a module from the repository root:

    python -m benchmarks.suites.rt_params --quick
    python -m benchmarks.suites.rt_params --prior both --repeats 3

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from benchmarks.core.env import configure_jax

configure_jax()

from benchmarks.core.comparison import DEFAULT_METRICS, ComparisonSpec
from benchmarks.core.signals import DatasetBundle
from benchmarks.core.suite import Arm, comparison_suite
from benchmarks.models.he import HEModelConfig, build_he_model

DEFAULT_TIGHT_SD = 0.01
DEFAULT_LOOSE_SD = 0.10
DEFAULT_TIGHT_AUTOREG = 0.9
DEFAULT_LOOSE_AUTOREG = 0.5
TIGHT_PRIOR: tuple[float, float] = (DEFAULT_TIGHT_SD, DEFAULT_TIGHT_AUTOREG)
LOOSE_PRIOR: tuple[float, float] = (DEFAULT_LOOSE_SD, DEFAULT_LOOSE_AUTOREG)
PARAMETERIZATIONS: tuple[str, ...] = ("innovation", "state")

SPEC: ComparisonSpec = ComparisonSpec(
    name="rt_params",
    arms=PARAMETERIZATIONS,
    baseline="innovation",
    match_keys=("dataset", "innovation_sd", "autoreg"),
    metrics=DEFAULT_METRICS,
)


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
    if sd <= 0:
        raise ValueError(f"Prior innovation sd must be positive; got {sd:g}")
    if not -1 < ar < 1:
        raise ValueError(f"Prior autoreg must satisfy -1 < autoreg < 1; got {ar:g}")
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


def _add_prior_arg(parser: argparse.ArgumentParser) -> None:
    """Register the ``--prior`` regime sweep argument."""
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


def _arms(args: argparse.Namespace, bundle: DatasetBundle) -> list[Arm]:
    """Build one arm per parameterization and prior regime.

    The two parameterizations are the arms; each prior pair is held equal across
    arms via the spec match keys. The day-of-week effect is fixed to the
    bundle's known weekday signal so it does not confound the comparison.

    Returns
    -------
    list[Arm]
        One arm per (prior, parameterization) combination.
    """
    priors = _resolve_priors(args.prior)
    arms: list[Arm] = []
    for innovation_sd, autoreg in priors:
        for parameterization in PARAMETERIZATIONS:
            arms.append(
                Arm(
                    name=parameterization,
                    label=_fit_label(
                        parameterization, innovation_sd, autoreg, len(priors)
                    ),
                    config=HEModelConfig(
                        rt=parameterization,
                        rt_innovation_sd=innovation_sd,
                        rt_autoreg=autoreg,
                        day_of_week="data",
                    ),
                    config_fields={
                        "parameterization": parameterization,
                        "innovation_sd": innovation_sd,
                        "autoreg": autoreg,
                    },
                )
            )
    return arms


main = comparison_suite(
    SPEC, _arms, build_he_model, description=__doc__, add_args=_add_prior_arg
)

if __name__ == "__main__":
    main()
