# ruff: noqa: E402

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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from benchmarks.core.env import configure_jax

configure_jax()

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import pyrenew.transformation as transformation
from benchmarks.core.cli import add_common_args, settings_from_args
from benchmarks.core.comparison import DEFAULT_METRICS, ComparisonSpec
from benchmarks.core.datasets import (
    SYNTHETIC_HE_WEEKLY_HOSPITAL,
    SyntheticProvider,
)
from benchmarks.core.models import BuiltFit, align_weekly_observations
from benchmarks.core.priors import (
    real_he_ed_day_of_week_prior,
    real_he_i0_prior,
)
from benchmarks.core.real_data import RealDataProvider, RealDataSpec
from benchmarks.core.reporting import print_data_summary
from benchmarks.core.run import run_comparison
from benchmarks.core.runner import Candidate
from benchmarks.core.signals import DatasetBundle
from pyrenew.ascertainment import AscertainmentModel, JointAscertainment
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    DifferencedAR1,
    PopulationInfections,
    WeeklyTemporalProcess,
)
from pyrenew.model import PyrenewBuilder
from pyrenew.observation import (
    NegativeBinomialNoise,
    PopulationCounts,
)
from pyrenew.randomvariable import (
    DistributionalVariable,
    TransformedVariable,
)
from pyrenew.time import MMWR_WEEK

COMPARISON_NAME = "rt_params"
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
HE_BUNDLE_KEY = "he"
Disease = str


PARAMETERIZATIONS: tuple[str, ...] = ("innovation", "state")

COMPARISON_SPEC: ComparisonSpec = ComparisonSpec(
    name=COMPARISON_NAME,
    arms=PARAMETERIZATIONS,
    baseline="innovation",
    match_keys=("dataset", "innovation_sd", "autoreg"),
    metrics=DEFAULT_METRICS,
)

Parameterization = Literal["innovation", "state"]


@dataclass(frozen=True)
class BuildConfig:
    """Configurable axes of an rt_params candidate.

    Parameters
    ----------
    parameterization
        ``"innovation"`` or ``"state"`` for the Rt temporal process.
    innovation_sd
        Per-step standard deviation of the weekly AR(1) on first differences
        of $\\log \\mathcal{R}(t)$.
    autoreg
        Autoregressive coefficient for the same process.
    """

    parameterization: Parameterization
    innovation_sd: float = 0.05
    autoreg: float = 0.9


def _build_rt_process(config: BuildConfig) -> WeeklyTemporalProcess:
    """Build the weekly Rt temporal process for the H+E model.

    ``config.innovation_sd`` is the per-step standard deviation of innovations
    to the rate of change in $\\log \\mathcal{R}(t)$ at weekly cadence.

    Returns
    -------
    WeeklyTemporalProcess
        Weekly differenced AR(1) Rt process.
    """
    inner = DifferencedAR1(
        autoreg_rv=DeterministicVariable("rt_diff_autoreg", config.autoreg),
        innovation_sd_rv=DeterministicVariable(
            "rt_diff_innovation_sd", config.innovation_sd
        ),
        parameterization=config.parameterization,
    )
    return WeeklyTemporalProcess(inner, start_dow=MMWR_WEEK)


def _build_he_ascertainment() -> AscertainmentModel:
    """Build the joint Gaussian H+E ascertainment model.

    Returns
    -------
    AscertainmentModel
        Joint Gaussian ascertainment over hospital and ED visit rates.
    """
    sd = 0.3
    corr = 0.5
    cov = jnp.array([[sd**2, corr * sd**2], [corr * sd**2, sd**2]])
    return JointAscertainment(
        name="he_ascertainment",
        signals=("hospital", "ed_visits"),
        baseline_rates=jnp.array([0.004, 0.004]),
        covariance_matrix=cov,
    )


def build_he_model(
    config: BuildConfig,
    bundle: DatasetBundle | None = None,
) -> BuiltFit:
    """Build the H+E PopulationInfections model and its run kwargs.

    By default, uses :data:`SYNTHETIC_HE_WEEKLY_HOSPITAL`: weekly-aggregated
    hospital reporting plus daily ED visits, matching the production-style
    H+E setup. Callers may pass a bundle from another provider. The latent
    $\\mathcal{R}(t)$ process runs at weekly cadence.

    Returns
    -------
    BuiltFit
        Model and run kwargs ready for fitting.
    """
    if bundle is None:
        bundle = SyntheticProvider().get(SYNTHETIC_HE_WEEKLY_HOSPITAL)
    hospital_signal = bundle.signals["hospital"]
    ed_signal = bundle.signals["ed_visits"]
    if "i0_per_capita" in bundle.fixed_params:
        i0_per_capita = float(bundle.fixed_params["i0_per_capita"])
        i0_rv = TransformedVariable(
            name="I0",
            base_rv=DistributionalVariable(
                name="logit_I0",
                distribution=dist.Normal(
                    transformation.SigmoidTransform().inv(i0_per_capita),
                    0.25,
                ),
            ),
            transforms=transformation.SigmoidTransform(),
        )
    else:
        i0_rv = real_he_i0_prior()
    ed_right_truncation_rv = None
    if "right_truncation_pmf" in bundle.fixed_params:
        ed_right_truncation_rv = DeterministicPMF(
            "ed_right_truncation",
            bundle.fixed_params["right_truncation_pmf"],
        )
    ascertainment = _build_he_ascertainment()

    builder = PyrenewBuilder()
    builder.configure_latent(
        PopulationInfections,
        gen_int_rv=DeterministicPMF("gen_int", bundle.gen_int_pmf),
        I0_rv=i0_rv,
        log_rt_time_0_rv=DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5)),
        single_rt_process=_build_rt_process(config),
    )
    builder.add_ascertainment(ascertainment)

    hospital_kwargs: dict[str, object] = {}
    if hospital_signal.cadence == "weekly":
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
    else:
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
            )
        )
    builder.add_observation(
        PopulationCounts(
            name="ed_visits",
            ascertainment_rate_rv=ascertainment.for_signal("ed_visits"),
            delay_distribution_rv=DeterministicPMF(
                "ed_delay", ed_signal.extras["delay_pmf"]
            ),
            right_truncation_rv=ed_right_truncation_rv,
            noise=NegativeBinomialNoise(
                DistributionalVariable("ed_conc", dist.LogNormal(4.0, 1.0))
            ),
            day_of_week_rv=(
                DeterministicVariable(
                    "ed_day_of_week_effect",
                    ed_signal.extras["day_of_week_effects"],
                )
                if "day_of_week_effects" in ed_signal.extras
                else real_he_ed_day_of_week_prior()
            ),
        )
    )
    model = builder.build()

    if hospital_signal.cadence == "weekly":
        hospital_obs = align_weekly_observations(
            model,
            "hospital",
            hospital_signal.values,
            bundle.obs_start_date,
            bundle.n_days_post_init,
        )
    else:
        hospital_obs = model.pad_observations(hospital_signal.values)
    ed_obs = model.pad_observations(ed_signal.values)
    hospital_kwargs["obs"] = hospital_obs
    ed_kwargs: dict[str, object] = {"obs": ed_obs}
    if "right_truncation_offset" in bundle.fixed_params:
        ed_kwargs["right_truncation_offset"] = bundle.fixed_params[
            "right_truncation_offset"
        ]
    return BuiltFit(
        model=model,
        run_kwargs={
            "n_days_post_init": bundle.n_days_post_init,
            "population_size": bundle.population_size,
            "obs_start_date": bundle.obs_start_date,
            "hospital": hospital_kwargs,
            "ed_visits": ed_kwargs,
        },
        dataset_name=bundle.name,
    )


def _load_bundles(args: argparse.Namespace) -> dict[str, DatasetBundle]:
    """Load the H+E dataset bundle for the suite.

    Returns
    -------
    dict[str, DatasetBundle]
        Loaded bundle keyed by dataset identifier.
    """
    bundles: dict[str, DatasetBundle] = {}
    if args.data_source == "synthetic":
        bundles[HE_BUNDLE_KEY] = SyntheticProvider().get(SYNTHETIC_HE_WEEKLY_HOSPITAL)
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
    bundles[HE_BUNDLE_KEY] = provider.get(REAL_HE_DATASET)
    return bundles


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
    add_common_args(parser)
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


def build_candidates(
    bundle: DatasetBundle, priors: Sequence[tuple[float, float]]
) -> list[Candidate]:
    """Build one candidate per parameterization and prior pair.

    The two parameterizations are the comparison arms. Each prior pair is held
    equal across arms via the spec's match keys, so the parameterizations are
    compared within a prior rather than across priors. Each candidate's
    ``build`` closes over its :class:`BuildConfig`, so the runner assembles a
    fresh model for every repeat.

    Returns
    -------
    list[Candidate]
        One candidate per (prior, parameterization) combination.
    """
    candidates: list[Candidate] = []
    for innovation_sd, autoreg in priors:
        for parameterization in PARAMETERIZATIONS:
            config = BuildConfig(
                parameterization=parameterization,
                innovation_sd=innovation_sd,
                autoreg=autoreg,
            )
            candidates.append(
                Candidate(
                    name=_fit_label(
                        parameterization, innovation_sd, autoreg, len(priors)
                    ),
                    arm=parameterization,
                    config_fields={
                        "parameterization": parameterization,
                        "innovation_sd": innovation_sd,
                        "autoreg": autoreg,
                    },
                    build=lambda config=config: build_he_model(config, bundle),
                )
            )
    return candidates


def main() -> None:
    """Run the rt_params suite from the command line."""
    args = _parse_args()
    settings = settings_from_args(args)
    numpyro.set_host_device_count(settings.num_chains)
    numpyro.enable_x64()

    try:
        priors = _resolve_priors(args.prior)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    try:
        bundles = _load_bundles(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.dry_run_data:
        print_data_summary(bundles.values())
        return

    bundle = bundles[HE_BUNDLE_KEY]
    run_comparison(
        build_candidates(bundle, priors),
        COMPARISON_SPEC,
        settings,
        comparison_name=COMPARISON_NAME,
        repeats=args.repeats,
        output_dir=None if args.no_write else args.output_dir,
    )


if __name__ == "__main__":
    main()
