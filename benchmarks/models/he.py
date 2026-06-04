"""Hospital + ED-visits (H+E) PyRenew model builder.

``build_he_model`` reads ``bundle.signals["hospital"]`` (weekly) and
``bundle.signals["ed_visits"]`` (daily) and builds a ``PopulationInfections``
renewal process with a weekly $\\mathcal{R}(t)$ temporal process and a joint
Gaussian H+E ascertainment.
:class:`HEModelConfig` carries the structural axes: the Rt parameterization and
its fixed hyperparameters, the ED day-of-week treatment, and the ascertainment
spread. Generation interval, initial prevalence, delays, right truncation, and
signal cadence come from the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpyro.distributions as dist

import pyrenew.transformation as transformation
from benchmarks.core.datasets import SYNTHETIC_HE_WEEKLY_HOSPITAL, SyntheticProvider
from benchmarks.core.models import BuiltFit, align_weekly_observations
from benchmarks.core.priors import real_he_ed_day_of_week_prior, real_he_i0_prior
from benchmarks.core.signals import DatasetBundle, SignalSeries
from benchmarks.core.suite import Arm
from pyrenew.ascertainment import AscertainmentModel, JointAscertainment
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    DifferencedAR1,
    PopulationInfections,
    WeeklyTemporalProcess,
)
from pyrenew.metaclass import RandomVariable
from pyrenew.model import PyrenewBuilder
from pyrenew.observation import NegativeBinomialNoise, PopulationCounts
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable
from pyrenew.time import MMWR_WEEK

DayOfWeek = Literal["none", "infer", "data"]
Parameterization = Literal["innovation", "state"]


@dataclass(frozen=True)
class HEModelConfig:
    """Structural axes of an H+E candidate.

    Parameters
    ----------
    rt
        Parameterization of the inner weekly ``DifferencedAR1``: ``innovation``
        (non-centered) or ``state`` (centered).
    rt_autoreg
        Fixed autoregressive coefficient of the weekly Rt process.
    rt_innovation_sd
        Fixed per-step innovation standard deviation of the weekly Rt process.
    day_of_week
        ED day-of-week treatment: ``none`` omits the weekday multiplier,
        ``infer`` samples it from the day-of-week prior, and ``data`` fixes it
        to the bundle's ``day_of_week_effects`` extra when present (falling back
        to the prior otherwise).
    ascertainment_sd
        Standard deviation of each marginal in the joint Gaussian ascertainment.
    ascertainment_corr
        Correlation between the hospital and ED ascertainment rates.
    """

    rt: Parameterization = "state"
    rt_autoreg: float = 0.9
    rt_innovation_sd: float = 0.05
    day_of_week: DayOfWeek = "infer"
    ascertainment_sd: float = 0.3
    ascertainment_corr: float = 0.5


def _report_fields(config: HEModelConfig) -> dict[str, Any]:
    """Build the reporting fields for an H+E candidate.

    Returns
    -------
    dict[str, Any]
        Flat fields used to label and group the candidate in reports.
    """
    return {
        "model_family": "pyrenew",
        "rt": f"weekly-{config.rt}",
        "ascertainment": "joint",
        "ed_day_of_week": config.day_of_week != "none",
    }


def he_arm(name: str, config: HEModelConfig, *, label: str | None = None) -> Arm:
    """Build a comparison :class:`Arm` for an H+E model config.

    Returns
    -------
    Arm
        Arm carrying ``config`` and curated H+E report fields.
    """
    return Arm(
        name=name, config=config, config_fields=_report_fields(config), label=label
    )


def _build_rt_process(config: HEModelConfig) -> WeeklyTemporalProcess:
    """Build the weekly $\\mathcal{R}(t)$ process for the given config.

    Returns
    -------
    WeeklyTemporalProcess
        Weekly differenced AR(1) Rt process in the configured parameterization.
    """
    inner = DifferencedAR1(
        autoreg_rv=DeterministicVariable("rt_diff_autoreg", config.rt_autoreg),
        innovation_sd_rv=DeterministicVariable(
            "rt_diff_innovation_sd", config.rt_innovation_sd
        ),
        parameterization=config.rt,
    )
    return WeeklyTemporalProcess(inner, start_dow=MMWR_WEEK)


def _build_ascertainment(config: HEModelConfig) -> AscertainmentModel:
    """Build the joint Gaussian H+E ascertainment model.

    Returns
    -------
    AscertainmentModel
        Joint Gaussian ascertainment over hospital and ED visit rates.
    """
    sd = config.ascertainment_sd
    corr = config.ascertainment_corr
    cov = jnp.array([[sd**2, corr * sd**2], [corr * sd**2, sd**2]])
    return JointAscertainment(
        name="he_ascertainment",
        signals=("hospital", "ed_visits"),
        baseline_rates=jnp.array([0.004, 0.004]),
        covariance_matrix=cov,
    )


def _ed_day_of_week_rv(
    mode: DayOfWeek, ed_signal: SignalSeries
) -> RandomVariable | None:
    """Resolve the ED day-of-week random variable for the given mode.

    Returns
    -------
    RandomVariable | None
        ``None`` when omitted, a deterministic effect when ``data`` and the
        bundle supplies one, otherwise the inferred day-of-week prior.
    """
    if mode == "none":
        return None
    if mode == "data" and "day_of_week_effects" in ed_signal.extras:
        return DeterministicVariable(
            "ed_day_of_week_effect", ed_signal.extras["day_of_week_effects"]
        )
    return real_he_ed_day_of_week_prior()


def _i0_rv(bundle: DatasetBundle) -> RandomVariable:
    """Build the initial-prevalence random variable from the bundle.

    Returns
    -------
    RandomVariable
        A tight prior around the bundle's fixed per-capita prevalence when
        provided, otherwise the real-data initial-prevalence prior.
    """
    if "i0_per_capita" in bundle.fixed_params:
        i0_per_capita = float(bundle.fixed_params["i0_per_capita"])
        return TransformedVariable(
            name="I0",
            base_rv=DistributionalVariable(
                name="logit_I0",
                distribution=dist.Normal(
                    transformation.SigmoidTransform().inv(i0_per_capita), 0.25
                ),
            ),
            transforms=transformation.SigmoidTransform(),
        )
    return real_he_i0_prior()


def build_he_model(
    config: HEModelConfig, bundle: DatasetBundle | None = None
) -> BuiltFit:
    """Build the H+E model and its run kwargs for one config.

    Weekly-aggregated hospital admissions plus daily ED visits, a joint Gaussian
    ascertainment, and a weekly $\\mathcal{R}(t)$ process. The bundle drives the
    generation interval, initial prevalence, delays, right truncation, and the
    hospital signal cadence; the config drives the structural axes.

    Returns
    -------
    BuiltFit
        Model and run kwargs ready for fitting.
    """
    if bundle is None:
        bundle = SyntheticProvider().get(SYNTHETIC_HE_WEEKLY_HOSPITAL)
    hospital_signal = bundle.signals["hospital"]
    ed_signal = bundle.signals["ed_visits"]

    ed_right_truncation_rv = None
    if "right_truncation_pmf" in bundle.fixed_params:
        ed_right_truncation_rv = DeterministicPMF(
            "ed_right_truncation", bundle.fixed_params["right_truncation_pmf"]
        )
    ascertainment = _build_ascertainment(config)

    builder = PyrenewBuilder()
    builder.configure_latent(
        PopulationInfections,
        gen_int_rv=DeterministicPMF("gen_int", bundle.gen_int_pmf),
        I0_rv=_i0_rv(bundle),
        log_rt_time_0_rv=DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5)),
        single_rt_process=_build_rt_process(config),
    )
    builder.add_ascertainment(ascertainment)

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
            day_of_week_rv=_ed_day_of_week_rv(config.day_of_week, ed_signal),
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
            "hospital": {"obs": hospital_obs},
            "ed_visits": ed_kwargs,
        },
        dataset_name=bundle.name,
    )
