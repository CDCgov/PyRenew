"""Model builders for benchmark suites.

Each ``build_*`` function takes a :class:`DatasetBundle` and a ``BuildConfig``
and returns a :class:`BuiltFit`, which carries the assembled
:class:`MultiSignalModel` together with the keyword arguments needed by
``model.run``.

The mapping from a benchmark candidate to a dataset is implicit: each model
builder calls one specific dataset name on the provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import jax.numpy as jnp
import numpyro.distributions as dist

import pyrenew.transformation as transformation
from benchmarks.core.datasets import (
    SYNTHETIC_HE_WEEKLY_HOSPITAL,
    SyntheticProvider,
)
from benchmarks.core.priors import real_he_ed_day_of_week_prior, real_he_i0_prior
from benchmarks.core.signals import DatasetBundle
from pyrenew.ascertainment import AscertainmentModel, JointAscertainment
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    DifferencedAR1,
    PopulationInfections,
    WeeklyTemporalProcess,
)
from pyrenew.model import MultiSignalModel, PyrenewBuilder
from pyrenew.observation import (
    NegativeBinomialNoise,
    PopulationCounts,
)
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable
from pyrenew.time import MMWR_WEEK

Parameterization = Literal["innovation", "state"]


@dataclass(frozen=True)
class BuildConfig:
    """Configurable axes of a benchmark candidate.

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


@dataclass
class BuiltFit:
    """Assembled model plus the kwargs that ``model.run`` needs.

    Parameters
    ----------
    model
        The compiled :class:`MultiSignalModel`.
    run_kwargs
        Mapping passed as ``**kwargs`` to ``model.run`` after the MCMC
        controls. Already includes ``n_days_post_init``, ``population_size``,
        ``obs_start_date`` and the per-signal observation dicts.
    dataset_name
        Identifier of the dataset bundle used.
    n_initialization_points
        Latent initialization points the model requires.
    """

    model: MultiSignalModel
    run_kwargs: dict[str, Any]
    dataset_name: str
    n_initialization_points: int = field(init=False)

    def __post_init__(self) -> None:
        """Cache ``n_initialization_points`` for reporting."""
        self.n_initialization_points = self.model.latent.n_initialization_points


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


def _align_weekly_observations(
    model: MultiSignalModel,
    signal_name: str,
    weekly_values: jnp.ndarray,
    obs_start_date: date,
    n_days_post_init: int,
) -> jnp.ndarray:
    """Pad a weekly observation series with leading NaNs to match the period grid.

    Returns
    -------
    jnp.ndarray
        Dense weekly observations aligned to the model's period grid.
    """
    obs = model.observations[signal_name]
    first_day_dow = model._resolve_first_day_dow(obs_start_date)
    n_total = model.latent.n_initialization_points + n_days_post_init
    offset = obs._compute_period_offset(first_day_dow, obs.start_dow)
    n_periods = (n_total - offset) // obs.aggregation_period
    n_pre = n_periods - len(weekly_values)
    if n_pre < 0:
        raise ValueError(
            f"Weekly observations for {signal_name!r} are longer than the "
            f"model period grid: {len(weekly_values)} > {n_periods}."
        )
    return jnp.concatenate([jnp.full(n_pre, jnp.nan, dtype=jnp.float32), weekly_values])


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

    hospital_kwargs: dict[str, Any] = {}
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
        hospital_obs = _align_weekly_observations(
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
    ed_kwargs: dict[str, Any] = {"obs": ed_obs}
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
