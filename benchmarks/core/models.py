"""Model builders for benchmark suites.

Each ``build_*`` function takes a :class:`DatasetBundle` and a ``BuildConfig``
and returns a :class:`BuiltFit`, which carries the assembled
:class:`MultiSignalModel` together with the keyword arguments needed by
``model.run``.

The mapping from a benchmark candidate to a dataset is implicit: each model
builder calls one specific dataset name on the provider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike

import pyrenew.transformation as transformation
from benchmarks.core.datasets import (
    SUBPOP_HOSPITAL_WASTEWATER_CA,
    SYNTHETIC_HE_WEEKLY_HOSPITAL,
    SyntheticProvider,
)
from benchmarks.core.priors import real_he_ed_day_of_week_prior, real_he_i0_prior
from benchmarks.core.signals import DatasetBundle
from pyrenew.ascertainment import AscertainmentModel, JointAscertainment
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    DifferencedAR1,
    GammaGroupSdPrior,
    HierarchicalNormalPrior,
    PopulationInfections,
    RandomWalk,
    SubpopulationInfections,
    WeeklyTemporalProcess,
)
from pyrenew.metaclass import RandomVariable
from pyrenew.model import MultiSignalModel, PyrenewBuilder
from pyrenew.observation import (
    HierarchicalNormalNoise,
    MeasurementNoise,
    MeasurementObservation,
    NegativeBinomialNoise,
    PopulationCounts,
)
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable
from pyrenew.time import MMWR_WEEK

Parameterization = Literal["innovation", "state"]
Cadence = Literal["daily", "weekly"]


@dataclass(frozen=True)
class BuildConfig:
    """Configurable axes of a benchmark candidate.

    Parameters
    ----------
    parameterization
        ``"innovation"`` or ``"state"`` for the Rt temporal process.
    rt_cadence
        ``"daily"`` or ``"weekly"`` for the H+E model. Subpopulation models
        always use daily Rt; the field is ignored for them.
    innovation_sd
        Daily-equivalent innovation standard deviation for the AR(1) on first
        differences of log-Rt. When ``rt_cadence == "weekly"``, the per-step
        SD is rescaled to $\\sigma \\sqrt{7}$ so that the implied cumulative
        variance of $\\log \\mathcal{R}(T)$ matches the daily configuration
        at the same horizon.
    autoreg
        Autoregressive coefficient for the same process. Passed through
        unchanged across cadences; see :func:`_build_rt_process`.
    """

    parameterization: Parameterization
    rt_cadence: Cadence = "daily"
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


class Wastewater(MeasurementObservation):
    """Wastewater viral concentration observation process."""

    def __init__(
        self,
        name: str,
        shedding_kinetics_rv: RandomVariable,
        log10_genome_per_infection_rv: RandomVariable,
        ml_per_person_per_day: float,
        noise: MeasurementNoise,
    ) -> None:
        """Initialize wastewater observation process.

        Parameters
        ----------
        name
            Unique observation name.
        shedding_kinetics_rv
            Viral shedding delay PMF.
        log10_genome_per_infection_rv
            Log10 genome copies shed per infection.
        ml_per_person_per_day
            Wastewater volume scaling.
        noise
            Continuous measurement noise model.
        """
        super().__init__(name=name, temporal_pmf_rv=shedding_kinetics_rv, noise=noise)
        self.log10_genome_per_infection_rv = log10_genome_per_infection_rv
        self.ml_per_person_per_day = ml_per_person_per_day

    def _predicted_obs(self, infections: ArrayLike) -> ArrayLike:
        """Transform subpopulation infections into log wastewater concentrations.

        Returns
        -------
        ArrayLike
            Predicted log concentrations with shape ``(time, subpop)``.
        """
        shedding_pmf = self.temporal_pmf_rv()
        log10_genome = self.log10_genome_per_infection_rv()

        def convolve_site(site_infections: ArrayLike) -> ArrayLike:
            """Convolve one subpopulation trajectory with shedding kinetics.

            Returns
            -------
            ArrayLike
                Convolved per-site shedding signal.
            """
            convolved, _ = self._convolve_with_alignment(
                site_infections, shedding_pmf, p_observed=1.0
            )
            return convolved

        shedding_signal = jax.vmap(convolve_site, in_axes=1, out_axes=1)(infections)
        genome_copies = 10**log10_genome
        concentration = shedding_signal * genome_copies / self.ml_per_person_per_day
        return jnp.log(concentration)


def _build_rt_process(
    config: BuildConfig,
) -> DifferencedAR1 | WeeklyTemporalProcess:
    """Build the Rt temporal process for the H+E model.

    ``config.innovation_sd`` is interpreted as the daily-equivalent per-step SD
    of innovations to the rate of change in $\\log \\mathcal{R}(t)$. When the
    inner process runs at weekly cadence, the per-step SD is rescaled by
    $\\sqrt{7}$ so the implied cumulative variance of $\\log \\mathcal{R}(T)$
    matches the daily configuration at the same horizon. ``config.autoreg``
    is passed through unchanged; its cadence-dependent interpretation is a
    known limitation of this rescaling.

    Returns
    -------
    DifferencedAR1 | WeeklyTemporalProcess
        Daily or weekly differenced AR(1) Rt process.
    """
    inner_sd = config.innovation_sd
    if config.rt_cadence == "weekly":
        inner_sd = inner_sd * math.sqrt(7.0)
    rt_process: DifferencedAR1 | WeeklyTemporalProcess = DifferencedAR1(
        autoreg_rv=DeterministicVariable("rt_diff_autoreg", config.autoreg),
        innovation_sd_rv=DeterministicVariable("rt_diff_innovation_sd", inner_sd),
        parameterization=config.parameterization,
    )
    if config.rt_cadence == "weekly":
        rt_process = WeeklyTemporalProcess(rt_process, start_dow=MMWR_WEEK)
    return rt_process


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
    H+E setup. Callers may pass a bundle from another provider. In all cases,
    ``config.rt_cadence`` controls the Rt latent process cadence, not the
    hospital observation cadence.

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


def build_subpop_hospital_wastewater_model(
    config: BuildConfig,
    bundle: DatasetBundle | None = None,
) -> BuiltFit:
    """Build the hospital + wastewater subpopulation model.

    Returns
    -------
    BuiltFit
        Model and run kwargs ready for fitting.
    """
    if bundle is None:
        bundle = SyntheticProvider().get(SUBPOP_HOSPITAL_WASTEWATER_CA)
    hospital_signal = bundle.signals["hospital"]
    wastewater_signal = bundle.signals["wastewater"]

    baseline_rt_process = DifferencedAR1(
        autoreg_rv=DeterministicVariable("subpop_rt_diff_autoreg", config.autoreg),
        innovation_sd_rv=DeterministicVariable(
            "subpop_rt_diff_innovation_sd", config.innovation_sd
        ),
        parameterization=config.parameterization,
    )
    subpop_deviation_process = RandomWalk(
        innovation_sd_rv=DeterministicVariable("subpop_deviation_innovation_sd", 0.025),
        parameterization=config.parameterization,
    )

    builder = PyrenewBuilder()
    builder.configure_latent(
        SubpopulationInfections,
        gen_int_rv=DeterministicPMF("subpop_gen_int", bundle.gen_int_pmf),
        I0_rv=DistributionalVariable("I0", dist.Beta(1.0, 100.0)),
        log_rt_time_0_rv=DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5)),
        baseline_rt_process=baseline_rt_process,
        subpop_rt_deviation_process=subpop_deviation_process,
    )
    builder.add_observation(
        PopulationCounts(
            name="hospital",
            ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
            delay_distribution_rv=DeterministicPMF(
                "subpop_hosp_delay", hospital_signal.extras["delay_pmf"]
            ),
            noise=NegativeBinomialNoise(
                DeterministicVariable("subpop_hosp_concentration", 10.0)
            ),
        )
    )
    builder.add_observation(
        Wastewater(
            name="wastewater",
            shedding_kinetics_rv=DeterministicPMF(
                "shedding_kinetics", wastewater_signal.extras["shedding_pmf"]
            ),
            log10_genome_per_infection_rv=DeterministicVariable(
                "log10_genome_per_inf", 9.0
            ),
            ml_per_person_per_day=1000.0,
            noise=HierarchicalNormalNoise(
                HierarchicalNormalPrior(
                    "ww_site_mode",
                    sd_rv=DeterministicVariable("site_mode_sd", 0.5),
                ),
                GammaGroupSdPrior(
                    "ww_site_sd",
                    sd_mean_rv=DeterministicVariable("site_sd_mean", 0.3),
                    sd_concentration_rv=DeterministicVariable("site_sd_conc", 4.0),
                ),
            ),
        )
    )
    model = builder.build()
    return BuiltFit(
        model=model,
        run_kwargs={
            "n_days_post_init": bundle.n_days_post_init,
            "population_size": bundle.population_size,
            "obs_start_date": bundle.obs_start_date,
            "subpop_fractions": bundle.fixed_params["subpop_fractions"],
            "hospital": {
                "obs": model.pad_observations(hospital_signal.values),
            },
            "wastewater": {
                "obs": wastewater_signal.values,
                "times": model.shift_times(wastewater_signal.times),
                "subpop_indices": wastewater_signal.subpop_indices,
                "sensor_indices": wastewater_signal.sensor_indices,
                "n_sensors": wastewater_signal.extras["n_sensors"],
            },
        },
        dataset_name=bundle.name,
    )
