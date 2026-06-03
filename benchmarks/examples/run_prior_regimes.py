# ruff: noqa: E402

"""prior_regimes example driver.

Run one fixed H+E model structure under several prior choices ("regimes") and
compare how each regime samples. These models are weakly identified as usually
composed, so the priors are the lever that decides whether the sampler behaves.

This is a template. Copy it, change the model structure in :func:`_build_he_model`
and the regimes in :data:`REGIMES`, and run. The model structure and the regimes
are your research; the machinery (``run_comparison`` and ``benchmarks.core``) is
imported. Each regime is a self-contained function returning a bag of prior
factories, with the distributions written inline so the recorded source is the
complete prior specification.

Everything under ``benchmarks/examples/`` except the committed examples is
gitignored, so a copy you make here to run your own regimes stays out of version
control. See ``benchmarks/prior_regimes.md`` for the design. Run from the
repository root:

    python -m benchmarks.examples.run_prior_regimes --quick
"""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Callable

from benchmarks.core.env import configure_jax

configure_jax()

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from benchmarks.core.cli import add_common_args, settings_from_args
from benchmarks.core.comparison import DEFAULT_METRICS, ComparisonSpec
from benchmarks.core.datasets import SYNTHETIC_HE_WEEKLY_HOSPITAL, SyntheticProvider
from benchmarks.core.models import BuiltFit, align_weekly_observations
from benchmarks.core.run import run_comparison
from benchmarks.core.runner import Candidate
from benchmarks.core.signals import DatasetBundle
from pyrenew.ascertainment import JointAscertainment
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    DifferencedAR1,
    PopulationInfections,
    WeeklyTemporalProcess,
)
from pyrenew.model import PyrenewBuilder
from pyrenew.observation import NegativeBinomialNoise, PopulationCounts
from pyrenew.randomvariable import DistributionalVariable
from pyrenew.time import MMWR_WEEK

COMPARISON_NAME = "prior_regimes"
BASELINE_REGIME = "example"

PriorBag = dict[str, Callable[[], object]]

REQUIRED_SLOTS: frozenset[str] = frozenset(
    {
        "rt_diff_innovation_sd",
        "rt_diff_autoreg",
        "log_rt_time_0",
        "hosp_conc",
        "ed_conc",
        "I0",
        "he_ascertainment",
    }
)
"""Prior slots that :func:`_build_he_model` requires every regime to supply."""


def example_priors() -> PriorBag:
    """Starting-point priors for the H+E comparison.

    These are the values currently used elsewhere in the benchmark. They are a
    starting point, not authoritative: establishing good priors is the purpose
    of the comparison, so copy this function, change the distributions, and add
    your regime to :data:`REGIMES`.

    Returns
    -------
    PriorBag
        One prior factory per model slot.
    """
    return {
        "rt_diff_innovation_sd": lambda: DeterministicVariable(
            "rt_diff_innovation_sd", 0.01
        ),
        "rt_diff_autoreg": lambda: DeterministicVariable("rt_diff_autoreg", 0.9),
        "log_rt_time_0": lambda: DistributionalVariable(
            "log_rt_time_0", dist.Normal(0.0, 0.5)
        ),
        "hosp_conc": lambda: DistributionalVariable(
            "hosp_conc", dist.LogNormal(5.0, 1.0)
        ),
        "ed_conc": lambda: DistributionalVariable("ed_conc", dist.LogNormal(4.0, 1.0)),
        "I0": lambda: DistributionalVariable("I0", dist.Beta(1.0, 10.0)),
        "he_ascertainment": lambda: JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed_visits"),
            baseline_rates=jnp.array([0.004, 0.004]),
            covariance_matrix=jnp.array([[0.09, 0.045], [0.045, 0.09]]),
        ),
    }


# Add your prior regimes here. Each is a function shaped like `example_priors`,
# returning a factory for every slot in `REQUIRED_SLOTS`. A second regime turns
# the run from a single-model profile into a comparison. To vary only a few
# slots, spread a regime that is already in REGIMES and override the rest, e.g.
# `return {**example_priors(), "rt_diff_autoreg": ...}`; see prior_regimes.md.
REGIMES: dict[str, Callable[[], PriorBag]] = {
    "example": example_priors,
}

COMPARISON_SPEC: ComparisonSpec = ComparisonSpec(
    name=COMPARISON_NAME,
    arms=tuple(REGIMES),
    baseline=BASELINE_REGIME,
    match_keys=("dataset",),
    metrics=DEFAULT_METRICS,
)


def _build_he_model(bundle: DatasetBundle, priors: PriorBag) -> BuiltFit:
    """Assemble the fixed H+E structure, drawing every prior from ``priors``.

    Weekly-aggregated hospital admissions plus daily ED visits, joint
    ascertainment, and a weekly state-centered $\\mathcal{R}(t)$ process. No ED
    day-of-week effect (poorly identified). Delay PMFs and the generation
    interval come from the dataset, not the priors.

    Returns
    -------
    BuiltFit
        Model and run kwargs ready for the runner.
    """
    hospital_signal = bundle.signals["hospital"]
    ed_signal = bundle.signals["ed_visits"]
    ascertainment = priors["he_ascertainment"]()

    builder = PyrenewBuilder()
    builder.configure_latent(
        PopulationInfections,
        gen_int_rv=DeterministicPMF("gen_int", bundle.gen_int_pmf),
        I0_rv=priors["I0"](),
        log_rt_time_0_rv=priors["log_rt_time_0"](),
        single_rt_process=WeeklyTemporalProcess(
            DifferencedAR1(
                autoreg_rv=priors["rt_diff_autoreg"](),
                innovation_sd_rv=priors["rt_diff_innovation_sd"](),
                parameterization="state",
            ),
            start_dow=MMWR_WEEK,
        ),
    )
    builder.add_ascertainment(ascertainment)
    builder.add_observation(
        PopulationCounts(
            name="hospital",
            ascertainment_rate_rv=ascertainment.for_signal("hospital"),
            delay_distribution_rv=DeterministicPMF(
                "hosp_delay", hospital_signal.extras["delay_pmf"]
            ),
            noise=NegativeBinomialNoise(priors["hosp_conc"]()),
            aggregation="weekly",
            reporting_schedule="regular",
            start_dow=MMWR_WEEK,
        )
    )
    builder.add_observation(
        PopulationCounts(
            name="ed_visits",
            ascertainment_rate_rv=ascertainment.for_signal("ed_visits"),
            delay_distribution_rv=DeterministicPMF(
                "ed_delay", ed_signal.extras["delay_pmf"]
            ),
            noise=NegativeBinomialNoise(priors["ed_conc"]()),
        )
    )
    model = builder.build()

    hospital_obs = align_weekly_observations(
        model,
        "hospital",
        hospital_signal.values,
        bundle.obs_start_date,
        bundle.n_days_post_init,
    )
    ed_obs = model.pad_observations(ed_signal.values)
    return BuiltFit(
        model=model,
        run_kwargs={
            "n_days_post_init": bundle.n_days_post_init,
            "population_size": bundle.population_size,
            "obs_start_date": bundle.obs_start_date,
            "hospital": {"obs": hospital_obs},
            "ed_visits": {"obs": ed_obs},
        },
        dataset_name=bundle.name,
    )


def _validate_regimes() -> None:
    """Check every regime supplies the prior slots the structure consumes."""
    for name, regime_fn in REGIMES.items():
        missing = REQUIRED_SLOTS - set(regime_fn())
        if missing:
            raise ValueError(
                f"Regime {name!r} is missing prior slots: {sorted(missing)}"
            )


def prior_provenance() -> dict[str, dict[str, str]]:
    """Capture each regime's identity and source text for the results.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of regime name to its module and ``inspect.getsource`` text.
    """
    return {
        name: {
            "module": regime_fn.__module__,
            "source": inspect.getsource(regime_fn),
        }
        for name, regime_fn in REGIMES.items()
    }


def build_candidates(bundle: DatasetBundle) -> list[Candidate]:
    """Build one candidate per regime over the fixed model structure.

    Returns
    -------
    list[Candidate]
        One candidate per entry in :data:`REGIMES`.
    """
    return [
        Candidate(
            name=name,
            arm=name,
            config_fields={"prior_config": name},
            build=lambda regime_fn=regime_fn: _build_he_model(bundle, regime_fn()),
        )
        for name, regime_fn in REGIMES.items()
    ]


def _parse_args() -> argparse.Namespace:
    """Parse the prior_regimes CLI.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    """Run the prior_regimes comparison from the command line."""
    args = _parse_args()
    settings = settings_from_args(args)
    numpyro.set_host_device_count(settings.num_chains)
    numpyro.enable_x64()

    _validate_regimes()
    bundle = SyntheticProvider().get(SYNTHETIC_HE_WEEKLY_HOSPITAL)
    run_comparison(
        build_candidates(bundle),
        COMPARISON_SPEC,
        settings,
        comparison_name=COMPARISON_NAME,
        repeats=args.repeats,
        output_dir=None if args.no_write else args.output_dir,
        extra_payload={"prior_configs": prior_provenance()},
    )


if __name__ == "__main__":
    main()
