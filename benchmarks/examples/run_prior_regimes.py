# ruff: noqa: E402

"""prior_regimes example driver.

Run one fixed H+E model structure under several prior choices ("regimes") and
compare how each regime samples.

This is a template. Copy it, edit the regimes in :data:`REGIMES`, and run. A
regime is a function returning an :class:`HEModelConfig` with its prior fields
set; the shared :func:`benchmarks.models.he.build_he_model` assembles the model.
Each regime function's source is recorded in the results, so the report carries
the exact prior specification. Everything under ``benchmarks/examples/`` except
the committed examples is gitignored.

See ``benchmarks/prior_regimes.md`` for the design. Run from the repository root:

    python -m benchmarks.examples.run_prior_regimes --quick
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from benchmarks.core.env import configure_jax

configure_jax()

import jax.numpy as jnp
import numpyro.distributions as dist

from benchmarks.core.comparison import DEFAULT_METRICS, ComparisonSpec
from benchmarks.core.suite import Arm, comparison_suite
from benchmarks.models.he import HEModelConfig, build_he_model
from pyrenew.ascertainment import JointAscertainment
from pyrenew.deterministic import DeterministicVariable
from pyrenew.randomvariable import DistributionalVariable

BASELINE_REGIME = "example"


def example() -> HEModelConfig:
    """Starting-point priors for the H+E comparison.

    Copy this function, change the distributions, and add it to :data:`REGIMES`.
    The fixed structure is weekly hospital admissions plus daily ED visits, a
    joint ascertainment, and a weekly state-centered Rt process with no ED
    day-of-week effect.

    Returns
    -------
    HEModelConfig
        The model config for this regime.
    """
    return HEModelConfig(
        rt="state",
        day_of_week="none",
        autoreg_rv=DeterministicVariable("rt_diff_autoreg", 0.9),
        innovation_sd_rv=DeterministicVariable("rt_diff_innovation_sd", 0.01),
        log_rt_time_0_rv=DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5)),
        i0_rv=DistributionalVariable("I0", dist.Beta(1.0, 10.0)),
        hosp_conc_rv=DistributionalVariable("hosp_conc", dist.LogNormal(5.0, 1.0)),
        ed_conc_rv=DistributionalVariable("ed_conc", dist.LogNormal(4.0, 1.0)),
        ascertainment=JointAscertainment(
            name="he_ascertainment",
            signals=("hospital", "ed_visits"),
            baseline_rates=jnp.array([0.004, 0.004]),
            covariance_matrix=jnp.array([[0.09, 0.045], [0.045, 0.09]]),
        ),
    )


REGIMES: dict[str, Callable[[], HEModelConfig]] = {
    "example": example,
}

SPEC: ComparisonSpec = ComparisonSpec(
    name="prior_regimes",
    arms=tuple(REGIMES),
    baseline=BASELINE_REGIME,
    match_keys=("dataset",),
    metrics=DEFAULT_METRICS,
)

ARMS = [
    Arm(name=name, config=regime(), config_fields={"prior_config": name})
    for name, regime in REGIMES.items()
]


def _prior_provenance() -> dict[str, dict[str, str]]:
    """Capture each regime's source text for the results.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of regime name to its module and ``inspect.getsource`` text.
    """
    return {
        name: {"module": regime.__module__, "source": inspect.getsource(regime)}
        for name, regime in REGIMES.items()
    }


main = comparison_suite(
    SPEC,
    ARMS,
    build_he_model,
    description=__doc__,
    extra_payload={"prior_configs": _prior_provenance()},
)

if __name__ == "__main__":
    main()
