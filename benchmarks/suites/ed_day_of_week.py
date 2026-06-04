# ruff: noqa: E402

"""ed_day_of_week benchmark suite.

Compare a PyRenew ``MultiSignalModel`` with and without a daily ED-visit
day-of-week effect, holding everything else fixed: weekly-aggregated hospital
admissions plus daily ED visits, a joint Gaussian ascertainment, and a weekly
state-centered $\\mathcal{R}(t)$ process. The two arms differ only in whether
the ED observation infers a weekday multiplier, so any difference in sampling
or posterior spread is attributable to that one effect. The ``dow`` arm infers
the weekday effect; the ``no_dow`` arm omits it.

Run as a module from the repository root:

    python -m benchmarks.suites.ed_day_of_week --quick

See ``--help`` for all options.
"""

from __future__ import annotations

from benchmarks.core.env import configure_jax

configure_jax()

from benchmarks.core.comparison import DEFAULT_METRICS, ComparisonSpec
from benchmarks.core.suite import comparison_suite
from benchmarks.models.he import HEModelConfig, build_he_model, he_arm

SPEC: ComparisonSpec = ComparisonSpec(
    name="ed_day_of_week",
    arms=("no_dow", "dow"),
    baseline="no_dow",
    match_keys=("dataset",),
    metrics=DEFAULT_METRICS,
)

ARMS = [
    he_arm("no_dow", HEModelConfig(rt="state", day_of_week="none")),
    he_arm("dow", HEModelConfig(rt="state", day_of_week="infer")),
]

main = comparison_suite(SPEC, ARMS, build_he_model, description=__doc__)

if __name__ == "__main__":
    main()
