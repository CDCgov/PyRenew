"""Declarative description of an A/B(/C...) benchmark comparison.

A suite owns a :class:`ComparisonSpec`. It names the arms being compared,
which arm is the baseline that ratios are taken against, which fields make
two fits comparable, and which metrics to report and in which direction.
Reporting in :mod:`benchmarks.core.reporting` reads the spec and derives the
candidate, comparison, and per-site tables from it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    """One reported metric and the direction that counts as better.

    Parameters
    ----------
    label
        Human-readable name for console and Markdown tables.
    key
        Attribute name on :class:`benchmarks.core.runner.FitMetrics`.
    fmt
        ``str.format`` template used to render the value.
    higher_is_better
        Whether a larger value is an improvement. Controls how the
        baseline-relative benefit ratio is computed.
    """

    label: str
    key: str
    fmt: str
    higher_is_better: bool


@dataclass(frozen=True)
class ComparisonSpec:
    """Describes an A/B comparison and how to report it.

    Parameters
    ----------
    name
        Identifier used in output filenames and headers.
    arms
        Ordered arm labels. Each fit is assigned to one arm.
    baseline
        Arm that benefit ratios are computed relative to. Must be one of
        ``arms``.
    match_keys
        Fields that must be equal for two fits to form a comparable group.
        Each key is resolved from a candidate row, which carries ``dataset``
        and the flattened ``config_fields``.
    metrics
        Metrics to report, with their improvement direction.
    """

    name: str
    arms: tuple[str, ...]
    baseline: str
    match_keys: tuple[str, ...]
    metrics: tuple[MetricSpec, ...]

    def __post_init__(self) -> None:
        """Validate that the baseline is one of the declared arms.

        A single arm is permitted: the suite then profiles one model and the
        comparison table is empty, while the candidate, run, and per-site
        tables are still produced.
        """
        if not self.arms:
            raise ValueError(f"ComparisonSpec {self.name!r} needs at least one arm.")
        if self.baseline not in self.arms:
            raise ValueError(
                f"ComparisonSpec {self.name!r} baseline {self.baseline!r} "
                f"is not one of the arms {self.arms}."
            )

    @property
    def other_arms(self) -> tuple[str, ...]:  # numpydoc ignore=RT01
        """Return the non-baseline arms in declared order."""
        return tuple(arm for arm in self.arms if arm != self.baseline)


DEFAULT_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("Wall time (s)", "wall_time_s", "{:.1f}", False),
    MetricSpec("ESS/s Rt (median)", "ess_per_sec_rt_median", "{:.3f}", True),
    MetricSpec("ESS/s Rt (min)", "ess_per_sec_rt_min", "{:.3f}", True),
    MetricSpec("Divergences", "divergences", "{:d}", False),
    MetricSpec("Tree depth (mean)", "tree_depth_mean", "{:.2f}", False),
    MetricSpec("Tree depth (max)", "tree_depth_max", "{:d}", False),
    MetricSpec("E-BFMI (min)", "ebfmi_min", "{:.3f}", True),
    MetricSpec("R-hat Rt (max)", "rhat_rt_max", "{:.3f}", False),
)
"""Standard NUTS performance and convergence metrics shared by suites."""
