"""Reporting helpers for benchmark suites.

Aggregation, console tables, and file artifacts are driven by a
:class:`benchmarks.core.comparison.ComparisonSpec`. The spec names the arms,
the baseline, the fields that make fits comparable, and the metrics to
report. Nothing here hard-codes a particular comparison, so a suite that
pits a PyRenew model against the production HEW model reuses the same
reporting path as a parameterization A/B.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import numpy as np

from benchmarks.core.comparison import ComparisonSpec
from benchmarks.core.runner import FitResult
from benchmarks.core.signals import DatasetBundle, SignalSeries

_METRIC_REDUCERS: dict[str, str] = {
    "divergences": "sum",
    "tree_depth_max": "max",
    "ebfmi_min": "min",
    "rhat_rt_max": "max",
}


def print_fit_progress(
    candidate: str, repeat: int, total_repeats: int, result: FitResult
) -> None:
    """Print one progress line after a fit completes."""
    repeat_label = (
        f" (repeat {repeat + 1}/{total_repeats})" if total_repeats > 1 else ""
    )
    print(
        f"   done {candidate}{repeat_label}: "
        f"{result.metrics.wall_time_s:.1f}s, "
        f"divergences={result.metrics.divergences}, "
        f"min ESS/s={result.metrics.ess_per_sec_rt_min:.2f}",
        flush=True,
    )


def print_data_summary(bundles: Iterable[DatasetBundle]) -> None:
    """Print a compact summary of each dataset bundle before fitting."""
    for bundle in bundles:
        print()
        print(format_bundle_summary(bundle))


def format_bundle_summary(bundle: DatasetBundle) -> str:
    """Format a compact text summary of one dataset bundle.

    Returns
    -------
    str
        Multi-line description of the bundle and each of its signals.
    """
    fixed_keys = ", ".join(sorted(bundle.fixed_params)) or "none"
    lines = [
        f"Dataset: {bundle.name}",
        f"  population_size: {bundle.population_size:g}",
        f"  obs_start_date: {bundle.obs_start_date}",
        f"  n_days_post_init: {bundle.n_days_post_init}",
        f"  gen_int_pmf_len: {len(bundle.gen_int_pmf)}",
        f"  fixed_params: {fixed_keys}",
    ]
    for signal in bundle.signals.values():
        lines.extend(_format_signal_summary(signal))
    return "\n".join(lines)


def _format_signal_summary(signal: SignalSeries) -> list[str]:
    """Format the indented summary lines for one signal series.

    Returns
    -------
    list[str]
        Description lines covering shape, date range, missingness, and value
        statistics.
    """
    values = np.asarray(signal.values, dtype=float)
    finite = values[np.isfinite(values)]
    missing = int(values.size - finite.size)
    if finite.size:
        value_summary = (
            f"min={np.min(finite):.4g}, "
            f"mean={np.mean(finite):.4g}, "
            f"max={np.max(finite):.4g}"
        )
    else:
        value_summary = "no finite values"
    extras = ", ".join(sorted(signal.extras)) or "none"
    return [
        f"  signal: {signal.name}",
        f"    cadence: {signal.cadence}",
        f"    n_obs: {len(values)}",
        f"    date_range: {signal.start_date} to {signal.end_date}",
        f"    missing_or_nan: {missing}",
        f"    values: {value_summary}",
        f"    extras: {extras}",
    ]


def aggregate_candidates(results: list[FitResult]) -> list[dict[str, Any]]:
    """Aggregate per-fit results into one row per candidate.

    Metrics are averaged across repeats, except worst-case diagnostics:
    divergences are summed, and tree depth, E-BFMI, and R-hat keep their
    worst observed value. Each row carries the candidate's ``arm`` and its
    flattened ``config_fields`` so downstream grouping can read them.

    Returns
    -------
    list[dict[str, Any]]
        One row per candidate, sorted by candidate name.
    """
    by_candidate: dict[str, list[FitResult]] = {}
    for result in results:
        by_candidate.setdefault(result.candidate, []).append(result)

    candidates: list[dict[str, Any]] = []
    for candidate, group in by_candidate.items():
        first = group[0]
        row: dict[str, Any] = {
            "candidate": candidate,
            "arm": first.arm,
            "n_runs": len(group),
            "dataset": first.dataset,
            **first.config_fields,
        }
        for field_name in asdict(first.metrics):
            values = [getattr(r.metrics, field_name) for r in group]
            row[field_name] = _reduce_metric(field_name, values)
        candidates.append(row)

    return sorted(candidates, key=lambda row: row["candidate"])


def _reduce_metric(field_name: str, values: list[Any]) -> Any:
    """Reduce one metric across repeats using its configured reducer.

    Returns
    -------
    Any
        Summed, min, max, or mean value depending on the metric.
    """
    reducer = _METRIC_REDUCERS.get(field_name, "mean")
    if reducer == "sum":
        return sum(values)
    if reducer == "min":
        return min(values)
    if reducer == "max":
        return max(values)
    return _mean(values)


def build_comparison(
    candidates: list[dict[str, Any]], spec: ComparisonSpec
) -> list[dict[str, Any]]:
    """Lay candidates side by side per comparable group.

    Candidates are grouped by ``spec.match_keys``. A group is reported only
    when the baseline arm and at least one other arm are present. Each metric
    contributes one value column per arm and, for each non-baseline arm, a
    baseline-relative benefit ratio.

    Returns
    -------
    list[dict[str, Any]]
        One row per comparable group, sorted by the match-key values.
    """
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in candidates:
        key = tuple(row.get(match_key) for match_key in spec.match_keys)
        groups.setdefault(key, {})[row["arm"]] = row

    comparison: list[dict[str, Any]] = []
    for key, arms in groups.items():
        if spec.baseline not in arms:
            continue
        present_others = [arm for arm in spec.other_arms if arm in arms]
        if not present_others:
            continue

        baseline_row = arms[spec.baseline]
        out: dict[str, Any] = dict(zip(spec.match_keys, key))
        for metric in spec.metrics:
            for arm in spec.arms:
                if arm in arms:
                    out[f"{metric.key}__{arm}"] = arms[arm][metric.key]
            for arm in present_others:
                out[f"{metric.key}__ratio__{arm}"] = _comparison_ratio(
                    baseline_row[metric.key],
                    arms[arm][metric.key],
                    metric.higher_is_better,
                )
        comparison.append(out)

    return sorted(
        comparison, key=lambda row: tuple(_sort_key(row[k]) for k in spec.match_keys)
    )


def _sort_key(value: Any) -> tuple[int, Any]:
    """Build a None-tolerant sort key.

    Returns
    -------
    tuple[int, Any]
        Pair ordering ``None`` ahead of present values of mixed type.
    """
    return (0, "") if value is None else (1, value)


def aggregate_parameter_summaries(results: list[FitResult]) -> list[dict[str, Any]]:
    """Aggregate scalar posterior summaries by candidate, arm, and site.

    Returns
    -------
    list[dict[str, Any]]
        One row per candidate, dataset, arm, and posterior site. ESS/s values
        are computed per scalar element using that fit's wall time before
        aggregation.
    """
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for result in results:
        for summary in result.parameter_summaries:
            key = (result.candidate, result.dataset, result.arm, summary.site)
            group = groups.setdefault(
                key,
                {
                    "candidate": result.candidate,
                    "dataset": result.dataset,
                    "arm": result.arm,
                    "site": summary.site,
                    "n_elements": 0,
                    "ess_values": [],
                    "ess_per_sec_values": [],
                    "rhat_values": [],
                },
            )
            group["n_elements"] += 1
            if math.isfinite(summary.ess):
                group["ess_values"].append(summary.ess)
                ess_per_sec = _ratio(summary.ess, result.metrics.wall_time_s)
                if ess_per_sec is not None:
                    group["ess_per_sec_values"].append(ess_per_sec)
            if math.isfinite(summary.rhat):
                group["rhat_values"].append(summary.rhat)

    rows: list[dict[str, Any]] = []
    for group in groups.values():
        ess_values = group.pop("ess_values")
        ess_per_sec_values = group.pop("ess_per_sec_values")
        rhat_values = group.pop("rhat_values")
        rows.append(
            {
                **group,
                "n_finite_ess": len(ess_values),
                "ess_median": _median(ess_values),
                "ess_min": min(ess_values) if ess_values else float("nan"),
                "ess_per_sec_median": _median(ess_per_sec_values),
                "ess_per_sec_min": (
                    min(ess_per_sec_values) if ess_per_sec_values else float("nan")
                ),
                "rhat_max": max(rhat_values) if rhat_values else float("nan"),
            }
        )

    return sorted(
        rows,
        key=lambda row: (row["candidate"], row["dataset"], row["arm"], row["site"]),
    )


def print_comparison_tables(results: list[FitResult], spec: ComparisonSpec) -> None:
    """Print baseline-relative comparison and per-site parameter ESS tables."""
    candidates = aggregate_candidates(results)
    comparison = build_comparison(candidates, spec)
    if not comparison:
        print("No comparable arm groups to summarize.")
        print_parameter_site_table(results)
        return

    label_width = 22
    arm_width = 14
    benefit_width = 16
    for row in comparison:
        print()
        group_label = ", ".join(f"{k}={row.get(k)}" for k in spec.match_keys)
        print(f"--- {group_label} ---")
        header = f"{'metric':<{label_width}}"
        header += "".join(f"{arm:>{arm_width}}" for arm in spec.arms)
        header += "".join(
            f"{arm + ' benefit':>{benefit_width}}" for arm in spec.other_arms
        )
        print(header)
        print("-" * len(header))
        for metric in spec.metrics:
            line = f"{metric.label:<{label_width}}"
            for arm in spec.arms:
                value = row.get(f"{metric.key}__{arm}")
                cell = metric.fmt.format(value) if value is not None else "n/a"
                line += f"{cell:>{arm_width}}"
            for arm in spec.other_arms:
                ratio = row.get(f"{metric.key}__ratio__{arm}")
                line += f"{_format_ratio(ratio):>{benefit_width}}"
            print(line)

    print()
    print(
        f"(* marks improvement over the {spec.baseline} baseline; "
        "ratios > 1 favor the listed arm)"
    )
    print_parameter_site_table(results)


def print_parameter_site_table(results: list[FitResult]) -> None:
    """Print per-site ESS summaries for posterior parameters."""
    rows = aggregate_parameter_summaries(results)
    if not rows:
        print()
        print("No parameter summaries to report.")
        return

    print()
    print("--- Parameter ESS by site ---")
    print(
        f"{'candidate':<18} {'site':<42} "
        f"{'ESS med':>10} {'ESS min':>10} {'ESS/s med':>10} "
        f"{'ESS/s min':>10} {'R-hat max':>10}"
    )
    print("-" * 116)
    previous_candidate = None
    for row in rows:
        if previous_candidate is not None and row["candidate"] != previous_candidate:
            print("-" * 116)
        previous_candidate = row["candidate"]
        print(
            f"{str(row['candidate']):<18} "
            f"{_truncate(str(row['site']), 42):<42} "
            f"{_format_console_number(row['ess_median']):>10} "
            f"{_format_console_number(row['ess_min']):>10} "
            f"{_format_console_number(row['ess_per_sec_median']):>10} "
            f"{_format_console_number(row['ess_per_sec_min']):>10} "
            f"{_format_console_number(row['rhat_max']):>10}"
        )


def write_results(
    output_dir: Path,
    *,
    comparison_name: str,
    results: list[FitResult],
    spec: ComparisonSpec,
    extra_payload: dict[str, Any] | None = None,
) -> None:
    """Write CSV, JSON, and Markdown artifacts to ``output_dir``.

    ``extra_payload`` is merged into the JSON payload, letting a suite attach
    provenance such as the source of its prior-config functions. Its keys must
    not collide with the standard payload keys.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = aggregate_candidates(results)
    comparison = build_comparison(candidates, spec)
    runs = [_result_to_row(result) for result in results]
    parameters = _parameter_summary_rows(results)
    parameter_sites = aggregate_parameter_summaries(results)
    generated_at = datetime.now(UTC).isoformat()

    _write_csv(output_dir / f"{comparison_name}_runs.csv", runs)
    _write_csv(output_dir / f"{comparison_name}_candidates.csv", candidates)
    _write_csv(output_dir / f"{comparison_name}_comparison.csv", comparison)
    _write_csv(output_dir / f"{comparison_name}_parameters.csv", parameters)

    payload = {
        "suite": comparison_name,
        "generated_at": generated_at,
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "arms": list(spec.arms),
        "baseline": spec.baseline,
        "runs": runs,
        "candidates": candidates,
        "comparison": comparison,
        "parameters": parameters,
        "parameter_sites": parameter_sites,
    }
    if extra_payload is not None:
        payload.update(extra_payload)
    with open(output_dir / f"{comparison_name}_runs.json", "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
        f.write("\n")

    comparison_columns = list(spec.match_keys) + [
        f"{metric.key}__ratio__{arm}"
        for metric in spec.metrics
        for arm in spec.other_arms
    ]
    report = "\n".join(
        [
            f"# {comparison_name} benchmark",
            "",
            f"Generated: {generated_at}",
            f"Runs: {len(results)}",
            f"Parameter rows: {len(parameters)}",
            f"x64 enabled: {bool(jax.config.jax_enable_x64)}",
            f"Arms: {', '.join(spec.arms)} (baseline: {spec.baseline})",
            "",
            "## Candidates",
            "",
            _markdown_table(candidates, _ordered_union_keys(candidates)),
            "",
            "## Comparison",
            "",
            _markdown_table(comparison, comparison_columns),
            "",
            "## Parameter ESS by Site",
            "",
            _markdown_table(
                parameter_sites,
                [
                    "candidate",
                    "dataset",
                    "arm",
                    "site",
                    "n_elements",
                    "n_finite_ess",
                    "ess_median",
                    "ess_min",
                    "ess_per_sec_median",
                    "ess_per_sec_min",
                    "rhat_max",
                ],
            ),
            "",
        ]
    )
    (output_dir / f"{comparison_name}_report.md").write_text(report)


def _mean(values: Any) -> float:
    """Compute the arithmetic mean of an iterable.

    Returns
    -------
    float
        Mean of the provided values.
    """
    values = list(values)
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    """Compute the median of a finite-valued list.

    Returns
    -------
    float
        Median value, or NaN when no values are provided.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _ratio(numerator: float, denominator: float) -> float | None:
    """Compute a ratio when finite.

    Returns
    -------
    float | None
        Ratio, or ``None`` when either input makes the ratio invalid.
    """
    if (
        denominator == 0
        or not math.isfinite(float(denominator))
        or not math.isfinite(float(numerator))
    ):
        return None
    return numerator / denominator


def _comparison_ratio(
    baseline: float, arm: float, higher_is_better: bool
) -> float | None:
    """Compute an arm's benefit ratio relative to the baseline.

    Returns
    -------
    float | None
        Ratio greater than 1 when the arm beats the baseline, or ``None``
        when invalid.
    """
    if higher_is_better:
        return _ratio(arm, baseline)
    return _ratio(baseline, arm)


def _format_ratio(ratio: float | None) -> str:
    """Format a comparison ratio for terminal tables.

    Returns
    -------
    str
        Human-readable ratio string, with an improvement marker when relevant.
    """
    if ratio is None:
        return "n/a"
    improved = ratio > 1.05
    return f"{ratio:.2f}x{' *' if improved else ''}"


def _format_console_number(value: Any) -> str:
    """Format a compact numeric value for fixed-width console tables.

    Returns
    -------
    str
        Fixed-point number string, or ``"n/a"`` for missing values.
    """
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    formatted = f"{float(value):.3f}".rstrip("0").rstrip(".")
    if formatted == "-0":
        return "0"
    return formatted


def _truncate(value: str, width: int) -> str:
    """Truncate text for fixed-width console tables.

    Returns
    -------
    str
        Text truncated to the requested width.
    """
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "~"


def _result_to_row(result: FitResult) -> dict[str, Any]:
    """Flatten one fit result into a serializable row.

    Returns
    -------
    dict[str, Any]
        Row containing metadata, configuration axes, settings, and metrics.
    """
    return {
        "candidate": result.candidate,
        "arm": result.arm,
        "repeat": result.repeat,
        "dataset": result.dataset,
        **result.config_fields,
        **asdict(result.settings),
        **asdict(result.metrics),
        "n_init_points": result.n_initialization_points,
    }


def _parameter_summary_rows(results: list[FitResult]) -> list[dict[str, Any]]:
    """Flatten per-parameter posterior summaries.

    Returns
    -------
    list[dict[str, Any]]
        One row per scalar posterior site element per fit.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        for summary in result.parameter_summaries:
            rows.append(
                {
                    "candidate": result.candidate,
                    "arm": result.arm,
                    "repeat": result.repeat,
                    "dataset": result.dataset,
                    **result.config_fields,
                    "site": summary.site,
                    "index": summary.index,
                    "mean": summary.mean,
                    "ess": summary.ess,
                    "rhat": summary.rhat,
                }
            )
    return rows


def _ordered_union_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Collect dict keys across rows, preserving first-seen order.

    Returns
    -------
    list[str]
        Union of keys in the order they first appear.
    """
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(key, None)
    return list(seen)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a CSV file using the union of keys across rows."""
    if not rows:
        return
    fieldnames = _ordered_union_keys(rows)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render rows as a Markdown table.

    Returns
    -------
    str
        Markdown table text, or a placeholder when there are no rows.
    """
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_format_value(row.get(c)) for c in columns) + " |"
        )
    return "\n".join(lines) + "\n"


def _format_value(value: Any) -> str:
    """Format one value for Markdown output.

    Returns
    -------
    str
        Compact string representation of the value.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4g}"
    return str(value)


def _json_default(value: Any) -> Any:
    """Convert benchmark objects for JSON serialization.

    Returns
    -------
    Any
        JSON-compatible representation of the value.
    """
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")
