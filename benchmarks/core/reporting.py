"""Reporting helpers for benchmark suites."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax

from benchmarks.core.runner import FitResult


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


def aggregate_results(
    results: list[FitResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate per-fit results into summary rows.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]]]
        Per-candidate rows and matched state-vs-innovation comparison rows.
    """
    by_candidate: dict[str, list[FitResult]] = {}
    for result in results:
        by_candidate.setdefault(result.candidate, []).append(result)

    candidates: list[dict[str, Any]] = []
    for candidate, group in by_candidate.items():
        first = group[0]
        n_runs = len(group)
        candidates.append(
            {
                "candidate": candidate,
                "n_runs": n_runs,
                "dataset": first.dataset,
                "parameterization": first.config.parameterization,
                "rt_cadence": first.config.rt_cadence,
                "innovation_sd": first.config.innovation_sd,
                "autoreg": first.config.autoreg,
                "wall_time_s": _mean(result.metrics.wall_time_s for result in group),
                "ess_per_sec_rt_median": _mean(
                    result.metrics.ess_per_sec_rt_median for result in group
                ),
                "ess_per_sec_rt_min": _mean(
                    result.metrics.ess_per_sec_rt_min for result in group
                ),
                "divergences_total": sum(
                    result.metrics.divergences for result in group
                ),
                "tree_depth_mean": _mean(
                    result.metrics.tree_depth_mean for result in group
                ),
                "tree_depth_max": max(
                    result.metrics.tree_depth_max for result in group
                ),
                "ebfmi_min": _mean(result.metrics.ebfmi_min for result in group),
                "rhat_rt_max": _mean(result.metrics.rhat_rt_max for result in group),
            }
        )

    pairs: list[dict[str, Any]] = []
    by_pair: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in candidates:
        key = (
            row["dataset"],
            row["rt_cadence"],
            row["innovation_sd"],
            row["autoreg"],
        )
        by_pair.setdefault(key, {})[row["parameterization"]] = row

    for key, sides in by_pair.items():
        innovation = sides.get("innovation")
        state = sides.get("state")
        if innovation is None or state is None:
            continue
        dataset, rt_cadence, innovation_sd, autoreg = key
        pairs.append(
            {
                "dataset": dataset,
                "rt_cadence": rt_cadence,
                "innovation_sd": innovation_sd,
                "autoreg": autoreg,
                "wall_s_innov": innovation["wall_time_s"],
                "wall_s_state": state["wall_time_s"],
                "wall_s_ratio": _ratio(state["wall_time_s"], innovation["wall_time_s"]),
                "ess_per_s_med_innov": innovation["ess_per_sec_rt_median"],
                "ess_per_s_med_state": state["ess_per_sec_rt_median"],
                "ess_per_s_med_ratio": _ratio(
                    state["ess_per_sec_rt_median"],
                    innovation["ess_per_sec_rt_median"],
                ),
                "ess_per_s_min_innov": innovation["ess_per_sec_rt_min"],
                "ess_per_s_min_state": state["ess_per_sec_rt_min"],
                "ess_per_s_min_ratio": _ratio(
                    state["ess_per_sec_rt_min"],
                    innovation["ess_per_sec_rt_min"],
                ),
                "divergences_innov": innovation["divergences_total"],
                "divergences_state": state["divergences_total"],
                "tree_depth_mean_innov": innovation["tree_depth_mean"],
                "tree_depth_mean_state": state["tree_depth_mean"],
                "tree_depth_max_innov": innovation["tree_depth_max"],
                "tree_depth_max_state": state["tree_depth_max"],
                "ebfmi_min_innov": innovation["ebfmi_min"],
                "ebfmi_min_state": state["ebfmi_min"],
                "rhat_rt_max_innov": innovation["rhat_rt_max"],
                "rhat_rt_max_state": state["rhat_rt_max"],
            }
        )

    return (
        sorted(candidates, key=lambda row: row["candidate"]),
        sorted(
            pairs,
            key=lambda row: (
                row["dataset"],
                row["rt_cadence"],
                row["innovation_sd"],
                row["autoreg"],
            ),
        ),
    )


def print_pairwise_tables(results: list[FitResult]) -> None:
    """Print one paired comparison table per matched pair."""
    _, pairs = aggregate_results(results)
    if not pairs:
        print("No state-vs-innovation pairs to summarize.")
        return

    metrics = [
        ("Wall time (s)", "wall_s", "{:.1f}", False),
        ("ESS/s Rt (median)", "ess_per_s_med", "{:.3f}", True),
        ("ESS/s Rt (min)", "ess_per_s_min", "{:.3f}", True),
        ("Divergences", "divergences", "{:d}", False),
        ("Tree depth (mean)", "tree_depth_mean", "{:.2f}", False),
        ("Tree depth (max)", "tree_depth_max", "{:d}", False),
        ("E-BFMI (min)", "ebfmi_min", "{:.3f}", True),
        ("R-hat Rt (max)", "rhat_rt_max", "{:.3f}", False),
    ]

    for row in pairs:
        print()
        print(
            f"--- {row['dataset']} | cadence={row['rt_cadence']} "
            f"| innovation_sd={row['innovation_sd']:g} ---"
        )
        print(f"{'metric':<22} {'innovation':>12} {'state':>12} {'state/innov':>12}")
        print("-" * 62)
        for label, prefix, fmt, higher_is_better in metrics:
            innovation = row[f"{prefix}_innov"]
            state = row[f"{prefix}_state"]
            ratio = row.get(f"{prefix}_ratio", _ratio(state, innovation))
            print(
                f"{label:<22} {fmt.format(innovation):>12} {fmt.format(state):>12} "
                f"{_format_ratio(ratio, higher_is_better):>12}"
            )

    print()
    print("(* marks an improvement over innovation; ratios are state / innovation)")


def write_results(
    output_dir: Path,
    *,
    suite_name: str,
    results: list[FitResult],
) -> None:
    """Write CSV, JSON, and Markdown artifacts to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates, pairs = aggregate_results(results)
    runs = [_result_to_row(result) for result in results]
    generated_at = datetime.now(UTC).isoformat()

    _write_csv(output_dir / f"{suite_name}_runs.csv", runs)
    _write_csv(output_dir / f"{suite_name}_candidates.csv", candidates)
    _write_csv(output_dir / f"{suite_name}_pairs.csv", pairs)

    payload = {
        "suite": suite_name,
        "generated_at": generated_at,
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "runs": runs,
        "candidates": candidates,
        "pairs": pairs,
    }
    with open(output_dir / f"{suite_name}_runs.json", "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
        f.write("\n")

    report = "\n".join(
        [
            f"# {suite_name} benchmark",
            "",
            f"Generated: {generated_at}",
            f"Runs: {len(results)}",
            f"x64 enabled: {bool(jax.config.jax_enable_x64)}",
            "",
            "## Candidates",
            "",
            _markdown_table(
                candidates,
                [
                    "candidate",
                    "n_runs",
                    "dataset",
                    "rt_cadence",
                    "parameterization",
                    "innovation_sd",
                    "autoreg",
                    "wall_time_s",
                    "ess_per_sec_rt_median",
                    "ess_per_sec_rt_min",
                    "divergences_total",
                ],
            ),
            "",
            "## State vs Innovation",
            "",
            _markdown_table(
                pairs,
                [
                    "dataset",
                    "rt_cadence",
                    "innovation_sd",
                    "autoreg",
                    "wall_s_ratio",
                    "ess_per_s_med_ratio",
                    "ess_per_s_min_ratio",
                    "divergences_innov",
                    "divergences_state",
                ],
            ),
            "",
        ]
    )
    (output_dir / f"{suite_name}_report.md").write_text(report)


def _mean(values: Any) -> float:
    """Compute the arithmetic mean of an iterable.

    Returns
    -------
    float
        Mean of the provided values.
    """
    values = list(values)
    return sum(values) / len(values)


def _ratio(state: float, innovation: float) -> float | None:
    """Compute the state-to-innovation ratio when finite.

    Returns
    -------
    float | None
        Ratio, or ``None`` when either input makes the ratio invalid.
    """
    if (
        innovation == 0
        or not math.isfinite(float(innovation))
        or not math.isfinite(float(state))
    ):
        return None
    return state / innovation


def _format_ratio(ratio: float | None, higher_is_better: bool) -> str:
    """Format a comparison ratio for terminal tables.

    Returns
    -------
    str
        Human-readable ratio string, with an improvement marker when relevant.
    """
    if ratio is None:
        return "n/a"
    improved = (higher_is_better and ratio > 1.05) or (
        not higher_is_better and ratio < 0.95
    )
    return f"{ratio:.2f}x{' *' if improved else ''}"


def _result_to_row(result: FitResult) -> dict[str, Any]:
    """Flatten one fit result into a serializable row.

    Returns
    -------
    dict[str, Any]
        Row containing metadata, settings, and metrics for one fit.
    """
    return {
        "candidate": result.candidate,
        "repeat": result.repeat,
        "dataset": result.dataset,
        **asdict(result.config),
        **asdict(result.settings),
        **asdict(result.metrics),
        "n_init_points": result.n_initialization_points,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a CSV file when rows are present."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
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
