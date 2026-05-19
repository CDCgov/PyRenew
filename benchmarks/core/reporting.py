"""Reporting helpers for benchmark suites.

The module exposes:

- :func:`print_fit_progress` for one-line stdout updates while fits run.
- :func:`print_pairwise_tables` for a human-readable stdout summary that
  compares the innovation and state parameterizations of each candidate
  pair.
- :func:`write_results` for persistent CSV / JSON / Markdown output with
  readable column names.

Column names use short, lowercase tokens. State-vs-innovation pair columns
follow the convention ``<metric>_innov``, ``<metric>_state``,
``<metric>_ratio`` (ratio is ``state / innovation``).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax

from benchmarks.core.runner import FitResult


@dataclass(frozen=True)
class PairKey:
    """Identity of one state-vs-innovation comparison.

    Two :class:`FitResult` rows form a pair when their ``PairKey`` values are
    equal; only ``parameterization`` differs.
    """

    dataset: str
    rt_cadence: str
    innovation_sd: float
    autoreg: float


def _pair_key(result: FitResult) -> PairKey:
    """Return the comparison key for a result.

    Returns
    -------
    PairKey
        Identity used to pair state and innovation fits.
    """
    return PairKey(
        dataset=result.dataset,
        rt_cadence=result.config.rt_cadence,
        innovation_sd=result.config.innovation_sd,
        autoreg=result.config.autoreg,
    )


def _ratio(state: float, innov: float, higher_is_better: bool) -> tuple[str, bool]:
    """Format a state/innovation ratio and flag a state-side improvement.

    Returns
    -------
    tuple[str, bool]
        Formatted ratio and whether the state side improves over innovation
        by at least 5%.
    """
    if innov == 0 or innov != innov:
        return "n/a", False
    ratio = state / innov
    improved = (higher_is_better and ratio > 1.05) or (
        not higher_is_better and ratio < 0.95
    )
    return f"{ratio:.2f}x", improved


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


def _aggregate_by_candidate(
    results: list[FitResult],
) -> dict[str, dict[str, Any]]:
    """Average metrics across repeats for each candidate name.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from candidate name to averaged metric fields plus the shared
        config and dataset.
    """
    grouped: dict[str, list[FitResult]] = {}
    for r in results:
        grouped.setdefault(r.candidate, []).append(r)

    aggregated: dict[str, dict[str, Any]] = {}
    for candidate, group in grouped.items():
        n = len(group)
        sum_wall = sum(r.metrics.wall_time_s for r in group)
        sum_ess_med = sum(r.metrics.ess_per_sec_rt_median for r in group)
        sum_ess_min = sum(r.metrics.ess_per_sec_rt_min for r in group)
        sum_td_mean = sum(r.metrics.tree_depth_mean for r in group)
        sum_ebfmi = sum(r.metrics.ebfmi_min for r in group)
        sum_rhat = sum(r.metrics.rhat_rt_max for r in group)
        max_td = max(r.metrics.tree_depth_max for r in group)
        total_div = sum(r.metrics.divergences for r in group)
        first = group[0]
        aggregated[candidate] = {
            "candidate": candidate,
            "n_runs": n,
            "dataset": first.dataset,
            "parameterization": first.config.parameterization,
            "rt_cadence": first.config.rt_cadence,
            "innovation_sd": first.config.innovation_sd,
            "autoreg": first.config.autoreg,
            "wall_time_s": sum_wall / n,
            "ess_per_sec_rt_median": sum_ess_med / n,
            "ess_per_sec_rt_min": sum_ess_min / n,
            "divergences_total": total_div,
            "tree_depth_mean": sum_td_mean / n,
            "tree_depth_max": max_td,
            "ebfmi_min": sum_ebfmi / n,
            "rhat_rt_max": sum_rhat / n,
        }
    return aggregated


def _build_pair_rows(
    aggregated: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair state and innovation candidates that share a :class:`PairKey`.

    Returns
    -------
    list[dict[str, Any]]
        One row per matched pair.
    """
    by_key: dict[tuple, dict[str, dict[str, Any]]] = {}
    for row in aggregated.values():
        key = (
            row["dataset"],
            row["rt_cadence"],
            row["innovation_sd"],
            row["autoreg"],
        )
        by_key.setdefault(key, {})[row["parameterization"]] = row

    pair_rows: list[dict[str, Any]] = []
    for key, sides in by_key.items():
        innov = sides.get("innovation")
        state = sides.get("state")
        if innov is None or state is None:
            continue
        dataset, rt_cadence, innovation_sd, autoreg = key
        pair_rows.append(
            {
                "dataset": dataset,
                "rt_cadence": rt_cadence,
                "innovation_sd": innovation_sd,
                "autoreg": autoreg,
                "wall_s_innov": innov["wall_time_s"],
                "wall_s_state": state["wall_time_s"],
                "wall_s_ratio": _safe_ratio(
                    state["wall_time_s"], innov["wall_time_s"]
                ),
                "ess_per_s_med_innov": innov["ess_per_sec_rt_median"],
                "ess_per_s_med_state": state["ess_per_sec_rt_median"],
                "ess_per_s_med_ratio": _safe_ratio(
                    state["ess_per_sec_rt_median"], innov["ess_per_sec_rt_median"]
                ),
                "ess_per_s_min_innov": innov["ess_per_sec_rt_min"],
                "ess_per_s_min_state": state["ess_per_sec_rt_min"],
                "ess_per_s_min_ratio": _safe_ratio(
                    state["ess_per_sec_rt_min"], innov["ess_per_sec_rt_min"]
                ),
                "divergences_innov": innov["divergences_total"],
                "divergences_state": state["divergences_total"],
                "tree_depth_mean_innov": innov["tree_depth_mean"],
                "tree_depth_mean_state": state["tree_depth_mean"],
                "tree_depth_max_innov": innov["tree_depth_max"],
                "tree_depth_max_state": state["tree_depth_max"],
                "ebfmi_min_innov": innov["ebfmi_min"],
                "ebfmi_min_state": state["ebfmi_min"],
                "rhat_rt_max_innov": innov["rhat_rt_max"],
                "rhat_rt_max_state": state["rhat_rt_max"],
            }
        )
    return pair_rows


def _safe_ratio(state: float, innov: float) -> float | None:
    """Compute ``state / innov`` guarding against zero and NaN.

    Returns
    -------
    float | None
        Ratio, or ``None`` if the divisor is zero or non-finite.
    """
    if innov == 0 or innov != innov:
        return None
    return state / innov


def print_pairwise_tables(results: list[FitResult]) -> None:
    """Print one paired comparison table per matched pair."""
    aggregated = _aggregate_by_candidate(results)
    pairs = _build_pair_rows(aggregated)
    if not pairs:
        print("No state-vs-innovation pairs to summarize.")
        return

    for row in pairs:
        label = (
            f"{row['dataset']} | cadence={row['rt_cadence']}"
            f" | innovation_sd={row['innovation_sd']:g}"
        )
        print()
        print(f"--- {label} ---")
        print(
            f"{'metric':<22} {'innovation':>12} {'state':>12} {'state/innov':>12}"
        )
        print("-" * 62)
        _print_metric_row(
            "Wall time (s)",
            row["wall_s_innov"],
            row["wall_s_state"],
            "{:.1f}",
            higher_is_better=False,
        )
        _print_metric_row(
            "ESS/s Rt (median)",
            row["ess_per_s_med_innov"],
            row["ess_per_s_med_state"],
            "{:.3f}",
            higher_is_better=True,
        )
        _print_metric_row(
            "ESS/s Rt (min)",
            row["ess_per_s_min_innov"],
            row["ess_per_s_min_state"],
            "{:.3f}",
            higher_is_better=True,
        )
        _print_metric_row(
            "Divergences",
            row["divergences_innov"],
            row["divergences_state"],
            "{:d}",
            higher_is_better=False,
        )
        _print_metric_row(
            "Tree depth (mean)",
            row["tree_depth_mean_innov"],
            row["tree_depth_mean_state"],
            "{:.2f}",
            higher_is_better=False,
        )
        _print_metric_row(
            "Tree depth (max)",
            row["tree_depth_max_innov"],
            row["tree_depth_max_state"],
            "{:d}",
            higher_is_better=False,
        )
        _print_metric_row(
            "E-BFMI (min)",
            row["ebfmi_min_innov"],
            row["ebfmi_min_state"],
            "{:.3f}",
            higher_is_better=True,
        )
        _print_metric_row(
            "R-hat Rt (max)",
            row["rhat_rt_max_innov"],
            row["rhat_rt_max_state"],
            "{:.3f}",
            higher_is_better=False,
        )
    print()
    print("(* marks an improvement over innovation; ratios are state / innovation)")


def _print_metric_row(
    label: str,
    innov: float | int,
    state: float | int,
    fmt: str,
    higher_is_better: bool,
) -> None:
    """Print one labeled metric row to stdout."""
    ratio_text, improved = _ratio(float(state), float(innov), higher_is_better)
    marker = " *" if improved else ""
    print(
        f"{label:<22} {fmt.format(innov):>12} {fmt.format(state):>12} "
        f"{ratio_text + marker:>12}"
    )


def _result_to_csv_row(result: FitResult) -> dict[str, Any]:
    """Convert one :class:`FitResult` to a flat CSV row.

    Returns
    -------
    dict[str, Any]
        Flat mapping with primitive values.
    """
    metrics = asdict(result.metrics)
    config = asdict(result.config)
    settings = asdict(result.settings)
    row = {
        "candidate": result.candidate,
        "repeat": result.repeat,
        "dataset": result.dataset,
        **config,
        **settings,
        **metrics,
        "n_init_points": result.n_initialization_points,
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` to ``path`` as a CSV."""
    if not rows:
        return
    columns = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _format_md_value(value: Any) -> str:
    """Format a value for a Markdown table cell.

    Returns
    -------
    str
        Markdown-safe string. Floats use four significant digits.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:.4g}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Format ``rows`` as a Markdown table over ``columns``.

    Returns
    -------
    str
        Markdown table text. ``"_No rows._"`` when ``rows`` is empty.
    """
    if not rows:
        return "_No rows._\n"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format_md_value(row.get(c)) for c in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body]) + "\n"


def _write_markdown_report(
    path: Path,
    *,
    suite_name: str,
    results: list[FitResult],
    aggregated: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    x64_enabled: bool,
) -> None:
    """Write a compact Markdown report covering candidates and pairwise comparisons."""
    lines = [
        f"# {suite_name} benchmark",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Runs: {len(results)}",
        f"x64 enabled: {x64_enabled}",
        "",
        "## Candidates (averaged over repeats)",
        "",
        _markdown_table(
            sorted(aggregated.values(), key=lambda r: r["candidate"]),
            [
                "candidate",
                "n_runs",
                "dataset",
                "rt_cadence",
                "parameterization",
                "innovation_sd",
                "wall_time_s",
                "ess_per_sec_rt_median",
                "ess_per_sec_rt_min",
                "divergences_total",
                "tree_depth_mean",
                "ebfmi_min",
                "rhat_rt_max",
            ],
        ),
        "",
        "## Pairwise: state vs innovation",
        "",
        "Ratios are `state / innovation`. ESS-ratio > 1 favors state-centered.",
        "Wall-time ratio > 1 means state is slower.",
        "",
        _markdown_table(
            pairs,
            [
                "dataset",
                "rt_cadence",
                "innovation_sd",
                "wall_s_ratio",
                "ess_per_s_med_ratio",
                "ess_per_s_min_ratio",
                "divergences_innov",
                "divergences_state",
                "ebfmi_min_innov",
                "ebfmi_min_state",
                "rhat_rt_max_innov",
                "rhat_rt_max_state",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines))


def write_results(
    output_dir: Path,
    *,
    suite_name: str,
    results: list[FitResult],
) -> None:
    """Write CSV, JSON, and Markdown artifacts to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated = _aggregate_by_candidate(results)
    pairs = _build_pair_rows(aggregated)
    x64_enabled = bool(jax.config.jax_enable_x64)

    raw_rows = [_result_to_csv_row(r) for r in results]
    _write_csv(output_dir / f"{suite_name}_runs.csv", raw_rows)
    _write_csv(
        output_dir / f"{suite_name}_candidates.csv",
        sorted(aggregated.values(), key=lambda r: r["candidate"]),
    )
    _write_csv(output_dir / f"{suite_name}_pairs.csv", pairs)

    payload = {
        "suite": suite_name,
        "generated_at": datetime.now(UTC).isoformat(),
        "x64_enabled": x64_enabled,
        "runs": raw_rows,
        "candidates": sorted(aggregated.values(), key=lambda r: r["candidate"]),
        "pairs": pairs,
    }
    with open(output_dir / f"{suite_name}_runs.json", "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
        f.write("\n")

    _write_markdown_report(
        output_dir / f"{suite_name}_report.md",
        suite_name=suite_name,
        results=results,
        aggregated=aggregated,
        pairs=pairs,
        x64_enabled=x64_enabled,
    )


def _json_default(value: Any) -> Any:
    """JSON encoder fallback for dataclasses and JAX scalars.

    Returns
    -------
    Any
        JSON-serializable representation.
    """
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def candidate_summary(results: Iterable[FitResult]) -> list[dict[str, Any]]:
    """Return per-candidate aggregated rows (averaged over repeats).

    Returns
    -------
    list[dict[str, Any]]
        Rows sorted by candidate name.
    """
    return sorted(
        _aggregate_by_candidate(list(results)).values(),
        key=lambda r: r["candidate"],
    )
