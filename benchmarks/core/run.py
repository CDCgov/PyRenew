"""Drive a list of candidates through fitting and reporting.

``run_comparison`` is the shared loop a driver calls once it has a list of
:class:`benchmarks.core.runner.Candidate` and a
:class:`benchmarks.core.comparison.ComparisonSpec`. It fits every candidate
over the requested repeats, prints the comparison and per-site tables, and
optionally writes the artifacts. Drivers supply the model construction; this
supplies the orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.core.comparison import ComparisonSpec
from benchmarks.core.reporting import (
    print_comparison_tables,
    print_fit_progress,
    write_results,
)
from benchmarks.core.runner import Candidate, FitResult, McmcSettings, fit_candidate


def run_comparison(
    candidates: list[Candidate],
    spec: ComparisonSpec,
    settings: McmcSettings,
    *,
    suite_name: str,
    repeats: int = 1,
    output_dir: Path | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> list[FitResult]:
    """Fit every candidate over repeats, print tables, and optionally write artifacts.

    Parameters
    ----------
    candidates
        Candidates to fit, each built and measured once per repeat.
    spec
        Comparison declaration used for reporting.
    settings
        MCMC controls shared across candidates.
    suite_name
        Identifier used in progress output and artifact filenames.
    repeats
        Number of times to refit each candidate, perturbing the seed.
    output_dir
        Directory for CSV / JSON / Markdown artifacts. When ``None``, results
        are printed but not written.
    extra_payload
        Optional mapping merged into the JSON payload, for provenance such as
        prior-config sources.

    Returns
    -------
    list[FitResult]
        Every per-fit result, in candidate-then-repeat order.
    """
    n_fits = len(candidates) * repeats
    print(
        f"{suite_name}: {len(candidates)} candidate(s) x "
        f"{repeats} repeat(s) = {n_fits} fits",
        flush=True,
    )

    results: list[FitResult] = []
    for candidate in candidates:
        for repeat in range(repeats):
            print(
                f">> fitting {candidate.name} (repeat {repeat + 1}/{repeats}) ...",
                flush=True,
            )
            result = fit_candidate(candidate, settings, repeat)
            results.append(result)
            print_fit_progress(candidate.name, repeat, repeats, result)

    print_comparison_tables(results, spec)
    if output_dir is not None:
        write_results(
            output_dir,
            suite_name=suite_name,
            results=results,
            spec=spec,
            extra_payload=extra_payload,
        )
        print(f"\nWrote results to {output_dir}", flush=True)
    return results
