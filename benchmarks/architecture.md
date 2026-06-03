# `benchmarks/` architecture (maintainer guide)

## Purpose

Opt-in MCMC model-comparison experiments.
Not part of CI.
A suite fits two or more model candidates on the same data and reports per-candidate diagnostics plus a baseline-relative head-to-head.
Correctness lives in `test/`; this directory is for sampler and model comparisons.

## The one organizing rule

`core/` is reusable machinery.
A suite is one experiment that composes that machinery and **owns its model-construction code locally**.
Adapters to external systems count as machinery and stay in `core`; the suite decides how to use them.
If suites proliferate, subdivide `suites/` into subpackages.

## Layout

```
benchmarks/
├── core/
│   ├── env.py            configure_jax: float64 + XLA device count, set before jax import
│   ├── cli.py            add_common_args, settings_from_args (shared sampler/output flags)
│   ├── signals.py        SignalSeries, DatasetBundle, DatasetProvider (Protocol)
│   ├── datasets.py       SyntheticProvider over pyrenew/datasets/ fixtures
│   ├── real_data.py      RealDataProvider over CDC NHSN+NSSP (lazy cfa.stf imports)
│   ├── reference_data.py static location populations/names
│   ├── priors.py         benchmark-local priors for real-data builds
│   ├── comparison.py     MetricSpec, ComparisonSpec, DEFAULT_METRICS
│   ├── models.py         BuiltFit, align_weekly_observations (shared machinery)
│   ├── hew_model.py      adapter to the production HEW model
│   ├── runner.py         FitResult, Candidate, fit_and_measure, fit_candidate, metrics
│   ├── run.py            run_comparison: the shared fit / report / write loop
│   └── reporting.py      spec-driven aggregation + CSV/JSON/Markdown writers
├── suites/
│   ├── rt_params.py      innovation vs state Rt parameterization (defines BuildConfig, build_he_model)
│   └── pyrenew_vs_hew.py production HEW vs PyRenew no-day-of-week (local build fns)
├── examples/
│   └── run_prior_regimes.py one structure under several prior regimes (template)
└── results/              output (gitignored)

pyrenew/datasets/synthetic_hew_export.py   write_synthetic_hew_model_dir (the export bridge)
```

## Three abstractions that make it work

1. **`BuiltFit`(`models.py`)** wraps any `pyrenew.metaclass.Model` (PyRenew `MultiSignalModel` or production `PyrenewHEWModel`) plus the `run_kwargs` for `model.run`.
   Because both model families expose `run` and `mcmc`, the runner is model-agnostic.
   `n_initialization_points` defaults from `model.latent.n_initialization_points`; builders whose model lacks that attribute (HEW) pass it explicitly.

2. **`Candidate`(`runner.py`)** packages a `build: () -> BuiltFit` callable with its `arm`, `config_fields`, and `rt_site_names`.
   The build callable is where a suite's model spec lives.
   The runner only sees `Candidate`s.

3. **`ComparisonSpec`(`comparison.py`)** is the single source of truth for reporting: `arms`, `baseline`, `match_keys`, `metrics`.
   Nothing in `core` hardcodes an arm name.
   A single-arm spec is valid (profile one model, empty comparison table).

4. **`run_comparison`(`run.py`)** is the shared orchestration: given a `list[Candidate]`, a `ComparisonSpec`, and `McmcSettings`, it runs the fit-and-repeat loop, prints the tables, and writes the artifacts.
   A driver supplies the candidates and the spec; it does not write the loop.
   `cli.add_common_args`/`settings_from_args` give every driver the same sampler/output flags, and `env.configure_jax` sets the JAX flags before import, so a suite module is just its build function, its spec, a `build_candidates`, and a thin `main`.

## Control flow

```
driver: build list[Candidate] + ComparisonSpec, settings_from_args(args)
  run_comparison(candidates, spec, settings, suite_name, repeats, output_dir):
    for candidate, repeat:
      fit_candidate(candidate, settings, repeat)
        -> candidate.build() returns BuiltFit
        -> fit_and_measure: model.run(extra_fields=("diverging","num_steps","energy"), **run_kwargs)
           -> compute_fit_metrics(model.mcmc, wall, rt_site_names)
           -> summarize_posterior_parameters(model.mcmc)
           -> FitResult(arm, config_fields, metrics, parameter_summaries, ...)
    print_comparison_tables(results, spec)
    write_results(output_dir, suite_name, results, spec)
```

`FitResult` carries `arm` and `config_fields`.
`config_fields` is the only configuration view reporting reads, which is what lets heterogeneous model families coexist in one result set.
Reporting groups candidates by `match_keys` (resolved from `dataset` plus the flattened `config_fields`), lays arms side by side, and computes each non-baseline arm's benefit ratio against `baseline`.

## Reporting outputs (`reporting.py`)

Per suite, prefixed `<suite>_`: `runs.csv` (one row per fit, union of keys across rows so mixed configs are fine), `candidates.csv` (aggregated over repeats), `comparison.csv` (`<metric>__<arm>` value columns and `<metric>__ratio__<arm>` ratios), `parameters.csv`, `runs.json` (everything plus `arms`/`baseline`/x64 header), `report.md`.
Metric aggregation across repeats is mean by default; `_METRIC_REDUCERS` special-cases divergences (sum) and the worst-case diagnostics (min/max).

## Data layer

`DatasetProvider` (`signals.py`) is a `Protocol` returning `DatasetBundle`s.
`SyntheticProvider` wraps the built-in fixtures; `RealDataProvider` wraps CDC feeds.
Suites and builders depend only on the protocol, so swapping synthetic for real data touches no model code.

## External dependencies

`pyrenew-multisignal` and `cfa-stf-routine-forecasting` are not project dependencies.
Their imports live inside function bodies (`real_data.py` for `cfa.stf.data`; `hew_model.py` for the HEW model and pipeline utils), so `core` imports cleanly without them and only the relevant code path requires them.
`hew_model._ensure_importable` prepends the two checkout paths to `sys.path`; defaults point at `~/github/CDC/...` and are overridable per call or via the suite CLI.

## Invariants and gotchas

- **x64 is mandatory.** In float32 the renewal recursion loses precision and NUTS diverges.
  Each driver calls `env.configure_jax()` before importing `jax`, setting `JAX_ENABLE_X64=true` and the XLA device count (via `setdefault`, so a value you export yourself is honored).
- **The HEW model is run directly, not via the pipeline's `fit_and_save_model`.** That entry point pickles to disk and requests `extra_fields` that omit `diverging` and `energy`, which the benchmark needs for divergence and E-BFMI.
  `build_hew_model` builds the model so the runner can request the diagnostic fields itself.
- **`rt_site_names`differ by model family** `("PopulationInfections::rt_single",)` for PyRenew (the `RT_SITE_NAMES` default), `("rt","rtu_subpop")` for HEW (`HEW_RT_SITE_NAMES`).
  A candidate sets its own.
- **Arms only compare when they share their `match_keys` values**, including `dataset`.
  The `pyrenew_vs_hew` suite passes the same `dataset_name` to both arms so they group.
- **`ComparisonSpec.baseline`must be one of `arms`.** Ratios that are undefined (for example dividing by zero divergences) render blank.
- **`synthetic_hew_export.py`has no external dependencies** by design; it only writes the file layout the production builder reads.
  The export bridge and `build_hew_model` are decoupled: the former writes a `model_dir`, the latter consumes one.

## Extending

- **New model spec** - write a `build() -> BuiltFit` in a suite, wrap it in a `Candidate` via `build_candidates`, and call `run_comparison` from `main`.
  No `core` change.
- **New data source** - implement `DatasetProvider`.
- **New metric** - add a field to `FitMetrics`, a `MetricSpec` to the suite's spec, and a reducer to `_METRIC_REDUCERS` if it is not a mean.
- **New comparison** - declare a `ComparisonSpec`.
