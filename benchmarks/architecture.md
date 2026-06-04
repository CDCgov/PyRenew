# `benchmarks/` architecture (maintainer guide)

## Purpose

Opt-in MCMC model comparisons.
Not part of CI.
A *driver* (a module with a `main()`) fits two or more model candidates on the same data and reports per-candidate diagnostics plus a baseline-relative head-to-head.
A committed driver in `suites/` is a *suite*; the `examples/` drivers are copy-me templates.
Correctness lives in `test/`; this directory is for sampler and model comparisons.

## The one organizing rule

Three layers.
`core/` is model-agnostic machinery.
`models/` holds reusable model builders (a builder takes a config and a `DatasetBundle` and returns a `BuiltFit`).
`suites/` are thin comparison declarations: arms, a `ComparisonSpec`, and a `main`.
A model builder never lives in `core` (it is not model-agnostic) and is not re-specified per suite (suites reuse `models/`).
If suites proliferate, subdivide `suites/` into subpackages.

## Layout

```
benchmarks/
├── core/                  model-agnostic machinery
│   ├── env.py            configure_jax: float64 + XLA device count, set before jax import
│   ├── cli.py            add_common_args, settings_from_args (shared sampler/output flags)
│   ├── data_source.py    add_data_source_args, load_he_bundle (synthetic-vs-real selection)
│   ├── signals.py        SignalSeries, DatasetBundle, DatasetProvider (Protocol)
│   ├── datasets.py       SyntheticProvider over pyrenew/datasets/ fixtures
│   ├── real_data.py      RealDataProvider over CDC NHSN+NSSP (lazy cfa.stf imports)
│   ├── reference_data.py static location populations/names
│   ├── priors.py         benchmark-local priors for real-data builds
│   ├── comparison.py     MetricSpec, ComparisonSpec, DEFAULT_METRICS
│   ├── models.py         BuiltFit, align_weekly_observations (shared machinery)
│   ├── runner.py         FitResult, Candidate, fit_and_measure, fit_candidate, metrics
│   ├── run.py            run_comparison: the shared fit / report / write loop
│   ├── suite.py          Arm, comparison_suite: declarative driver (CLI, load, fit, report)
│   └── reporting.py      spec-driven aggregation + CSV/JSON/Markdown writers
├── models/                reusable model builders
│   ├── he.py             HEModelConfig, build_he_model (hospital + ED visits)
│   └── hew.py            build_hew_model: adapter to the production HEW model
├── suites/                thin comparison declarations
│   ├── rt_params.py       innovation vs state Rt parameterization
│   ├── ed_day_of_week.py  ED day-of-week effect on vs off
│   └── pyrenew_vs_hew.py  production HEW vs PyRenew
├── examples/
│   └── run_prior_regimes.py one structure under several prior regimes (template)
└── results/              output (gitignored)

pyrenew/datasets/synthetic_hew_export.py   write_hew_model_dir / write_synthetic_hew_model_dir (export bridge)
```

## The abstractions that make it work

1. **`BuiltFit`(`models.py`)** wraps any `pyrenew.metaclass.Model` (PyRenew `MultiSignalModel` or the production HEW model) plus the `run_kwargs` for `model.run`.
   Because both model families expose `run` and `mcmc`, the runner is model-agnostic.
   `n_initialization_points` defaults from `model.latent.n_initialization_points`; builders whose model lacks that attribute (HEW) pass it explicitly.

2. **Model builders (`models/`)** turn a config and a `DatasetBundle` into a `BuiltFit`.
   `models/he.py:build_he_model` builds the H+E PyRenew model from an `HEModelConfig` (structural axes plus every prior as a field); `models/hew.py:build_hew_model` adapts the production HEW model from a model directory.

3. **`Candidate`(`runner.py`)** packages a `build: () -> BuiltFit` callable with its `arm`, `config_fields`, and `rt_site_names`.
   The runner only sees `Candidate`s.

4. **`Arm`+ `comparison_suite`(`suite.py`)** are the suite-facing layer.
   An `Arm` names a variant and carries either a config (assembled by a shared `build_fn`) or its own per-arm `build` for a different model family.
   `comparison_suite(spec, arms, build_fn)` returns a `main` that supplies the CLI (including the shared `--data-source` flags), sampler setup, data loading via `load_he_bundle`, the `--dry-run-data` short-circuit, and `run_comparison`; it resolves each `Arm` to a `Candidate`.
   A suite module is then its arms, its spec, and `main = comparison_suite(...)`.

5. **`ComparisonSpec`(`comparison.py`)** is the single source of truth for reporting: `arms`, `baseline`, `match_keys`, `metrics`.
   Nothing in `core` hardcodes an arm name.
   A single-arm spec is valid (profile one model, empty comparison table).

## Control flow

```
main = comparison_suite(spec, arms, build_fn):
  parse args; numpyro x64 + device count; bundle = load_he_bundle(args)
  candidates = [arm.to_candidate(bundle, build_fn) for arm in arms]
  run_comparison(candidates, spec, settings, comparison_name, repeats, output_dir):
    for candidate, repeat:
      fit_candidate(candidate, settings, repeat)
        -> candidate.build() returns BuiltFit
        -> fit_and_measure: model.run(extra_fields=("diverging","num_steps","energy"), **run_kwargs)
           -> compute_fit_metrics(model.mcmc, wall, rt_site_names)
           -> summarize_posterior_parameters(model.mcmc)
           -> FitResult(arm, config_fields, metrics, parameter_summaries, ...)
    print_comparison_tables(results, spec)
    write_results(output_dir, comparison_name, results, spec)
```

`FitResult` carries `arm` and `config_fields`.
`config_fields` is the only configuration view reporting reads, which is what lets heterogeneous model families coexist in one result set.
Reporting groups candidates by `match_keys` (resolved from `dataset` plus the flattened `config_fields`), lays arms side by side, and computes each non-baseline arm's benefit ratio against `baseline`.

## Reporting outputs (`reporting.py`)

Per driver, prefixed `<comparison_name>_`: `runs.csv` (one row per fit, union of keys across rows so mixed configs are fine), `candidates.csv` (aggregated over repeats), `comparison.csv` (`<metric>__<arm>` value columns and `<metric>__ratio__<arm>` ratios), `parameters.csv`, `runs.json` (everything plus `arms`/`baseline`/x64 header, keyed `suite` for back-compat), `report.md`.
`comparison_name` is just the comparison's output label, so an `examples/` driver carries one (e.g. `prior_regimes_*`) without being a suite.
Metric aggregation across repeats is mean by default; `_METRIC_REDUCERS` special-cases divergences (sum) and the worst-case diagnostics (min/max).

## Data layer

`DatasetProvider` (`signals.py`) is a `Protocol` returning `DatasetBundle`s.
`SyntheticProvider` wraps the built-in fixtures; `RealDataProvider` wraps CDC feeds.
Suites select between them with the shared `--data-source` flags (`data_source.py`); `load_he_bundle` returns the chosen bundle.
Suites and builders depend only on the protocol, so swapping synthetic for real data touches no model code.

## External dependencies

`pyrenew-multisignal` and `cfa-stf-routine-forecasting` are not project dependencies.
Their imports live inside function bodies (`real_data.py` for `cfa.stf.data`; `models/hew.py` for the HEW model and pipeline utils), so importing those modules works without them and only the relevant code path requires them.
`models.hew._ensure_importable` prepends the two checkout paths to `sys.path`; defaults point at `~/github/CDC/...` and are overridable per call or via the suite CLI.

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

- **New comparison (common case)** - declare arms (configs differing on one axis) and a `ComparisonSpec`, then `main = comparison_suite(spec, arms, build_fn)`.
  No `core` or `models/` change.
- **New model family** - add a builder to `models/` taking a config and a `DatasetBundle` and returning a `BuiltFit`; an arm of that family carries its own `build` (as `pyrenew_vs_hew`'s HEW arm does).
- **New data source** - implement `DatasetProvider`.
- **New metric** - add a field to `FitMetrics`, a `MetricSpec` to the spec, and a reducer to `_METRIC_REDUCERS` if it is not a mean.
