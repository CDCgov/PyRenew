# PyRenew benchmarks

Opt-in MCMC performance benchmarks which allow for systematic comparison of modeling choices: model parameterization and/or prior choice which can be run on synthetic or vintaged data from CDC data feeds.

Each comparison is run by a "driver": a Python module with a `main()` that is run from the repository root.
Committed comparisons live in `benchmarks/suites/` (a comparison is necessarily a *suite* of sampler runs).
Templates to copy for a new comparison live in `benchmarks/examples/`.

These benchmarks are not part of CI; the unit and integration tests in `test/` provide correctness checks.

## Layout

```
benchmarks/
├── core/
│   ├── env.py          configure_jax: float64 + XLA device count, before jax import
│   ├── cli.py          add_common_args / settings_from_args: shared sampler + output flags
│   ├── data_source.py  add_data_source_args / load_he_bundle: shared synthetic-vs-real selection
│   ├── signals.py      SignalSeries, DatasetBundle, DatasetProvider
│   ├── datasets.py     SyntheticProvider over pyrenew/datasets/
│   ├── real_data.py    RealDataProvider over CDC NHSN + NSSP feeds
│   ├── reference_data.py Static location names and populations
│   ├── priors.py       benchmark-local priors for real-data builds
│   ├── comparison.py   ComparisonSpec / MetricSpec: declares arms, baseline, metrics
│   ├── models.py       BuiltFit + align_weekly_observations (shared machinery)
│   ├── hew_model.py    adapter wrapping the production HEW model as a BuiltFit
│   ├── runner.py       Candidate, fit_and_measure/fit_candidate
│   ├── run.py          run_comparison: the shared fit / report / write loop
│   └── reporting.py    spec-driven stdout tables and CSV / JSON / Markdown writers
├── suites/
│   ├── rt_params.py    centered vs non-centered weekly Rt parameterization
│   └── pyrenew_vs_hew.py  production HEW vs PyRenew
├── examples/
│   └── run_prior_regimes.py  template: one structure under several prior regimes
└── results/            output (gitignored)
```

A driver asks a dataset provider for a bundle, builds one or more candidate models, and hands them to `run_comparison`, which fits each candidate and collects metrics.
The `DatasetProvider` protocol in `core/signals.py` lets reporting-input providers replace `SyntheticProvider` without touching the driver.

## Kinds of comparisons

A benchmark compares candidates that differ in one of two ways:

- **Model structure.** Different model specifications on the same data: a parameterization, a structural choice, or a whole model family.
  Benchmark suite `rt_params` compares the `innovation` and `state` parameterizations of the weekly $\mathcal{R}(t)$ process; `pyrenew_vs_hew` compares a PyRenew `MultiSignalModel` against the production HEW model.

- **Priors.** The same model structure under different prior sets (each set is a `regime`).
  The `prior_regimes` example (`benchmarks/examples/run_prior_regimes.py`) fits one fixed structure under several regimes and compares how each samples; see `prior_regimes.md`.

A `ComparisonSpec` expresses a comparison of either kind.
Its `arms` are the candidates compared side by side (as in a trial's treatment arms), `baseline` is the arm the others are rated against (the control), and `match_keys` are the fields that must be equal for two fits to form a comparable group, so whatever you are not varying (and the dataset) is held fixed.
To vary both at once, put one kind of difference in `arms` and pin the other through `match_keys`: `rt_params` does this, holding a fixed-hyperparameter prior regime equal while comparing parameterizations within it.

Model structure lives in the build function (structure is code); priors live in what a build function consumes (priors are data).
`prior_regimes.md` develops the prior comparison in full.

## rt_params suite

Compares the `innovation` (non-centered, NCP) and `state` (centered, CP) parameterizations of the inner `DifferencedAR1` weekly $\mathcal{R}(t)$ process, on the H+E model: weekly-aggregated hospital admissions plus daily ED visits.
Each fit uses one parameterization; the suite always runs both so the matched pair can be compared.

### Run

```bash
python -m benchmarks.suites.rt_params --quick
```

`--quick` overrides the sampler to 50 warmup, 50 samples, 1 chain.
Drop it for a full run.

```bash
python -m benchmarks.suites.rt_params --prior both --repeats 3
```

Useful options:

  | Option                                          | Effect                                                                                                                                                    |
  | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `--data-source`                                 | `synthetic` (built-in fixtures) or `real` (CDC-internal NHSN/NSSP feeds; requires `cfa-stf-routine-forecasting` access and `--as-of`).                    |
  | `--disease <name>`                              | Disease for `--data-source real`: `COVID-19`, `Influenza`, or `RSV`.                                                                                      |
  | `--location <abbr>`                             | Location abbreviation for `--data-source real`, e.g. `US` or `CA`.                                                                                        |
  | `--as-of YYYY-MM-DD`                            | Vintage date for `--data-source real`. Required for real data.                                                                                            |
  | `--training-days N`                             | Training window length for `--data-source real`. Default: 150.                                                                                            |
  | `--omit-last-days N`                            | Trailing days omitted from `--data-source real` to buffer right truncation. Default: 2.                                                                   |
  | `--dry-run-data`                                | Load and summarize selected data, then exit before model fitting. Useful for checking real-data access and signal noise.                                  |
  | `--prior <kind>`                                | `tight` (sd=0.01, autoreg=0.9), `loose` (sd=0.10, autoreg=0.5), `both`, or an explicit `sd,autoreg` pair (e.g. `0.05,0.7`). Repeatable. Default: `tight`. |
  | `--repeats N`                                   | Refit each cell `N` times with `seed + i` to estimate sampler noise.                                                                                      |
  | `--num-warmup`, `--num-samples`, `--num-chains` | NUTS controls. `--num-chains` defaults to `min(4, os.cpu_count())`.                                                                                       |
  | `--seed`                                        | Base seed (default 42).                                                                                                                                   |
  | `--output-dir`                                  | Where to write artifacts. Default `benchmarks/results/`.                                                                                                  |
  | `--no-write`                                    | Skip artifact files; print summary only.                                                                                                                  |

At startup the driver calls `core/env.py:configure_jax()`, which sets `XLA_FLAGS=--xla_force_host_platform_device_count=N` (where `N = min(8, os.cpu_count())`) so JAX exposes enough logical devices for parallel chains, and `JAX_ENABLE_X64=true`.
Both are set before `jax` is imported.
If you set either variable yourself before invocation, it is honored.
x64 is required: in float32 the renewal recursion loses precision and NUTS diverges (a full chain diverged at 500/500/4 in float32, none under x64).

### Real data on CDC infrastructure

The `--data-source` flags are shared core machinery (`core/data_source.py`), not specific to `rt_params`: every H+E suite registers them through `add_data_source_args` and loads its bundle through `load_he_bundle`, so the same `--data-source real ...` invocation shown below works for `pyrenew_vs_hew` as well.
For `pyrenew_vs_hew`, the HEW arm is written into a production model directory from the same bundle the PyRenew arm consumes (`core/hew_model.py:write_hew_model_dir_from_bundle`), so both arms fit identical real feeds.

Real-data mode is intended for CDC environments that can import `cfa-stf-routine-forecasting` and access the internal feeds used by `cfa.stf.data`.
PyRenew does not depend on those internal packages for normal use; the `cfa.stf.*` imports happen only when `--data-source real` loads a bundle.

Start with a data-only dry run:

```bash
python -m benchmarks.suites.rt_params \
  --data-source real \
  --disease RSV \
  --location US \
  --as-of 2025-01-15 \
  --training-days 150 \
  --omit-last-days 2 \
  --dry-run-data
```

This fetches NHSN weekly hospital admissions and NSSP daily ED visits, prints date ranges, missingness, and basic count summaries, then exits before model building or MCMC.

Then run a smoke benchmark:

```bash
python -m benchmarks.suites.rt_params \
  --data-source real \
  --disease RSV \
  --location US \
  --as-of 2025-01-15 \
  --training-days 150 \
  --omit-last-days 2 \
  --quick
```

The H+E real-data builder uses benchmark-local priors (`core/priors.py`) mirroring the production prior subset needed for initial infections and ED day-of-week effects.
Location metadata and population totals are static benchmark inputs in `core/reference_data.py`.
Generation interval and infection-to-observation delay PMFs are pulled from the CDC NNH parameter catalog through `cfa.stf.data`, so they remain disease-specific and vintage-aware.
Real-data mode currently does not apply ED right truncation PMFs; use `--omit-last-days` to leave a reporting buffer.

### Output files

Written to `--output-dir` with prefix `rt_params_`:

  | File                       | Contents                                                                                                                  |
  | -------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
  | `rt_params_runs.csv`       | One row per fit, with full config and metrics.                                                                            |
  | `rt_params_candidates.csv` | One row per candidate, aggregated over repeats, carrying `arm` and the candidate's config fields.                         |
  | `rt_params_comparison.csv` | One row per comparable group, with `<metric>__<arm>` value columns and `<metric>__ratio__<arm>` baseline-relative ratios. |
  | `rt_params_parameters.csv` | One row per scalar posterior site element per fit, with posterior mean, ESS, and R-hat.                                   |
  | `rt_params_runs.json`      | All of the above, site-level parameter ESS summaries, and a header (suite name, arms, baseline, x64 flag, timestamp).     |
  | `rt_params_report.md`      | Compact Markdown report with candidate, comparison, and per-site parameter ESS tables.                                    |

Column convention: `<metric>__<arm>` columns carry the per-arm values, and `<metric>__ratio__<arm>` columns are the arm's benefit relative to the spec baseline.
For higher-is-better metrics such as ESS-per-second, the ratio is `arm / baseline`.
For lower-is-better metrics such as wall time, the ratio is `baseline / arm`.
In all cases, a ratio `> 1` favors the listed arm over the baseline.
The baseline arm has no ratio column; ratios are omitted (blank) when undefined, such as dividing by zero divergences.

### Reading the metrics

Per fit:

- **Wall time**: total seconds for warmup + sampling, after JIT, with `jax.block_until_ready` so the work is fully complete.
- **ESS/s Rt (median / min)**: effective samples per wall-second on the Rt trajectory.
  Median summarizes typical timepoints; min identifies the worst-mixing timepoint that limits downstream inference.
- **Divergences**: total NUTS divergences across all chains and draws.
  A saturated tree depth can mask divergences; read with tree depth.
- **Tree depth (mean / max)**: log2 of NUTS leapfrog steps.
  NumPyro defaults to `max_tree_depth=10`.
  A mean near the ceiling indicates the sampler is running out of budget per draw.
- **E-BFMI (min)**: minimum across chains of the energy Bayesian fraction of missing information.
  Heuristic thresholds: >=0.3 acceptable, <0.3 warning, <0.1 strong pathology indicator.
- **R-hat Rt (max)**: max split R-hat across timepoints of the Rt trajectory.
  Requires more than one chain.

Candidate summaries average time and ESS metrics across repeats, sum divergences, and keep worst-case diagnostics: maximum tree depth, minimum E-BFMI, and maximum R-hat.

### Suite design

`rt_params` varies both model structure and priors at once.
Its arms are the **model structure**: the `innovation` (non-centered) and `state` (centered) modes of the inner `DifferencedAR1`.
Its `--prior` option steps a **prior** regime that fixes the weekly per-step innovation SD $\sigma$ and the autoregressive coefficient $\phi$ to chosen values: `tight` $(\sigma = 0.01, \phi = 0.9)$, `loose` $(\sigma = 0.10, \phi = 0.5)$, or an explicit pair.
The regime is a match key, held equal within each comparison, so the two parameterizations are compared within a regime rather than across regimes.
The cumulative variance of $\log \mathcal{R}(T)$ is far more sensitive to $\phi$ than to $\sigma$.

The latent $\mathcal{R}(t)$ runs at weekly cadence, matching the production HEW model and the weekly forecasting setting.
Production treats both hyperparameters as inferred (`eta_sd ~ TruncatedNormal(0.15, 0.05)`, `autoreg_rt ~ Beta(2, 40)`); `rt_params` fixes them so the comparison isolates the parameterization.
To make the priors themselves the object of study rather than holding them fixed, use the `prior_regimes` example.

## prior_regimes example

Fits one fixed H+E structure under a set of prior regimes and compares how each samples.
Run from the repository root:

```bash
python -m benchmarks.examples.run_prior_regimes --quick
```

`--quick` is a smoke run (50 warmup, 50 samples, 1 chain); drop it for a full run.
It accepts the same sampler and output flags as `rt_params`: `--num-warmup`, `--num-samples`, `--num-chains`, `--repeats`, `--seed`, `--output-dir` (default `benchmarks/results/`, prefix `prior_regimes_`), `--no-write`, and `--progress-bar`.

As shipped it has one regime (`example`), so it profiles that single model: per-candidate and per-site ESS tables plus the written artifacts, with an empty comparison table.
A comparison appears once you add a second regime.
The regimes and the model structure are yours to edit: copy `benchmarks/examples/run_prior_regimes.py` to another file under `benchmarks/examples/` (everything there but the committed examples is gitignored) and change `REGIMES`.
See `prior_regimes.md` for the full workflow, including how each run records the exact priors it used.

## Adding a benchmark

1. Write a build function in your suite that returns a `BuiltFit`.
   Model construction lives in the suite, not in `core`.
   The model may be any `pyrenew.metaclass.Model` exposing `run` and `mcmc`, so the build function can assemble a PyRenew `MultiSignalModel` or wrap the production HEW model via `core/hew_model.py`.
   `core/models.py` provides the shared `BuiltFit` container and `align_weekly_observations` helper.

2. If the model needs a new dataset, add a builder to `benchmarks/core/datasets.py` and expose it through `SyntheticProvider`.

3. Define a `ComparisonSpec` in the suite: its `arms`, `baseline`, `match_keys`, and `metrics` are the single source of truth for reporting.
   A single-arm spec is allowed; it profiles one model and emits an empty comparison table.

4. Add or extend a suite module in `benchmarks/suites/` following the shared driver shape:
   - call `core/env.py:configure_jax()` before importing `jax`;
   - write a `build_candidates(...)` that wraps each build function in a `Candidate` (with its `arm` and `config_fields`);
   - in `main()`, register flags with `core/cli.py:add_common_args`, build sampler settings with `settings_from_args`, then call `core/run.py:run_comparison` with the candidates and the spec.
     `run_comparison` runs the fit loop and the reporting; the suite supplies only the model construction and the spec.

## Wiring real data

`benchmarks.core.signals.DatasetProvider` is a `Protocol`.
Implement it for a reporting source and pass the provider to the suite; the model builder and runner do not change.
The expected payload is a `DatasetBundle` whose `signals` mapping carries one `SignalSeries` per observation source.

`benchmarks/core/real_data.py` provides `RealDataProvider`, a concrete implementation over the CDC NHSN (weekly hospital admissions) and NSSP (daily ED visits) feeds.
Construct it with a mapping of dataset name to `RealDataSpec` (disease, location, `as_of` vintage, training window) and request bundles by name, exactly as with `SyntheticProvider`.

`RealDataProvider` reads live H+E feeds through `cfa.stf.data` (from `cfa-stf-routine-forecasting`) and requires valid Azure credentials at call time.
It does not call the R `forecasttools` package for benchmark setup; location names and populations come from `benchmarks/core/reference_data.py`.
PyRenew intentionally does **not** declare `cfa-stf-routine-forecasting` as a dependency: the `cfa.stf.*` imports live inside the provider's function bodies, so `real_data.py` imports cleanly without it and the synthetic path is unaffected.
To use `RealDataProvider`, install `cfa-stf-routine-forecasting` into your own environment separately.
