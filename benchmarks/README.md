# PyRenew benchmarks

Opt-in MCMC performance benchmarks which allow for systematic comparison of modeling choices: model parameterization and/or prior choice which can be run on synthetic or vintaged data from CDC data feeds.

Each comparison is run by a "driver": a Python module with a `main()` that is run from the repository root.
Committed comparisons live in `benchmarks/suites/` (a comparison is necessarily a *suite* of sampler runs).
Templates to copy for a new comparison live in `benchmarks/examples/`.

These benchmarks are not part of CI; the unit and integration tests in `test/` provide correctness checks.

## Layout

```
benchmarks/
├── core/                 model-agnostic machinery
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
│   ├── runner.py       Candidate, fit_and_measure/fit_candidate
│   ├── run.py          run_comparison: the shared fit / report / write loop
│   ├── suite.py        Arm + comparison_suite: declarative driver (CLI, load, fit, report)
│   └── reporting.py    spec-driven stdout tables and CSV / JSON / Markdown writers
├── models/               reusable model builders
│   ├── he.py           HEModelConfig + build_he_model (hospital + ED visits)
│   └── hew.py          build_hew_model: adapter wrapping the production HEW model
├── suites/               thin comparison declarations
│   ├── rt_params.py    centered vs non-centered weekly Rt parameterization
│   ├── ed_day_of_week.py  ED day-of-week effect on vs off
│   └── pyrenew_vs_hew.py  production HEW vs PyRenew
├── examples/
│   └── run_prior_regimes.py  template: one structure under several prior regimes
└── results/            output (gitignored)
```

The three layers separate concerns: `core/` is model-agnostic machinery, `models/` holds reusable model builders, and `suites/` are thin declarations.
A suite declares its arms (each an `Arm` carrying a model config) and its `ComparisonSpec`, then hands them to `core/suite.py:comparison_suite`, which supplies the CLI, sampler setup, data loading, and the fit / report loop.
A model builder is signal-scoped: `models/he.py:build_he_model` reads a bundle's hospital and ED-visits signals; a model over other signals (e.g. wastewater) is a new builder, not a new field on `HEModelConfig`.
The `DatasetProvider` protocol in `core/signals.py` lets reporting-input providers replace `SyntheticProvider` without touching any suite.

## Kinds of comparisons

A benchmark compares candidates that differ in one of two ways:

- **Model structure.** Different model specifications on the same data: a parameterization, a structural choice, or a whole model family.
  Benchmark suite `rt_params` compares the `innovation` and `state` parameterizations of the weekly $\mathcal{R}(t)$ process; `pyrenew_vs_hew` compares a PyRenew `MultiSignalModel` against the production HEW model.

- **Priors.** The same model structure under different prior sets (each set is a `regime`).
  The `prior_regimes` example (`benchmarks/examples/run_prior_regimes.py`) fits one fixed structure under several regimes and compares how each samples; see `prior_regimes.md`.

Model structure lives in the build function (structure is code); priors live in what a build function consumes (priors are data).

A `ComparisonSpec` expresses a comparison of either kind.
Its `arms` are the candidates compared side by side (as in a trial's treatment arms), `baseline` is the arm the others are rated against (the control), and `match_keys` are the fields that must be equal for two fits to form a comparable group, so whatever you are not varying (and the dataset) is held fixed.

To vary both at once, put one kind of difference in `arms` and pin the other through `match_keys`.
For example, the comparison in file `suite/rt_params.py` does this by holding a fixed-hyperparameter prior regime equal while comparing parameterizations within it.

```python
SPEC: ComparisonSpec = ComparisonSpec(
    name="rt_params",
    arms=("innovation", "state")
    baseline="innovation",
    match_keys=("dataset", "innovation_sd", "autoreg"),
```

## Running a comparison

#### Data selection

  | Option               | Effect                                                                     |
  | -------------------- | -------------------------------------------------------------------------- |
  | `--data-source`      | `synthetic` (built-in fixtures) or `real` (CDC-internal NHSN/NSSP feeds)   |
  | `--disease <name>`   | `real` only: : `COVID-19`, `Influenza`, or `RSV`.                          |
  | `--location <abbr>`  | `real` only: Location abbreviation, e.g. `US` or `CA`.                     |
  | `--as-of YYYY-MM-DD` | `real` only: Vintage date                                                  |
  | `--training-days N`  | `real` only: Training window length.  Default: 150.                        |
  | `--omit-last-days N` | `real` only: Trailing days omitted to buffer right truncation. Default: 2. |
  | `--dry-run-data`     | Load and summarize selected data, then exit before model fitting.          |

#### Run controls

  | Option                                          | Effect                                                               |
  | ----------------------------------------------- | -------------------------------------------------------------------- |
  | `--repeats N`                                   | Refit each cell `N` times with `seed + i` to estimate sampler noise. |
  | `--num-warmup`, `--num-samples`, `--num-chains` | NUTS controls. `--num-chains` defaults to `min(4, os.cpu_count())`.  |
  | `--seed`                                        | Base seed (default 42).                                              |
  | `--quick`                                       | Shorthand for `--num-warmup 50 --num-samples 50                      |

#### Output controls

  | Option         | Effect                                                   |
  | -------------- | -------------------------------------------------------- |
  | `--output-dir` | Where to write artifacts. Default `benchmarks/results/`. |
  | `--no-write`   | Skip artifact files; print summary only.                 |

At startup the driver calls `core/env.py:configure_jax()`, which sets `XLA_FLAGS=--xla_force_host_platform_device_count=N` (where `N = min(8, os.cpu_count())`) so JAX exposes enough logical devices for parallel chains, and `JAX_ENABLE_X64=true`.
Both are set before `jax` is imported.
If you set either variable yourself before invocation, it is honored.
x64 is required: in float32 the renewal recursion loses precision and NUTS diverges (a full chain diverged at 500/500/4 in float32, none under x64).

Individual comparisons can add further command-line controls.
For example, the `rt_params` comparison suite varies both model structure and priors at once.
Its arms are the **model structure**: the `innovation` (non-centered) and `state` (centered) modes of the inner `DifferencedAR1` temporal process for the latent infection renewal equation.
Its `--prior` option steps a **prior** regime that fixes the weekly per-step innovation SD $\sigma$ and the autoregressive coefficient $\phi$ to chosen values: `tight` $(\sigma = 0.01, \phi = 0.9)$, `loose` $(\sigma = 0.10, \phi = 0.5)$, or an explicit pair.
The regime is a match key, held equal within each comparison, so the two parameterizations are compared within a regime rather than across regimes.

### Real data on CDC infrastructure

The `--data-source` flags are shared core machinery (`core/data_source.py`), not specific to `rt_params`: every H+E suite registers them through `add_data_source_args` and loads its bundle through `load_he_bundle`, so the same `--data-source real ...` invocation shown below works for `pyrenew_vs_hew` as well.
For `pyrenew_vs_hew`, the HEW arm is written into a production model directory from the same bundle the PyRenew arm consumes (`models/hew.py:write_hew_model_dir_from_bundle`), so both arms fit identical real feeds.

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

## Adding a benchmark

The directory `examples` contains file `run_prior_regimes.py` which fits one fixed H+E structure under a set of prior regimes and compares how each samples.

As shipped it has one regime (`example`), so it profiles that single model: per-candidate and per-site ESS tables plus the written artifacts, with an empty comparison table.
A comparison appears once you add a second regime.
The regimes and the model structure are yours to edit: copy `benchmarks/examples/run_prior_regimes.py` to another file under `benchmarks/examples/` (everything there but the committed examples is gitignored) and change `REGIMES`.
See `prior_regimes.md` for the full workflow, including how each run records the exact priors it used.

Most new comparisons reuse an existing model builder and differ on one axis, so the suite is a thin declaration.
The whole of `suites/ed_day_of_week.py`:

```python
from benchmarks.core.comparison import DEFAULT_METRICS, ComparisonSpec
from benchmarks.core.suite import comparison_suite
from benchmarks.models.he import HEModelConfig, build_he_model, he_arm

SPEC = ComparisonSpec(
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
```

(The suite also calls `core/env.py:configure_jax()` before importing `jax`; see an existing suite.)

To add a comparison:

1. **Pick or write a model builder.** If an existing builder in `benchmarks/models/` (e.g. `he.py`) covers your signals, reuse it and vary its config.
   If you need a different signal set (e.g. wastewater), add a new builder module there; a builder takes a config and a `DatasetBundle` and returns a `BuiltFit`.
   Model construction lives in `models/`, never in `core/`.

2. **Declare the arms.** Each `Arm` carries a config consumed by the shared builder, or its own `build` callable for an arm of a different model family (as `pyrenew_vs_hew`'s HEW arm does).
   `Arm.config_fields` label and group the candidate in reports; `he_arm` fills curated fields for H+E configs.

3. **Define the `ComparisonSpec`:** `arms`, `baseline`, `match_keys`, and `metrics` are the single source of truth for reporting.
   A single-arm spec is allowed; it profiles one model.

4. **Wire `main`:** `main = comparison_suite(spec, arms, build_fn)`.
   Pass `arms` as a `(args, bundle) -> list[Arm]` factory when the arms depend on CLI options (a prior sweep), and `add_args=` to register suite-specific flags (`rt_params` does both).
   `comparison_suite` supplies the CLI (including the shared `--data-source` flags), sampler setup, data loading, and the fit / report loop.

5. If the model needs a new synthetic dataset, add a builder to `benchmarks/core/datasets.py` and expose it through `SyntheticProvider`.

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
