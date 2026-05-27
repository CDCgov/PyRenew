# PyRenew benchmarks

Opt-in MCMC performance experiments.
The suite is a CLI entry point under `benchmarks/suites/`.
Run from the repository root.

Benchmarks are not part of CI.
Use `test/` for correctness checks and this suite for sampler comparisons.

## Layout

```
benchmarks/
├── core/
│   ├── signals.py      SignalSeries, DatasetBundle, DatasetProvider
│   ├── datasets.py     SyntheticProvider over pyrenew/datasets/
│   ├── real_data.py    RealDataProvider over CDC NHSN + NSSP feeds
│   ├── priors.py       benchmark-local priors for real-data builds
│   ├── models.py       H+E model builder (weekly hospital + daily ED)
│   ├── runner.py       fit_and_measure and ArviZ-free FitMetrics computation
│   └── reporting.py    stdout tables and CSV / JSON / Markdown writers
├── suites/
│   └── rt_params.py    centered vs non-centered weekly Rt parameterization
├── diagnose.py         single-fit diagnostic harness
└── results/            output (gitignored)
```

The suite asks the dataset provider for the H+E bundle, builds the model under each parameterization, and the runner fits the model and collects metrics.
The `DatasetProvider` protocol in `core/signals.py` is the seam where real reporting inputs replace `SyntheticProvider` without touching the suite.

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

On import, the suite sets `XLA_FLAGS=--xla_force_host_platform_device_count=N` (where `N = min(8, os.cpu_count())`) so JAX exposes enough logical devices for parallel chains, and `JAX_ENABLE_X64=true`.
If you set either variable yourself before invocation, it is honored.
x64 is required: in float32 the renewal recursion loses precision and NUTS diverges (a full chain diverged at 500/500/4 in float32, none under x64).

### Real data on CDC infrastructure

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

The H+E real-data builder uses benchmark-local priors (`core/priors.py`) mirroring the production prior subset needed for initial infections and ED day-of-week effects; PMFs, right truncation, and population are pulled from the `cfa.stf` data helpers.

### Output files

Written to `--output-dir` with prefix `rt_params_`:

  | File                       | Contents                                                                                                         |
  | -------------------------- | ---------------------------------------------------------------------------------------------------------------- |
  | `rt_params_runs.csv`       | One row per fit, with full config and metrics.                                                                   |
  | `rt_params_candidates.csv` | One row per parameterization, averaged over repeats.                                                             |
  | `rt_params_pairs.csv`      | One row per matched state-vs-innovation pair, with `<metric>_innov`, `<metric>_state`, `<metric>_ratio` columns. |
  | `rt_params_runs.json`      | All of the above plus a header (suite name, x64 flag, timestamp).                                                |
  | `rt_params_report.md`      | Compact Markdown report (per-parameterization table and pairwise table).                                         |

Column convention: `_innov` and `_state` carry the per-side values, and `_ratio` is `state / innovation`.
Wall-time `_ratio > 1` means state is slower.
ESS-per-second `_ratio > 1` means state mixes faster per second.

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

### Suite design

The suite varies two axes:

1. **Parameterization**: `innovation` (non-centered) and `state` (centered) modes of the inner `DifferencedAR1`.
2. **Prior regime**: tight $(\sigma = 0.01, \phi = 0.9)$ or loose $(\sigma = 0.10, \phi = 0.5)$, where $\sigma$ is the weekly per-step innovation SD and $\phi$ the autoregressive coefficient.
   The cumulative variance of $\log \mathcal{R}(T)$ is far more sensitive to $\phi$ than to $\sigma$.

The latent $\mathcal{R}(t)$ runs at weekly cadence, matching the production HEW model and the weekly forecasting setting.
Production treats both hyperparameters as inferred (`eta_sd ~ TruncatedNormal(0.15, 0.05)`, `autoreg_rt ~ Beta(2, 40)`); the benchmark fixes them to isolate the parameterization axis.

## Diagnostics

`benchmarks/diagnose.py` builds one model on one dataset under one config and reports the data-side summary, the priors `build_he_model` selects and the initial scale they imply, prior-predictive ranges, whether the initial potential energy and gradient (under the sampler's `init_to_sample` strategy) are finite, and optionally a short NUTS run with its divergence count.

Its `--real-i0`, `--real-dow`, `--real-trunc`, and `--all-real` flags force the real-data priors onto the synthetic bundle one at a time, so a real-data sampler failure can be bisected off the CDC VM.
`--data-source real` runs the same diagnostics against a live bundle.

```bash
python -m benchmarks.diagnose --all-real --mcmc
python -m benchmarks.diagnose --real-i0
```

## Adding a benchmark

1. Add a model builder to `benchmarks/core/models.py` that returns a `BuiltFit`.
   Reuse `BuildConfig` if the new model fits the existing axes.
2. If the model needs a new dataset, add a builder to `benchmarks/core/datasets.py` and expose it through `SyntheticProvider`.
3. Add or extend a suite module in `benchmarks/suites/` with a `main()` CLI.
   Use `fit_and_measure`, `print_pairwise_tables`, and `write_results` from `benchmarks.core`.

## Wiring real data

`benchmarks.core.signals.DatasetProvider` is a `Protocol`.
Implement it for a reporting source and pass the provider to the suite; the model builder and runner do not change.
The expected payload is a `DatasetBundle` whose `signals` mapping carries one `SignalSeries` per observation source.

`benchmarks/core/real_data.py` provides `RealDataProvider`, a concrete implementation over the CDC NHSN (weekly hospital admissions) and NSSP (daily ED visits) feeds.
Construct it with a mapping of dataset name to `RealDataSpec` (disease, location, `as_of` vintage, training window) and request bundles by name, exactly as with `SyntheticProvider`.

`RealDataProvider` reads its feeds through `cfa.stf.data` and `cfa.stf.forecasttools` (from `cfa-stf-routine-forecasting`), and requires valid Azure credentials at call time.
PyRenew intentionally does **not** declare that package as a dependency: the `cfa.stf.*` imports live inside the provider's function bodies, so `real_data.py` imports cleanly without it and the synthetic path is unaffected.
To use `RealDataProvider`, install `cfa-stf-routine-forecasting` into your own environment separately.
