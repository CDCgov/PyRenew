# PyRenew benchmarks

Opt-in MCMC performance experiments. Each suite is a CLI entry point under
`benchmarks/suites/`. Run from the repository root.

Benchmarks are not part of CI. Use `test/` for correctness checks and these
suites for runtime comparisons.

## Layout

```
benchmarks/
├── core/
│   ├── signals.py      SignalSeries, DatasetBundle, DatasetProvider
│   ├── datasets.py     SyntheticProvider over pyrenew/datasets/
│   ├── models.py       model builders (H+E, subpop hospital+wastewater)
│   ├── metrics.py      ArviZ-free FitMetrics computation
│   ├── runner.py       fit_and_measure
│   └── reporting.py    stdout tables and CSV / JSON / Markdown writers
├── suites/
│   └── rt_params.py    innovation vs state Rt parameterization
└── results/            output (gitignored)
```

A suite picks a model builder, the builder asks the dataset provider for the
bundle it needs, and the runner fits the model and collects metrics. The
signal interface in `core/signals.py` is the seam where real reporting
inputs can later replace `SyntheticProvider` without touching the suites.

## rt_params suite

Compares the `innovation` and `state` parameterizations of the inner
`DifferencedAR1` Rt process.

### Run

```bash
python -m benchmarks.suites.rt_params --quick
```

`--quick` overrides the sampler to 50 warmup, 50 samples, 1 chain. Drop it
for a full run.

```bash
python -m benchmarks.suites.rt_params \
  --candidate he --prior both --repeats 3
```

Useful options:

| Option | Effect |
|---|---|
| `--candidate <name>` | One candidate per use. Repeat for several. Special names: `all`, `he`, `subpop`. |
| `--prior <kind>` | `tight` (sd=0.01, autoreg=0.9), `loose` (sd=0.10, autoreg=0.5), `both`, or an explicit `sd,autoreg` pair (e.g. `0.05,0.7`). Repeatable. Default: `tight`. |
| `--repeats N` | Refit each cell `N` times with `seed + i` to estimate sampler noise. |
| `--num-warmup`, `--num-samples`, `--num-chains` | NUTS controls. `--num-chains` defaults to `min(4, os.cpu_count())`. |
| `--seed` | Base seed (default 42). |
| `--output-dir` | Where to write artifacts. Default `benchmarks/results/`. |
| `--no-write` | Skip artifact files; print summary only. |
| `--no-x64` | Disable JAX 64-bit precision (enabled by default). |

On import, the suite sets `XLA_FLAGS=--xla_force_host_platform_device_count=N` (where `N = min(8, os.cpu_count())`) so JAX exposes enough logical devices for parallel chains. If you set `XLA_FLAGS` yourself before invocation, it is honored.

### Candidate names

H+E models (`pyrenew.latent.PopulationInfections`):

```
he_<rt_cadence>_<parameterization>
he_daily_innovation
he_daily_state
he_weekly_innovation
he_weekly_state
```

- `rt_cadence`: cadence of the latent Rt process. Hospital observations are
  weekly-aggregated in both cases.
- `parameterization`: inner `DifferencedAR1` mode.

Subpopulation models (`pyrenew.latent.SubpopulationInfections`):

```
subpop_hw_innovation
subpop_hw_state
```

Hospital + wastewater on a six-subpopulation California fixture. Daily Rt
only.

### Output files

Written to `--output-dir` with prefix `rt_params_`:

| File | Contents |
|---|---|
| `rt_params_runs.csv` | One row per fit, with full config and metrics. |
| `rt_params_candidates.csv` | One row per candidate, averaged over repeats. |
| `rt_params_pairs.csv` | One row per matched state-vs-innovation pair, with `<metric>_innov`, `<metric>_state`, `<metric>_ratio` columns. |
| `rt_params_runs.json` | All of the above plus a header (suite name, x64 flag, timestamp). |
| `rt_params_report.md` | Compact Markdown report (candidates table and pairwise table). |

Column convention: `_innov` and `_state` carry the per-side values, and
`_ratio` is `state / innovation`. Wall-time `_ratio > 1` means state is
slower. ESS-per-second `_ratio > 1` means state mixes faster per second.

### Reading the metrics

Per fit:

- **Wall time**: total seconds for warmup + sampling, after JIT, with
  `jax.block_until_ready` so the work is fully complete.
- **ESS/s Rt (median / min)**: effective samples per wall-second on the Rt
  trajectory. Median summarizes typical timepoints; min identifies the
  worst-mixing timepoint that limits downstream inference.
- **Divergences**: total NUTS divergences across all chains and draws. A
  saturated tree depth can mask divergences in the worst-mixed runs; read
  with tree depth.
- **Tree depth (mean / max)**: log2 of NUTS leapfrog steps. NumPyro defaults
  to `max_tree_depth=10`. A mean near the ceiling indicates the sampler is
  running out of budget per draw.
- **E-BFMI (min)**: minimum across chains of the energy Bayesian fraction
  of missing information. Heuristic thresholds: >=0.3 acceptable, <0.3
  warning, <0.1 strong pathology indicator.
- **R-hat Rt (max)**: max split R-hat across timepoints of the Rt
  trajectory. Values within 0.01 of 1.0 indicate chain agreement on each
  timepoint.

A pair "favors state" when ESS-per-second ratio is materially > 1 and the
other diagnostics are at least as good. A wall-time difference under 15 %
between parameterizations is expected; the geometric advantage shows up in
ESS, not in per-step cost.

### Suite design

The suite varies three axes:

1. **Parameterization**: `innovation` and `state` modes of the inner
   `DifferencedAR1`.
2. **Prior regime**: tight $(\sigma = 0.01, \phi = 0.9)$ or loose
   $(\sigma = 0.10, \phi = 0.5)$. Both knobs move together; the cumulative
   variance of $\log \mathcal{R}(T)$ scales like
   $\sigma^2 T / (1 - \phi)^2$ and is much more sensitive to $\phi$ than
   to $\sigma$ over the 90 to 126 day horizons used here.
3. **Cadence** (H+E only): daily or weekly cadence of the inner
   `DifferencedAR1`. At 126 days, daily gives 126 latent $\mathcal{R}_t$
   values and weekly gives 18, against the same observed data.

The benchmark interprets $\sigma$ as **daily-equivalent**. When the inner
process runs at weekly cadence, `_build_rt_process` rescales the per-step
SD to $\sigma \sqrt{7}$ so the implied cumulative variance of
$\log \mathcal{R}(T)$ matches the daily configuration at the same horizon.
Without this rescaling, the same numerical $\sigma$ would impose a tighter
per-unit-time prior at weekly cadence than at daily, conflating cadence
with prior strength. The autoregressive coefficient $\phi$ is not
rescaled; matching autocorrelation across cadences would require
$\phi_w \approx \phi_d^7$.

Production HEW pipelines treat both hyperparameters as inferred
(`eta_sd ~ TruncatedNormal(0.15, 0.05)`, `autoreg_rt ~ Beta(2, 40)`); the
benchmark fixes them.

## Adding a benchmark

1. Add a model builder to `benchmarks/core/models.py` that returns a
   `BuiltFit`. Reuse `BuildConfig` if the new model fits the existing axes.
2. If the model needs a new dataset, add a builder to
   `benchmarks/core/datasets.py` and expose it through `SyntheticProvider`.
3. Create a suite module in `benchmarks/suites/` with a `main()` CLI. Use
   `fit_and_measure`, `print_pairwise_tables`, and `write_results` from
   `benchmarks.core`.

## Wiring real data

`benchmarks.core.signals.DatasetProvider` is a `Protocol`. Implement it for
a CDC reporting source and pass the provider to a custom suite; the model
builders and runner do not change. The expected payload is a
`DatasetBundle` whose `signals` mapping carries one `SignalSeries` per
observation source.
