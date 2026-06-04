# Prior regimes: comparing one model under many priors

## Why this exists

These renewal models are strongly prior-driven.
As usually composed they are only weakly identified, so the sampler behaves well or badly largely according to how informative the priors are.
The practical research question is therefore not "innovation vs state" but "given this model structure, which set of priors lets the sampler work, and where does it struggle."
That means running the *same model structure* under *many prior choices* and comparing the runs.
Doing this by hand produces a pile of loosely related fits that is hard to organize.
This document describes the pattern the benchmark harness uses to make that comparison systematic and the results self-documenting.

## The core separation: structure is code, priors are data

The model structure — which components exist (latent infection process, ascertainment, observation processes), their cadence, and how they are wired — lives in one shared builder, `benchmarks/models/he.py:build_he_model`.
It takes an `HEModelConfig` and a dataset bundle and returns a `BuiltFit`.

The priors are the fields of `HEModelConfig`.
Every random variable the builder passes to `configure_latent` or to an observation component is a config field with a default; a regime sets the ones it wants to change.
Swapping configs changes priors without touching structure, so the comparison is genuinely "same model, different priors."

```python
@dataclass(frozen=True)
class HEModelConfig:
    rt: Parameterization = "state"  # structural axis, not a prior
    day_of_week: DayOfWeek = "infer"
    autoreg_rv: RandomVariable = ...  # default Deterministic(0.9)
    innovation_sd_rv: RandomVariable = ...  # default Deterministic(0.05)
    log_rt_time_0_rv: RandomVariable = ...  # default Normal(0, 0.5)
    i0_rv: RandomVariable | None = None  # None: derive from the bundle
    hosp_conc_rv: RandomVariable = ...  # default LogNormal(5, 1)
    ed_conc_rv: RandomVariable = ...  # default LogNormal(4, 1)
    ascertainment: AscertainmentModel = ...  # default joint Gaussian
```

Data inputs (generation interval, delays, signal values, cadence, right truncation) come from the bundle, not the config.

## Defining a prior regime

A regime is a function returning an `HEModelConfig` with the prior fields it wants set.
Fields it leaves out take the standard defaults.

Write the distributions directly.
Do not call named helper functions that hide the values.
The results capture each regime by its source text, so an inline `Beta(1.0, 10.0)` documents itself, whereas a call like `real_he_i0_prior` would record only the name and leave the actual prior outside the results.

```python
def example() -> HEModelConfig:
    """Starting-point priors. Not authoritative; copy this and override it."""
    return HEModelConfig(
        rt="state",
        day_of_week="none",
        autoreg_rv=DeterministicVariable("rt_diff_autoreg", 0.9),
        innovation_sd_rv=DeterministicVariable("rt_diff_innovation_sd", 0.01),
        log_rt_time_0_rv=DistributionalVariable("log_rt_time_0", dist.Normal(0.0, 0.5)),
        i0_rv=DistributionalVariable("I0", dist.Beta(1.0, 10.0)),
        hosp_conc_rv=DistributionalVariable("hosp_conc", dist.LogNormal(5.0, 1.0)),
        ed_conc_rv=DistributionalVariable("ed_conc", dist.LogNormal(4.0, 1.0)),
        ascertainment=JointAscertainment(...),
    )
```

Fixing a hyperparameter versus inferring it is a one-field change inside a regime, with no change to structure.
To stop pinning the autoregressive coefficient $\phi$ and instead infer it toward the production setting, change one field:

```python
autoreg_rv = DistributionalVariable("rt_diff_autoreg", dist.Beta(2, 40))
```

A "looser" regime is another function that widens the same fields.
The weekly per-step innovation standard deviation $\sigma$ and the autoregressive coefficient $\phi$ are the usual knobs, with the cumulative variance of $\log \mathcal{R}(T)$ far more sensitive to $\phi$ than to $\sigma$.

## Varying only what differs

`HEModelConfig` has a default for every field, so a regime sets only what it changes.
A comparison is usually a set of controlled single changes off a common baseline, so build each variant by `replace`-ing one field of a baseline regime rather than retyping the rest:

```python
from dataclasses import replace


def weak_phi() -> HEModelConfig:
    """Example, but with a weaker autoregressive coefficient."""
    return replace(example(), autoreg_rv=DeterministicVariable("rt_diff_autoreg", 0.5))
```

The comparison then reads as the baseline versus each one-change variant, which is the controlled-experiment structure you want for prior sensitivity.

One rule keeps this safe: build off a regime that is itself in `REGIMES`.
Its source is then recorded in the `prior_configs` block too, so a reader composes the base and the diff to recover the full effective priors.
Building off a helper that is not recorded reintroduces the hidden-value gap that writing distributions inline was meant to avoid.

## Assembling the comparison

The example driver `benchmarks/examples/run_prior_regimes.py` enumerates the regimes, turns each into an `Arm`, and hands them to `comparison_suite`, which supplies the CLI, data loading, and the fit / report loop.
Run it with `python -m benchmarks.examples.run_prior_regimes`; copy it to another file under `benchmarks/examples/` (everything there but the committed examples is gitignored) to run your own regimes.

```python
REGIMES: dict[str, Callable[[], HEModelConfig]] = {
    "example": example,
    # add your own regimes here, e.g. "weak_phi": weak_phi,
}

ARMS = [
    Arm(name=name, config=regime(), config_fields={"prior_config": name})
    for name, regime in REGIMES.items()
]

main = comparison_suite(
    SPEC,
    ARMS,
    build_he_model,
    extra_payload={"prior_configs": _prior_provenance()},
)
```

The comparison is declared once.
The regimes are the arms; one is the baseline that the others are rated against.

```python
SPEC = ComparisonSpec(
    name="prior_regimes",
    arms=tuple(REGIMES),
    baseline="example",
    match_keys=("dataset",),
    metrics=DEFAULT_METRICS,
)
```

## What the results give you

- A comparison table with the regimes side by side, each rated against the baseline, with columns such as `divergences__weak_rt` and `ess_per_sec_rt_min__weak_rt` and `ess_per_sec_rt_min__ratio__weak_rt`.
  This answers directly which prior choices sample, and how much better or worse than the baseline.
- A per-site ESS and R-hat table, per regime, showing which specific parameters mix poorly under each prior.
  This is where non-identifiability shows itself, so it is the diagnostic you actually want when a prior choice fails.
- A `prior_config` column in `runs.csv` and `candidates.csv`, so every row states which prior set produced it.
- The exact priors recorded in `runs.json`, captured as the source text of each regime function.
  Because regimes are self-contained, the recorded source is the complete prior specification with nothing to look up.

## Reading the metrics for prior sensitivity

The standard metric set is exactly what distinguishes a workable prior from one that leaves the model under-identified:

- Divergences.
  Nonzero and rising under a loosened prior is the clearest sign the geometry has degraded.
- Minimum ESS per second on $\mathcal{R}(t)$.
  The worst-mixing timepoint is what limits inference; a vague prior usually drops this sharply.
- E-BFMI.
  Low values flag momentum-resampling problems that vague priors tend to induce.
- Maximum R-hat.
  Cross-chain disagreement that appears only under looser priors points to multimodality or drift.
- Per-site ESS.
  Identifies the individual parameters whose effective sample size collapses, naming where the non-identifiability lives.

## Adding and scaling regimes

Adding a prior choice is one regime function plus one entry in `REGIMES`.
Nothing in the shared machinery changes.
A single regime is allowed; it profiles one prior set and emits an empty comparison table while still producing the per-candidate and per-site tables.

If you also vary something structural, choose which axis is compared side by side and which is held equal.
For example, comparing regimes while also varying the Rt parameterization: keep the regimes as arms and set `match_keys = ("dataset", "parameterization")`, which yields one regime comparison per parameterization.

## Provenance

Each run records the source of its regime function via `inspect.getsource`, stored once per regime in a `prior_configs` block in `runs.json`, with the regime name carried on each run.
Keeping regimes self-contained (inline distributions, no hidden helpers) makes that captured source the full, unambiguous prior specification, independent of later edits to any helper.

## Relationship to the other docs

`README.md` covers how to run a suite and read its output files.
`architecture.md` covers the machinery (`Candidate`, `ComparisonSpec`, the runner, and the reporting layer) that this workflow composes.
This document covers the research workflow those pieces are meant to support: comparing prior regimes over a fixed structure.
