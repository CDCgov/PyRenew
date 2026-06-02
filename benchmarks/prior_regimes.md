# Prior regimes: comparing one model under many priors

## Why this exists

These renewal models are strongly prior-driven.
As usually composed they are only weakly identified, so the sampler behaves well or badly largely according to how informative the priors are.
The practical research question is therefore not "innovation vs state" but "given this model structure, which set of priors lets the sampler work, and where does it struggle."
That means running the *same model structure* under *many prior choices* and comparing the runs.
Doing this by hand produces a pile of loosely related fits that is hard to organize.
This document describes the pattern the benchmark harness uses to make that comparison systematic and the results self-documenting.

## The core separation: structure is code, priors are data

A suite holds one build function that fixes the model structure: which components exist (latent infection process, ascertainment, observation processes), their cadence, and how they are wired.
The build function takes the priors as an argument and does not hard-code any prior values.

A prior choice is a separate, self-contained function (a "regime") that returns the priors the build function consumes.
Swapping regimes changes priors without touching structure, so the comparison is genuinely "same model, different priors."

```python
def _build_he_model(bundle, priors):
    """Assemble the H+E model structure, drawing every prior from `priors`."""
    builder = PyrenewBuilder()
    builder.configure_latent(
        PopulationInfections,
        gen_int_rv=DeterministicPMF("gen_int", bundle.gen_int_pmf),
        I0_rv=priors["I0"](),
        log_rt_time_0_rv=priors["log_rt_time_0"](),
        single_rt_process=WeeklyTemporalProcess(
            DifferencedAR1(
                autoreg_rv=priors["rt_diff_autoreg"](),
                innovation_sd_rv=priors["rt_diff_innovation_sd"](),
                parameterization="state",
            ),
            start_dow=MMWR_WEEK,
        ),
    )
    # ... ascertainment and observation processes, also drawn from `priors` ...
    return BuiltFit(model=builder.build(), run_kwargs=..., dataset_name=bundle.name)
```

## Defining a prior regime

A regime is a function returning a dict of zero-arg factories, one per prior slot.
Each factory builds a fresh random variable, so each model build gets its own instances.

Write the distributions directly.
Do not call named helper functions that hide the values.
The results capture each regime by its source text, so an inline `Beta(1.0, 10.0)` documents itself, whereas a call like `real_he_i0_prior` would record only the name and leave the actual prior outside the results.

```python
def example_priors():
    """Starting-point priors. Not authoritative; copy this and override it."""
    return {
        "rt_diff_innovation_sd": lambda: DeterministicVariable(
            "rt_diff_innovation_sd", 0.01
        ),
        "rt_diff_autoreg": lambda: DeterministicVariable("rt_diff_autoreg", 0.9),
        "log_rt_time_0": lambda: DistributionalVariable(
            "log_rt_time_0", dist.Normal(0.0, 0.5)
        ),
        "hosp_conc": lambda: DistributionalVariable(
            "hosp_conc", dist.LogNormal(5.0, 1.0)
        ),
        "ed_conc": lambda: DistributionalVariable("ed_conc", dist.LogNormal(4.0, 1.0)),
        "I0": lambda: DistributionalVariable("I0", dist.Beta(1.0, 10.0)),
    }
```

Fixing a hyperparameter versus inferring it is a one-line change inside a regime, with no change to structure.
To stop pinning the autoregressive coefficient $\phi$ and instead infer it toward the production setting, change one entry:

```python
"rt_diff_autoreg": lambda: DistributionalVariable("rt_diff_autoreg", dist.Beta(2, 40)),
```

A "looser" regime is another function that widens the same slots.
The weekly per-step innovation standard deviation $\sigma$ and the autoregressive coefficient $\phi$ are the usual knobs, with the cumulative variance of $\log \mathcal{R}(T)$ far more sensitive to $\phi$ than to $\sigma$.

## Varying only what differs

Every regime must return a complete bag (all slots are required; there are no defaults, matching PyRenew's components, which require explicit priors).
A comparison is usually a set of controlled single changes off a common baseline, so retyping every slot in each regime buries the one line that differs and invites copy-paste drift in the slots you meant to hold fixed.
Instead, spread an existing regime and override only what changes:

```python
def weak_phi_priors():
    """Example, but with a weaker autoregressive coefficient."""
    return {
        **example_priors(),
        "rt_diff_autoreg": lambda: DeterministicVariable("rt_diff_autoreg", 0.5),
    }
```

This returns a full bag, so validation passes, and the regime's recorded source shows exactly the diff.
The comparison then reads as the baseline versus each one-change variant, which is the controlled-experiment structure you want for prior sensitivity.

One rule keeps this safe: spread only from a regime that is itself in `REGIMES`.
Its source is then recorded in the `prior_configs` block too, so a reader composes the base and the diff to recover the full effective priors.
Spreading from a helper that is not recorded reintroduces the hidden-value gap that writing distributions inline was meant to avoid.

## Assembling the comparison

The example driver `benchmarks/examples/run_prior_regimes.py` enumerates the regimes and turns each into a candidate over the fixed structure, then hands them to `run_comparison`.
Run it with `python -m benchmarks.examples.run_prior_regimes`; copy it to another file under `benchmarks/examples/` (everything there but the committed examples is gitignored) to run your own regimes.

```python
REGIMES = {
    "example": example_priors,  # ships with the template; a starting point to override
    # add your own regimes here, e.g.:
    # "weak_rt": weak_rt_priors,
    # "weak_i0": weak_i0_priors,
}

candidates = [
    Candidate(
        name=name,
        arm=name,
        config_fields={"prior_config": name},
        build=lambda fn=regime_fn: _build_he_model(bundle, fn()),
    )
    for name, regime_fn in REGIMES.items()
]
```

The comparison is declared once.
The regimes are the arms; one is the baseline that the others are rated against.

```python
COMPARISON_SPEC = ComparisonSpec(
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
