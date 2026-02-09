# PyRenew

**Bayesian renewal modeling with JAX and NumPyro.**

PyRenew is a flexible tool for simulation and statistical inference of epidemiological models, emphasizing hierarchical multi-signal renewal models.
Built on top of [NumPyro](https://num.pyro.ai/), PyRenew provides configurable classes that encapsulate the components of a renewal model, and methods to orchestrate their composition into programs that clearly express model structure and choices.

## Renewal models

A renewal model estimates new infections from recent past infections using a generation interval distribution $w(s)$: the probability that $s$ time units separate infection in an index case and a secondary case.
The core renewal equation is:

$$I(t) = R_t \sum_{s} I(t-s) \, w(s)$$

where $R_t$ is the time-varying reproduction number.

Inference is complicated by the fact that observational data require their own models ([Bhatt et al., 2023, S2](https://doi.org/10.1093/jrsssa/qnad030)).
The observation equation links infections to expected observations:

$$\mu(t) = \alpha \sum_{s} I(t-s) \, \pi(s)$$

where $\alpha$ is the ascertainment rate and $\pi(s)$ is the delay distribution from infection to observation.

## Design

PyRenew's building blocks are:

- **`RandomVariable`** -- an abstract base class for quantities that models can sample from, including stochastic draws via `numpyro.sample()`, mechanistic computations, and fixed values.
- **`Model`** -- an abstract base class that defines model structure through a `sample()` method and provides functionality for fitting and simulation.
- **`PyrenewBuilder`** -- an orchestrator that composes `RandomVariable` components into a complete renewal model, auto-computing derived quantities like the number of initialization points.

Components (generation interval, reproduction number process, observation process) are specified independently, so each can be swapped without changing the rest of the model.
This makes it straightforward to move a quantity between "known" and "inferred" and keeps modeling choices explicit and reviewable.


## Multi-signal models

PyRenew's strength lies in multi-signal integration: pooling information across diverse observed data streams such as hospital admissions, wastewater concentrations, and emergency department visits, where each signal has distinct observation delays, noise characteristics, and spatial resolutions.
For single-signal renewal models, we recommend the excellent R package [EpiNow2](https://epiforecasts.io/EpiNow2/).

## Installation

```bash
pip install git+https://github.com/CDCgov/PyRenew@main
```

## Tutorials

- [The RandomVariable abstract base class](tutorials/random_variables.md) -- PyRenew's core abstraction and its concrete implementations.
- [Building multi-signal models](tutorials/building_multisignal_models.md) -- composing a renewal model from PyRenew components using `PyrenewBuilder`.
- [Observation processes: count data](tutorials/observation_processes_counts.md) -- connecting latent infections to observed counts.
- [Observation processes: measurements](tutorials/observation_processes_measurements.md) -- connecting latent infections to continuous measurements.
- [Latent hierarchical infections](tutorials/latent_hierarchical_infections.md) -- modeling infections with subpopulation structure.

## Resources

- [Model equations](https://github.com/CDCgov/PyRenew/blob/main/equations.md) -- the mathematics of the multi-signal renewal processes PyRenew supports.
- [Developer documentation](developer_documentation.md)

### Models implemented with PyRenew

- [pyrenew-covid-wastewater](https://github.com/CDCgov/pyrenew-covid-wastewater) -- forecasting COVID-19 hospitalizations using wastewater data.
- [pyrenew-flu-light](https://github.com/CDCgov/pyrenew-flu-light/) -- an influenza forecasting model used in the 2023-24 respiratory season.

### Further reading

- [Semi-mechanistic Bayesian modelling of COVID-19 with renewal processes](https://academic.oup.com/jrsssa/article-pdf/186/4/601/54770289/qnad030.pdf) (Bhatt et al., 2023)
- [Unifying incidence and prevalence under a time-varying general branching process](https://link.springer.com/content/pdf/10.1007/s00285-023-01958-w.pdf)
