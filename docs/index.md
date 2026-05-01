# PyRenew

**Bayesian renewal modeling with JAX and NumPyro.**

PyRenew is a flexible tool for simulation and statistical inference of epidemiological models, emphasizing hierarchical multi-signal renewal models.
Built on top of [NumPyro](https://num.pyro.ai/), PyRenew provides configurable classes that encapsulate the components of a renewal model, and methods to orchestrate their composition into programs that clearly express model structure and choices.

## Renewal models

A renewal model estimates new infections from recent past infections.
It combines two distinct discrete convolutions which describe different processes: transmission between infections and delay from infection to observation.

- The **renewal equation** maps past infections to new infections using the generation interval distribution $w_\tau$.
- The **observation equation** maps latent infections to expected observed events using the delay distribution $\pi_d$.

#### Renewal equation

New infections arise from past infections through a generation interval distribution.

Let $I(t)$ denote the latent number of new infections at time $t$, and let $\mathcal{R}(t)$ denote the time-varying reproduction number.
Assume the generation interval distribution has finite support over lags $\tau = 1, \dots, G$. Let $w_\tau$ denote the probability that a secondary infection occurs $\tau$ days after infection in the primary case, with

$$
\sum_{\tau=1}^{G} w_\tau = 1, \qquad w_\tau \ge 0.
$$

Then the renewal equation is

$$
I(t) = \mathcal{R}(t) \sum_{\tau=1}^{G} I(t - \tau)\, w_\tau.
$$

Here, $\tau$ indexes the generation interval.

In PyRenew, the latent process is represented on a **per-capita scale** (infection proportion) and is multiplied by a population size downstream when connecting to count observations.

#### Observation equation

Infections are latent and are not directly observed; instead, the data consist of events that occur some time after infection, such as hospitalizations or emergency department visits.

Let $\mu(t)$ denote the expected number of observed events at time $t$, and let $\alpha$ denote an **ascertainment rate**, the probability an infection is observed as an event. Assume the delay from infection to observation has finite support over lags $d = 0, \dots, D$. Let $\pi_d$ denote the probability that an infection is observed $d$ days later, with

$$
\sum_{d=0}^{D} \pi_d = 1, \qquad \pi_d \ge 0.
$$

Then the observation equation is

$$
\mu(t) = \alpha \sum_{d=0}^{D} I(t - d)\, \pi_d.
$$

Here, $d$ indexes lags in the infection-to-observation delay distribution.

#### Stochastic observation model

The observation equation defines the expected number of observed events at time $t$, but the actual observed data are stochastic.

Let $Y(t)$ denote the observed number of events at time $t$. We model observations as draws from a count distribution with central value (typically mean) $\mu(t)$:

$$
Y(t) \sim \text{Distribution}(\mu(t), \theta).
$$

One possible choice is the Poisson distribution, which assumes the variance equals the mean.
In practice, epidemiological count data are often overdispersed relative to the Poisson. Negative binomial distributions are a common choice for modeling these overdispersed counts.

The model thus has two layers:

- A **mechanistic layer**, where the renewal and delay convolutions determine the predicted number of observations $\mu(t)$ from the latent infections $I(t)$.
- A **stochastic observation layer**, where observed counts $Y(t)$ vary around $\mu(t)$ according to a specified distribution.

This separation allows the model to distinguish between systematic structure driven by transmission and reporting delays, and stochastic variability in observed data.

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
- [Latent infections](tutorials/latent_infections.md) -- modeling latent infection trajectories over time.
- [Latent subpopulation infections](tutorials/latent_subpopulation_infections.md) -- modeling latent infections with subpopulation structure.
- [Observation processes: count data](tutorials/observation_processes_counts.md) -- connecting latent infections to observed counts.
- [Observation processes: measurements](tutorials/observation_processes_measurements.md) -- connecting latent infections to continuous measurements.
- [Joint ascertainment](tutorials/ascertainment.md) -- sharing ascertainment structure across count signals.

## Resources

- [Developer documentation](developer_documentation.md)

### Models implemented with PyRenew

- [pyrenew-covid-wastewater](https://github.com/CDCgov/pyrenew-covid-wastewater) -- forecasting COVID-19 hospitalizations using wastewater data.
- [pyrenew-flu-light](https://github.com/CDCgov/pyrenew-flu-light/) -- an influenza forecasting model used in the 2023-24 respiratory season.

### Further reading

- [Semi-mechanistic Bayesian modelling of COVID-19 with renewal processes](https://academic.oup.com/jrsssa/article-pdf/186/4/601/54770289/qnad030.pdf) (Bhatt et al., 2023)
- [Unifying incidence and prevalence under a time-varying general branching process](https://link.springer.com/content/pdf/10.1007/s00285-023-01958-w.pdf) (Pakkanen et al., 2023)
