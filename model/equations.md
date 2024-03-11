# Equations

This document is a collection of mathematical definitions for some of commonly (and not so) model components that may be included in the model. Most of the document consists of **verbose reproduction of mathematical definitions/code** from the indicated source.

Eventually, it should incorporate information about the software implementation of these functions.

- [Equations](#equations)
  - [Generic models](#generic-models)
    - [Infections](#infections)
    - [Latent processes for reproductive number](#latent-processes-for-reproductive-number)
    - [Generation interval and delay to reporting time of reference](#generation-interval-and-delay-to-reporting-time-of-reference)
    - [Reporting delay between the time of reference and the time of report](#reporting-delay-between-the-time-of-reference-and-the-time-of-report)
  - [Signals](#signals)
    - [Hospitalizations](#hospitalizations)
    - [Wastewater](#wastewater)
- [Overview](#overview)

## Generic models

### Infections

- _Renewal process_ (from <a href="https://github.com/cdcent/cfa-forecast-renewal-ww/blob/main/model_definition.md#infection-component">the wastewater model</a>)

$$I(t) = \mathcal{R}(t) \sum_{\tau = 1}^{T_g} I(t-\tau) g(\tau)$$

- _Reproductive number damping_ (from <a href="https://github.com/cdcent/cfa-forecast-renewal-ww/blob/main/model_definition.md#infection-component">the wastewater model</a>). This allows the time-varying reproductive number $\mathcal{R}(t)$ to be split into two factors:

$$ \log \mathcal{R}(t) = \log \mathcal{R}^\mathrm{u}(t) - \gamma \sum_{\tau = 1}^{T_f}I(t-\tau)f(\tau).$$

1. $\mathcal{R}^\mathrm{u}(t)$ is the unadjusted reproductive number, which evolves as a latent process.
2. A factor representing the effect of past infections on the current reproductive number. We constrain this factor to be non-negative, so that it can be interpreted as a damping factor. The scalar $\gamma \geq 0$ scales the strength of this damping effect, and the function $f(\tau)$ the time-scale over which past infections influence the current time-varying reproductive number $\mathcal{R}(t)$.

### Latent processes for reproductive number

For the wastewater model $t_1,t_2,t_3$ represent different weeks, but in principle can represent other time scales. We will look to model the diffences in log-reproductive number between two time points, $t_2$ and $t_3$.

- _Differenced autoregressive_ (from from <a href="https://github.com/cdcent/cfa-forecast-renewal-ww/blob/main/model_definition.md#reproductive-number">the wastewater model</a>). This is likely to be the "go-to" model for the latent process of the reproductive numberm and includes _random walk_ as a special case.

$$
\log\mathcal{R}^\mathrm{u}(t_3) -  \log \mathcal{R}^\mathrm{u}(t_2) \sim \mathrm{Normal}\Big(\beta [\log\mathcal{R}^\mathrm{u}(t_2) - \log\mathcal{R}^\mathrm{u}(t_1) ], \sigma_r \Big).
$$

- In general, other _Gaussian processes_ for time-differences in reproductive number (from <a href="https://epiforecasts.io/EpiNow2/dev/articles/estimate_infections.html#time-varying-reproduction-number">EpiNow2</a>)

```math
\log\mathcal{R}^\mathrm{u}(t_3) - \log\mathcal{R}^\mathrm{u}(t_2) \sim \text{GP}_{t_2} | \{\log\mathcal{R}^\mathrm{u}(t)\}_{t \leq t_2}.
```

- Additional effects via a link function (from [epinowcast](https://package.epinowcast.org/dev/articles/model.html#instantaneous-reproduction-numbergrowth-rate))

$$
\log\mathcal{R}^\mathrm{u}(t_3) -  \log \mathcal{R}^\mathrm{u}(t_2) =\text{baseline model} + \beta_{f, r} X_r + \beta_{r,r} Z_r.
$$

Where $X$ and $Z$ are the design matrices for fixed and random effects, respectively, associated with the signal we are modelling. NB: This can be extended to spline regression.

### Generation interval and delay to reporting time of reference

1. The generation interval is the random time between the infection of an index infection and the infection of a secondary infection.
2. The reporting reference time delay is the random time between infection of an eventual case and the reference time of the case ascertainment (see [Epinowcast definition](https://package.epinowcast.org/dev/articles/model.html#decomposition-into-expected-final-notifications-and-report-delay-components)).

This is a discrete time model, likely to use daily dynamics. Therefore, the distributions of the random time intervals above must be expressed as discrete probability mass functions (PMFs) over discrete time lags.

Options for discretisation:
- User defined PMF (see [EpiSewer](https://github.com/adrian-lison/EpiSewer/blob/main/vignettes/model-definition.md) or wastewater model)

- Discretized PMF from a continuous distribution for the generation interval, (see [preprint](https://www.medrxiv.org/content/10.1101/2024.01.12.24301247v1)).

### Reporting delay between the time of reference and the time of report

The reporting delay is the random time between the time of reference of a case and the time of report when the data of that case becomes available to analysts (see [Epinowcast definition](https://package.epinowcast.org/dev/articles/model.html#decomposition-into-expected-final-notifications-and-report-delay-components)).

Using [epinowcast notation](https://package.epinowcast.org/dev/articles/model.html#decomposition-into-expected-final-notifications-and-report-delay-components), for any [signal](#signals) our the probability that a case with reference time $t$ is reported at time $t+d$ is given by, $p_{t,d}$.

Options for constructing the reporting delay PMF:

- [Discretised](#generation-interval-and-delay-to-reporting-time-of-reference) lognormal function (see [epinowcast](https://package.epinowcast.org/dev/articles/model.html#default-model-1)),

$$
\text{LogNormal}(\mu_d, \sigma_d).
$$

distribution, with parameters $\mu_d \sim \text{Normal}(0,1)$ and $\sigma_d \sim \text{Half-Normal}(0,1)$. NB: this assumes that the reporting delay is independent of the time of reference, excepting possible interval censoring effects.

- Hazard model (see [epinowcast](https://package.epinowcast.org/dev/articles/model.html#generalised-model-1))

$$
p_{t,0} = h_{t,0},\qquad p_{t,d}=\left(1−\sum_{d'=0}^{d−1}p_{t,d}\right) \times h_{t,d},
$$

The [hazard](https://en.wikipedia.org/wiki/Proportional_hazards_model) of a survival model with time-varying covariates, $W_{t,d}$, is given by,

$$h_{t,d} = P(\text{delay}=d|\text{delay} \geq d, W_{t,d}).$$



## Signals

### Hospitalizations

from <a href="https://github.com/cdcent/cfa-forecast-renewal-ww/blob/main/model_definition.md#hospital-admissions-component">the wastewater model</a> the eventual expected number of hospitalizations with [reference time](#reporting-delay-between-the-time-of-reference-and-the-time-of-report) $t$ is given by:

$$H(t) = \omega(t) ~ p_\mathrm{hosp}(t) \sum_{\tau = 0}^{T_d} d(\tau) I(t-\tau).$$

### Wastewater

from <a href="https://github.com/cdcent/cfa-forecast-renewal-ww/blob/main/model_definition.md#wastewater-viral-concentration-component">the wastewater model</a> the viral concentration at time $t$ is given by:

$$C(t) = \frac{G}{\alpha} \sum_{\tau = 0}^{\tau_\mathrm{shed}} s(\tau) I(t-\tau).$$

# Overview

The following diagram provides a general view of the main model components and how are these connected (under development):

```{mermaid}
flowchart LR

  %% Global level DGP ----------------------------------------------------------
  subgraph dgp_global["Global-level DGP"]

    direction TB

    subgraph rt["R(t)"]
      rt_elements["-Gaussian\n-AR\n-Link\n-User-def"]
    end
    style rt_elements text-align:left

    subgraph gt["g(t)"]
      gt_elements["-Log-Normal\n-Hazard\n-User-def"]
    end
    style gt_elements text-align:left

  end
  rt ~~~ gt

  %% Local level DGP -----------------------------------------------------------

  it(("I(t, j)\nLatent Cases\nat j"))

  subgraph i0["I(0, j)"]
    i0_elements["-Exp. growth\n-Single param\n-Random walk."]
    style i0_elements text-align:left
  end

  subgraph signals["Signals in location j"]
    direction TB
    signals_elements["y<sub>1</sub>: Cases
    y<sub>2</sub>: Hospitalizations
    y<sub>3</sub>: Wastewater
    ...
    y<sub>M</sub>: M-th signal"]
  end
  style signals_elements stroke-width:0px,fill:transparent,text-align:left;
  it --> |"As a parameter\n(e.g., Avg. NegBinom.)"| signals
  i0 --> |Used in| it

  %% Rt hierarchical process
  subgraph Rt_local["R hierarchical"]

    direction LR

    Rt(("R(t)")) --> Rtdots(("..."))
    Rtdots --> Rtj(("R(t, j)"))
    style Rtdots fill:transparent,stroke:transparent
  end

  dgp_global --> Rt_local
  dgp_global --> gt_local(("g(t)"))

  Rt_local --> |Used in| it
  gt_local --> |Used in| it

  classDef transparentClass fill:transparent,stroke:darkgray;
  class dgp_global,dgp_local_j transparentClass;
```
