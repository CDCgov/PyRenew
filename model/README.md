# PyRenew: A package for Bayesian renewal modeling with JAX and Numpyro.

`pyrenew` is a flexible tool for simulation and inference of epidemiological models with an emphasis on renewal models. Built on top of `numpyro`, `pyrenew` provides core components for model building as well as pre-defined models for processing various types of observational processes.

## Installation

Install via pip with

```bash
pip install git+https://github.com/cdcent/cfa-pyrenew.git
```

## Demo

The [`docs`](docs) folder contains quarto documents to get you started. It simulates observed hospitalizations using a simple renewal process model and then fits to it using a No-U-Turn Sampler.
