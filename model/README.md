# PyRenew: A Package for Bayesian Renewal Modeling with JAX and Numpyro.

`pyrenew` is a flexible tool for simulating and statistical inference of epidemiological models, emphasizing renewal models. Built on top of the [`numpyro`](https://num.pyro.ai/) Python library, `pyrenew` provides core components for model building, including pre-defined models for processing various types of observational processes.

## Installation

Install via pip with

```bash
pip install git+https://github.com/CDCgov/multisignal-epi-inference@main#subdirectory=model
```

## Container image

A container image is available at `ghcr.io/CDCgov/pyrenew:latest`. You can pull it with

```bash
docker pull ghcr.io/CDCgov/pyrenew:latest
```

## Demo

The [`docs`](docs) folder contains quarto documents to get you started. It simulates observed hospital admissions using a simple renewal process model and then fits it using a No-U-Turn Sampler.
