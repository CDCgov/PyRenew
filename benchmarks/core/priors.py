"""Benchmark-local priors for real-data model builds.

These priors mirror the small subset of production HEW prior choices needed
by the benchmark builders, without importing the CDC forecasting pipeline.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist

import pyrenew.transformation as transformation
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable


def real_he_i0_prior() -> DistributionalVariable:
    """Initial infections per capita prior for real H+E benchmark data.

    Returns
    -------
    DistributionalVariable
        Beta prior for the initial infections per capita parameter.
    """
    return DistributionalVariable("I0", dist.Beta(1.0, 10.0))


def real_he_ed_day_of_week_prior() -> TransformedVariable:
    """ED day-of-week effect prior for real H+E benchmark data.

    Returns
    -------
    TransformedVariable
        Dirichlet prior transformed to day-of-week multipliers.
    """
    return TransformedVariable(
        "ed_day_of_week_effect",
        DistributionalVariable(
            "ed_day_of_week_effect_raw",
            dist.Dirichlet(jnp.full(7, 5.0)),
        ),
        transforms=transformation.AffineTransform(loc=0, scale=7),
    )
