"""Factory for logit-normal random variables."""

import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.infer.reparam import Reparam
from numpyro.util import not_jax_tracer

import pyrenew.transformation as transformation
from pyrenew.randomvariable.distributionalvariable import DistributionalVariable
from pyrenew.randomvariable.transformedvariable import TransformedVariable


def LogitNormalVariable(
    name: str,
    median: ArrayLike,
    scale: ArrayLike,
    base_name: str | None = None,
    reparam: Reparam = None,
) -> TransformedVariable:
    """Create a logit-normal distributed random variable.

    Parameters
    ----------
    name
        Name of the random variable.
    median
        Median of the random variable on the probability scale. All values
        must be finite and strictly between 0 and 1.
    scale
        Standard deviation of the Normal distribution on the logit scale. All
        values must be finite and strictly positive.
    base_name
        Name of the underlying Normal random variable. Defaults to
        ``f"logit_{name}"``.
    reparam
        If not None, reparameterize sampling from the underlying Normal
        distribution according to the given NumPyro reparameterizer.

    Returns
    -------
    TransformedVariable
        A transformed variable whose underlying Normal random variable has
        location ``logit(median)`` and the specified scale.

    Raises
    ------
    ValueError
        If any statically available median is not finite and strictly between
        0 and 1, or any statically available scale is not finite and positive.
    """
    median = jnp.asarray(median)
    invalid_median = jnp.any(~jnp.isfinite(median) | (median <= 0) | (median >= 1))
    if not_jax_tracer(invalid_median) and bool(invalid_median):
        raise ValueError(
            "median must contain only finite values strictly between 0 and 1"
        )

    scale = jnp.asarray(scale)
    invalid_scale = jnp.any(~jnp.isfinite(scale) | (scale <= 0))
    if not_jax_tracer(invalid_scale) and bool(invalid_scale):
        raise ValueError("scale must contain only finite positive values")

    sigmoid_transform = transformation.SigmoidTransform()
    return TransformedVariable(
        name=name,
        base_rv=DistributionalVariable(
            name=base_name or f"logit_{name}",
            distribution=dist.Normal(
                sigmoid_transform.inv(median),
                scale,
            ),
            reparam=reparam,
        ),
        transforms=sigmoid_transform,
    )
