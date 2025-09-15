"""
Built-in pyrenew transformations created using [numpyro.distributions.transforms.Transform][].
"""

import numpyro.distributions.transforms as nt
from numpyro.distributions import constraints


def ScaledLogitTransform(
    x_max: float,
) -> nt.ComposeTransform:
    """
    Scaled logistic transformation from the
    interval (0, X_max) to the interval
    (-infinity, +infinity).

    Parameters
    ----------
    x_max: float
        Maximum value of the untransformed scale (will be transformed to
        +infinity).

    Returns
    -------
    nt.ComposeTransform
        A composition of the following transformations:
        - numpyro.distributions.transforms.AffineTransform(0.0, 1.0/x_max)
        - numpyro.distributions.transforms.SigmoidTransform().inv
    """
    return nt.ComposeTransform(
        [
            nt.AffineTransform(
                0.0, 1.0 / x_max, domain=constraints.interval(0.0, 1.0 * x_max)
            ),
            nt.SigmoidTransform().inv,
        ]
    )
