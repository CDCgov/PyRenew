# numpydoc ignore=GL08
"""
Return types for observation processes.

Named tuples providing structured access to observation process outputs.
"""

from typing import NamedTuple

from jax.typing import ArrayLike


class ObservationSample(NamedTuple):
    """
    Return type for observation process sample() methods.

    Attributes
    ----------
    observed
        Sampled or conditioned observations. Shape depends on the
        observation process and indexing.
    predicted
        Predicted values before noise is applied. Useful for
        diagnostics and posterior predictive checks.
    """

    observed: ArrayLike
    predicted: ArrayLike
