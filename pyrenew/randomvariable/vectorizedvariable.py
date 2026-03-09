"""
Vectorization wrapper for simple RandomVariables
"""

import numpyro
from jax.typing import ArrayLike

from pyrenew.metaclass import RandomVariable


class VectorizedVariable(RandomVariable):
    """
    Wrapper that adds n_groups support to simple RandomVariables.

    Uses numpyro.plate to vectorize sampling, enabling simple RVs
    to work with noise models expecting the group-level interface.
    """

    def __init__(self, name: str, rv: RandomVariable) -> None:
        """
        Initialize VectorizedVariable wrapper.

        Parameters
        ----------
        name
            A name for this random variable.
            The numpyro plate used to vectorize will
            have this name with the suffix `_plate"`.
        rv
            The underlying RandomVariable to wrap.
        """
        super().__init__(name=name)
        self.rv = rv
        self.plate_name = f"{name}_plate"

    def validate(self) -> None:  # pragma: no cover
        """Validate the underlying RV."""
        self.rv.validate()

    def sample(self, n_groups: int, **kwargs: object) -> ArrayLike:
        """
        Sample n_groups values using numpyro.plate.

        Parameters
        ----------
        n_groups
            Number of group-level values to sample.

        Returns
        -------
        ArrayLike
            Array of shape (n_groups,).
        """
        with numpyro.plate(self.plate_name, n_groups):
            return self.rv(**kwargs)
