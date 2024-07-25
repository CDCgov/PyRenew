# numpydoc ignore=GL08

import jax.numpy as jnp
from pyrenew.deterministic.deterministic import DeterministicVariable
from pyrenew.metaclass import SampledValue


class DeterministicProcess(DeterministicVariable):
    """
    A deterministic process (degenerate) random variable.
    Useful to pass fixed quantities over time."""

    __init__ = DeterministicVariable.__init__

    def sample(
        self,
        duration: int,
        **kwargs,
    ) -> tuple:
        """
        Retrieve the value of the deterministic Rv

        Parameters
        ----------
        duration : int
            Number of timepoints to sample.
        **kwargs : dict, optional
            Ignored.

        Returns
        -------
        tuple
            Containing the stored values during construction wrapped in a SampledValue.
        """

        res, *_ = super().sample(**kwargs)

        dif = duration - res.value.shape[0]

        if dif > 0:
            res = (
                SampledValue(
                    jnp.hstack([res.value, jnp.repeat(res.value[-1], dif)]),
                    t_start=self.t_start,
                    t_unit=self.t_unit,
                ),
            )
        else:
            res = (
                SampledValue(
                    value=res.value[:duration],
                    t_start=self.t_start,
                    t_unit=self.t_unit,
                ),
            )

        return res
