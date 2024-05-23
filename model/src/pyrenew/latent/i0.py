# numpydoc ignore=GL08


import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from pyrenew.metaclass import RandomVariable


class Infections0(RandomVariable):
    """
    Initial infections helper class.

    It creates a random variable for the initial infections with a prior
    distribution.
    """

    def __init__(
        self,
        name: str | None = "I0",
        I0_dist: dist.Distribution | None = dist.LogNormal(0, 1),
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        name : str, optional
            Name of the random variable, by default "I0"
        I0_dist : dist.Distribution, optional
            Distribution of the initial infections, by default dist.LogNormal(0, 1)

        Returns
        -------
        None
        """
        self.validate(I0_dist)

        self.name = name
        self.i0_dist = I0_dist

        return None

    @staticmethod
    def validate(i0_dist: any) -> None:
        """
        Validate the initial infections distribution.

        Parameters
        ----------
        i0_dist : any
            Distribution (expected dist.Distribution) of the initial infections.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the inputted distribution is not a Numpyro distribution.
        """
        assert isinstance(i0_dist, dist.Distribution)

    def sample(
        self,
        **kwargs,
    ) -> tuple:
        """
        Sample the initial infections.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        tuple
            Tuple with the initial infections.
        """
        return (
            jnp.atleast_1d(
                npro.sample(
                    name=self.name,
                    fn=self.i0_dist,
                )
            ),
        )
