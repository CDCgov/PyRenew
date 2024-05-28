# numpydoc ignore=GL08


import numpyro.distributions as dist
from pyrenew.metaclass import DistributionalRV


class Infections0(DistributionalRV):
    """
    Initial infections helper class. (wrapper of DistributionalRV)

    It creates a random variable for the initial infections with a prior
    distribution.
    """

    def __init__(
        self,
        dist: dist.Distribution,
        name: str | None = "I0",
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        dist : dist.Distribution, optional
            Distribution of the initial infection count.
        name : str, optional
            Name of the random variable, by default "I0"

        Returns
        -------
        None
        """
        super().__init__(
            dist=dist,
            name=name,
        )

        return None
