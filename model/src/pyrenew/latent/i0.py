import numpyro as npro
import numpyro.distributions as dist
from pyrenew.metaclass import RandomVariable


class Infections0(RandomVariable):
    """Initial infections helper class.

    It creates a random variable for the initial infections with a prior
    distribution.
    """

    def __init__(
        self,
        name: str = "I0",
        I0_dist: dist.Distribution = dist.LogNormal(0, 1),
    ) -> None:
        """Default constructor

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
    def validate(i0_dist):
        """Validate the initial infections distribution.

        Parameters
        ----------
        i0_dist : dist.Distribution
            Distribution of the initial infections.

        Returns
        -------
        None
        """
        assert isinstance(i0_dist, dist.Distribution)

    def sample(
        self,
        **kwargs,
    ) -> tuple:
        """Sample the initial infections.

        Parameters
        ----------
        kwargs : dict
            Ignored

        Returns
        -------
        tuple
            Tuple with the initial infections.
        """
        return (
            npro.sample(
                name=self.name,
                fn=self.i0_dist,
            ),
        )
