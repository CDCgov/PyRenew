import numpyro as npro
import numpyro.distributions as dist
from pyrenew.metaclass import RandomVariable


class Infections0(RandomVariable):
    def __init__(
        self,
        name: str = "I0",
        I0_dist: dist.Distribution = dist.LogNormal(0, 1),
    ) -> None:
        self.validate(I0_dist)

        self.name = name
        self.i0_dist = I0_dist

        return None

    @staticmethod
    def validate(i0_dist):
        assert isinstance(i0_dist, dist.Distribution)

    def sample(
        self,
        random_variables: dict,
        constants: dict,
    ) -> tuple:
        return (
            npro.sample(
                name=self.name,
                fn=self.i0_dist,
            ),
        )
