import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from pyrenew.metaclasses import RandomProcess


class SimpleRandomWalkProcess(RandomProcess):
    """
    Class for a Markovian
    random walk with an a
    abitrary step distribution
    """

    def __init__(
        self,
        error_distribution: dist.Distribution,
    ) -> None:
        self.error_distribution = error_distribution

    def sample(
        self,
        duration,
        name="randomwalk",
        init=None,
    ) -> tuple:
        if init is None:
            init = npro.sample(name + "_init", self.error_distribution)
        diffs = npro.sample(
            name + "_diffs", self.error_distribution.expand((duration,))
        )

        return (init + jnp.cumsum(jnp.pad(diffs, [1, 0], constant_values=0)),)

    @staticmethod
    def validate():
        return None
