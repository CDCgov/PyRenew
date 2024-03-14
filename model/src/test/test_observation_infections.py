import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import numpyro as npro
from pyrenew.observations import InfectionsObservation, PoissonObservation
from pyrenew.processes import RtRandomWalkProcess


def test_infections_as_deterministic():
    """
    Check that an InfectionObservation
    can be initialized and sampled from (deterministic)
    """

    np.random.seed(223)
    rt = RtRandomWalkProcess()
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_rt = rt.sample(data={"n_timepoints": 30})

    inf1 = InfectionsObservation(jnp.array([0.25, 0.25, 0.25, 0.25]))

    i0 = dict(I0=10)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        inf_sampled1 = inf1.sample(Rt=sim_rt, data=i0)
        inf_sampled2 = inf1.sample(Rt=sim_rt, data=i0)

    # Should match!
    testing.assert_array_equal(inf_sampled1, inf_sampled2)


def test_infections_as_random():
    """_summary_"""

    np.random.seed(223)
    rt = RtRandomWalkProcess()
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_rt = rt.sample(data={"n_timepoints": 30})

    inf1 = InfectionsObservation(
        jnp.array([0.25, 0.25, 0.25, 0.25]),
        inf_observation_model=PoissonObservation(),
    )

    i0 = dict(I0=10)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        inf_sampled0 = inf1.sample(Rt=sim_rt, data=i0)

        inf_sampled1 = inf1.sample(Rt=sim_rt, data=i0, obs=inf_sampled0)
        inf_sampled2 = inf1.sample(Rt=sim_rt, data=i0, obs=inf_sampled0)

    # Should match!
    testing.assert_array_equal(inf_sampled1, inf_sampled2)
