# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import numpyro as npro
from pyrenew.latent import Infections
from pyrenew.observation import DeterministicObs
from pyrenew.process import RtRandomWalkProcess


def test_infections_as_deterministic():
    """
    Check that an InfectionObservation
    can be initialized and sampled from (deterministic)
    """

    np.random.seed(223)
    rt = RtRandomWalkProcess()
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_rt, *_ = rt.sample(constants={"n_timepoints": 30})

    gen_int = DeterministicObs(
        (jnp.array([0.25, 0.25, 0.25, 0.25]),), validate_pmf=True
    )

    inf1 = Infections(gen_int=gen_int)

    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        obs = dict(Rt=sim_rt, I0=10)
        inf_sampled1 = inf1.sample(random_variables=obs)
        inf_sampled2 = inf1.sample(random_variables=obs)

    # Should match!
    testing.assert_array_equal(
        inf_sampled1.infections, inf_sampled2.infections
    )
