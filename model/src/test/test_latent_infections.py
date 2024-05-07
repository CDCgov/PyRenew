# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import numpyro as npro
from pyrenew.latent import Infections
from pyrenew.process import RtRandomWalkProcess


def test_infections_as_deterministic():
    """
    Check that an InfectionObservation
    can be initialized and sampled from (deterministic)
    """

    np.random.seed(223)
    rt = RtRandomWalkProcess()
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_rt, *_ = rt.sample(n_timepoints=30)

    gen_int = jnp.array([0.25, 0.25, 0.25, 0.25])

    inf1 = Infections()

    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        obs = dict(Rt=sim_rt, I0=10, gen_int=gen_int)
        inf_sampled1 = inf1.sample(**obs)
        inf_sampled2 = inf1.sample(**obs)

    # Should match!
    testing.assert_array_equal(
        inf_sampled1.infections, inf_sampled2.infections
    )
