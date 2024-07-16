# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import numpyro as npro
import numpyro.distributions as dist
import pyrenew.transformation as t
import pytest
from pyrenew.latent import Infections
from pyrenew.process import RtRandomWalkProcess


def test_infections_as_deterministic():
    """
    Test that the Infections class samples the same infections when
    the same seed is used.
    """

    np.random.seed(223)
    rt = RtRandomWalkProcess(
        Rt0_dist=dist.TruncatedNormal(loc=1.2, scale=0.2, low=0),
        Rt_transform=t.ExpTransform().inv,
        Rt_rw_dist=dist.Normal(0, 0.025),
    )
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_rt, *_ = rt(n_timepoints=30)

    gen_int = jnp.array([0.25, 0.25, 0.25, 0.25])

    inf1 = Infections()

    obs = dict(
        Rt=sim_rt,
        I0=jnp.zeros(gen_int.size),
        gen_int=gen_int,
    )
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        inf_sampled1 = inf1(**obs)
        inf_sampled2 = inf1(**obs)

    testing.assert_array_equal(
        inf_sampled1.post_initialization_infections,
        inf_sampled2.post_initialization_infections,
    )

    # Check that Initial infections vector must be at least as long as the generation interval.
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        with pytest.raises(ValueError):
            obs["I0"] = jnp.array([1])
            inf1(**obs)
