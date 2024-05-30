# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import numpyro as npro
import numpyro.distributions as dist
from pyrenew.deterministic import DeterministicPMF
from pyrenew.latent import HospitalAdmissions, Infections
from pyrenew.metaclass import DistributionalRV
from pyrenew.process import RtRandomWalkProcess


def test_admissions_sample():
    """
    Check that a HospitalAdmissions latent process
    can be initialized and sampled from.
    """

    # Generating Rt and Infections to compute the hospital admissions
    np.random.seed(223)
    rt = RtRandomWalkProcess()
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_rt, *_ = rt.sample(duration=30)

    gen_int = jnp.array([0.25, 0.25, 0.25, 0.25])
    i0 = 10

    inf1 = Infections()

    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        inf_sampled1 = inf1.sample(Rt=sim_rt, gen_int=gen_int, I0=i0)

    # Testing the hospital admissions
    inf_hosp = DeterministicPMF(
        jnp.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.25,
                0.5,
                0.1,
                0.1,
                0.05,
            ]
        ),
    )

    hosp1 = HospitalAdmissions(
        infection_to_admission_interval=inf_hosp,
        infect_hosp_rate_dist=DistributionalRV(
            dist=dist.LogNormal(jnp.log(0.05), 0.05), name="IHR"
        ),
    )

    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_hosp_1 = hosp1.sample(latent=inf_sampled1[0])

    testing.assert_array_less(
        sim_hosp_1.predicted,
        inf_sampled1[0],
    )
