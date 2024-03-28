# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import numpyro as npro
from pyrenew.deterministic import DeterministicPMF
from pyrenew.latent import HospitalAdmissions, Infections
from pyrenew.process import RtRandomWalkProcess


def test_hospitalizations_sample():
    """
    Check that an InfectionObservation
    can be initialized and sampled from (deterministic)
    """

    # Generating Rt and Infections to compute the hospitalizations
    np.random.seed(223)
    rt = RtRandomWalkProcess()
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_rt, *_ = rt.sample(constants={"n_timepoints": 30})

    gen_int = DeterministicPMF(
        (jnp.array([0.25, 0.25, 0.25, 0.25]),),
    )
    inf1 = Infections(gen_int=gen_int)

    i0 = dict(I0=10)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        inf_sampled1 = inf1.sample(random_variables=dict(Rt=sim_rt, data=i0))

    # Testing the hospitalizations
    inf_hosp = DeterministicPMF(
        (
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
        ),
    )
    hosp1 = HospitalAdmissions(inf_hosp_int=inf_hosp)

    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_hosp_1 = hosp1.sample(
            random_variables=dict(infections=inf_sampled1[0])
        )

    testing.assert_array_less(
        sim_hosp_1.predicted,
        inf_sampled1[0],
    )
