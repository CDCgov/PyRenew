# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import numpyro
import numpyro.distributions as dist
from pyrenew import transformation as t
from pyrenew.deterministic import DeterministicPMF
from pyrenew.latent import HospitalAdmissions, Infections
from pyrenew.metaclass import DistributionalRV, TransformedRandomVariable
from pyrenew.process import SimpleRandomWalkProcess


def test_admissions_sample():
    """
    Check that a HospitalAdmissions latent process
    can be initialized and sampled from.
    """

    # Generating Rt and Infections to compute the hospital admissions
    np.random.seed(223)

    rt = TransformedRandomVariable(
        "Rt_rv",
        base_rv=SimpleRandomWalkProcess(
            name="log_rt",
            step_rv=DistributionalRV(dist.Normal(0, 0.025), "rw_step_rv"),
            init_rv=DistributionalRV(dist.Normal(0, 0.2), "init_log_Rt_rv"),
        ),
        transforms=t.ExpTransform(),
    )

    with numpyro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_rt = rt(n_steps=30)[0].value

    gen_int = jnp.array([0.5, 0.1, 0.1, 0.2, 0.1])
    i0 = 10 * jnp.ones_like(gen_int)

    inf1 = Infections()

    with numpyro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        inf_sampled1 = inf1(Rt=sim_rt, gen_int=gen_int, I0=i0)

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
        name="inf_hosp",
    )

    hosp1 = HospitalAdmissions(
        infection_to_admission_interval_rv=inf_hosp,
        infect_hosp_rate_rv=DistributionalRV(
            dist=dist.LogNormal(jnp.log(0.05), 0.05), name="IHR"
        ),
    )

    with numpyro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_hosp_1 = hosp1(latent_infections=inf_sampled1[0].value)

    testing.assert_array_less(
        sim_hosp_1.latent_hospital_admissions.value,
        inf_sampled1[0].value,
    )

test_admissions_sample()