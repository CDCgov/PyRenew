# numpydoc ignore=GL08

from test.utils import SimpleRt

import jax.numpy as jnp
import numpy.testing as testing
import numpyro
import numpyro.distributions as dist

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import HospitalAdmissions, Infections
from pyrenew.randomvariable import DistributionalVariable


def test_admissions_sample():
    """
    Check that a HospitalAdmissions latent process
    can be initialized and sampled from.
    """

    # Generating Rt and Infections to compute the hospital admissions

    rt = SimpleRt()
    n_steps = 30

    with numpyro.handlers.seed(rng_seed=223):
        sim_rt = rt(n=n_steps)

    gen_int = jnp.array([0.5, 0.1, 0.1, 0.2, 0.1])
    inf_hosp_int_array = jnp.array(
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
    )
    i0 = 10 * jnp.ones_like(inf_hosp_int_array)
    inf1 = Infections()

    with numpyro.handlers.seed(rng_seed=223):
        inf_sampled1 = inf1(Rt=sim_rt, gen_int=gen_int, I0=i0)

    # Testing the hospital admissions
    inf_hosp = DeterministicPMF(
        name="inf_hosp",
        value=inf_hosp_int_array,
    )

    hosp1 = HospitalAdmissions(
        infection_to_admission_interval_rv=inf_hosp,
        infection_hospitalization_ratio_rv=DistributionalVariable(
            name="IHR", distribution=dist.LogNormal(jnp.log(0.05), 0.05)
        ),
    )

    with numpyro.handlers.seed(rng_seed=223):
        sim_hosp_1 = hosp1(
            latent_infections=jnp.hstack(
                [i0, inf_sampled1.post_initialization_infections]
            )
        )

    testing.assert_array_less(
        sim_hosp_1.latent_hospital_admissions[-n_steps:],
        inf_sampled1,
    )
    inf_hosp2 = jnp.ones(30)
    inf_hosp2 = DeterministicPMF("i2h", inf_hosp2 / sum(inf_hosp2))

    dow_effect = jnp.array([1, 1, 1, 1, 0.5, 0.5, 0.5])
    dow_effect = DeterministicPMF(
        name="dow_effect",
        value=dow_effect / sum(dow_effect),
    )

    dow_effect_wrong = DeterministicPMF(
        name="dow_effect",
        value=jnp.array([0.3, 0.3, 1 - 0.6]),
    )
    hosp2a = HospitalAdmissions(
        infection_to_admission_interval_rv=inf_hosp2,
        infection_hospitalization_ratio_rv=DeterministicVariable("ihr", 1),
        day_of_week_effect_rv=dow_effect,
        obs_data_first_day_of_the_week=0,
    )

    hosp2b = HospitalAdmissions(
        infection_to_admission_interval_rv=inf_hosp2,
        infection_hospitalization_ratio_rv=DeterministicVariable("ihr", 1),
        day_of_week_effect_rv=dow_effect,
        obs_data_first_day_of_the_week=2,
    )

    hosp3b = HospitalAdmissions(
        infection_to_admission_interval_rv=inf_hosp2,
        infection_hospitalization_ratio_rv=DeterministicVariable("ihr", 1),
        day_of_week_effect_rv=dow_effect_wrong,
        obs_data_first_day_of_the_week=2,
    )

    inf_sampled2 = jnp.ones(30)

    with numpyro.handlers.seed(rng_seed=223):
        sim_hosp_2a = hosp2a(latent_infections=inf_sampled2).multiplier

    with numpyro.handlers.seed(rng_seed=223):
        sim_hosp_2b = hosp2b(latent_infections=inf_sampled2).multiplier

    with numpyro.handlers.seed(rng_seed=223):
        with testing.assert_raises(ValueError):
            hosp3b(latent_infections=inf_sampled2).multiplier

    testing.assert_array_equal(
        sim_hosp_2a[2 : (sim_hosp_2b.size - 2)],
        sim_hosp_2b[: (sim_hosp_2b.size - 4)],
    )
