# -*- coding: utf-8 -*-

"""
Ensures that posterior predictive samples are not generated
when no posterior samples are available.
"""

from test.utils import SimpleRt

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pyrenew.deterministic import DeterministicPMF
from pyrenew.latent import (
    InfectionInitializationProcess,
    Infections,
    InitializeInfectionsZeroPad,
)
from pyrenew.model import RtInfectionsRenewalModel
from pyrenew.observation import PoissonObservation
from pyrenew.randomvariable import DistributionalVariable

pmf_array = jnp.array([0.25, 0.1, 0.2, 0.45])
gen_int = DeterministicPMF(name="gen_int", value=pmf_array)
I0 = InfectionInitializationProcess(
    "I0_initialization",
    DistributionalVariable(name="I0", distribution=dist.LogNormal(0, 1)),
    InitializeInfectionsZeroPad(n_timepoints=gen_int.size()),
    t_unit=1,
)
latent_infections = Infections()
observed_infections = PoissonObservation("poisson_rv")
rt = SimpleRt()

model = RtInfectionsRenewalModel(
    I0_rv=I0,
    gen_int_rv=gen_int,
    latent_infections_rv=latent_infections,
    infection_obs_process_rv=observed_infections,
    Rt_process_rv=rt,
)


def test_posterior_predictive_no_posterior():
    """
    Tests that posterior predictive samples are not generated when
    no posterior samples are available.
    """
    with pytest.raises(ValueError, match="No posterior"):
        model.posterior_predictive(n_datapoints=10)
