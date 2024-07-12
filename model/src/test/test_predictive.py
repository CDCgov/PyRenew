# -*- coding: utf-8 -*-

"""
Ensures that posterior predictive samples are not generated when no posterior samples are available.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
import pyrenew.transformation as t
import pytest
from pyrenew.deterministic import DeterministicPMF
from pyrenew.latent import (
    InfectionInitializationProcess,
    Infections,
    InitializeInfectionsZeroPad,
)
from pyrenew.metaclass import DistributionalRV
from pyrenew.model import RtInfectionsRenewalModel
from pyrenew.observation import PoissonObservation
from pyrenew.process import RtRandomWalkProcess

pmf_array = jnp.array([0.25, 0.25, 0.25, 0.25])
gen_int = DeterministicPMF(pmf_array, name="gen_int")
I0 = InfectionInitializationProcess(
    "I0_initialization",
    DistributionalRV(dist=dist.LogNormal(0, 1), name="I0"),
    InitializeInfectionsZeroPad(n_timepoints=gen_int.size()),
    t_unit=1,
)
latent_infections = Infections()
observed_infections = PoissonObservation()
rt = RtRandomWalkProcess(
    Rt0_dist=dist.TruncatedNormal(loc=1.2, scale=0.2, low=0),
    Rt_transform=t.ExpTransform().inv,
    Rt_rw_dist=dist.Normal(0, 0.025),
)
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
        model.posterior_predictive(n_timepoints_to_simulate=10)
