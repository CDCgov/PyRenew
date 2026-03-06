# numpydoc ignore=GL08

import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

from pyrenew.latent import (
    InfectionsWithSusceptibleDepletion,
)


def _infection_w_sus_depletion(
    gen_int: ArrayLike,
    Rt: ArrayLike,
    I0: ArrayLike,
    pop: ArrayLike,
    S0: ArrayLike,
) -> tuple:
    """
    Calculate the infections with susceptible depletion.
    Parameters
    ----------
    gen_int
        Generation interval.
    Rt
        Reproduction number.
    I0
        Initial infections.
    pop
        Population size.
    S0
        Initial susceptible population.

    Returns
    -------
    tuple
    """
    T = len(Rt)
    len_gen = len(gen_int)
    Inf = np.pad(I0, (0, T))
    S = np.pad(np.atleast_1d(S0), (0, T))
    R_adj = np.array(Rt)

    for t in range(T):
        infectiousness = jnp.dot(Inf[t : t + len_gen], jnp.flip(gen_int))
        Inf[t + len_gen] = S[t] * (-jnp.expm1(-Rt[t] * infectiousness / pop))
        S[t + 1] = S[t] - Inf[t + len_gen]
        R_adj[t] = np.where(infectiousness > 0, Inf[t + len_gen] / infectiousness, 0)

    return {
        "post_initialization_infections": Inf[-T:],
        "rt": R_adj,
    }


@pytest.mark.parametrize(
    "I0, gen_int, Rt, pop, S0",
    [
        (2 * jnp.ones(4), jnp.ones(4) / 4, jnp.ones(10), 100000, 10000.0),
    ],
)
def test_infections_with_sus_depletion(I0, gen_int, Rt, pop, S0):
    """
    Test the InfectionsWithSusceptibleDepletion class
    """
    Inf_w_sus_depletion = InfectionsWithSusceptibleDepletion(
        name="test_inf_sus_depletion"
    )

    res = Inf_w_sus_depletion.sample(
        Rt=Rt, I0=I0, gen_int=gen_int, S0=S0, population=pop
    )

    res_bf = _infection_w_sus_depletion(gen_int=gen_int, Rt=Rt, I0=I0, pop=pop, S0=S0)
    assert jnp.allclose(
        res.post_initialization_infections, res_bf["post_initialization_infections"]
    )
    assert jnp.allclose(res.rt, res_bf["rt"])


def test_infections_with_sus_depletion_invalid_input_shape():
    """
    Test the InfectionsWithSusceptibleDepletion class cannot
    be sampled when Rt, S0, and population have invalid input shapes
    """
    I0 = jnp.array([[5.0, 0.2]])
    gen_int = jnp.ones(1)
    Rt = jnp.ones((5, 2))
    pop = jnp.array([10, 10])
    S0 = jnp.array([10, 10], dtype=float)

    Inf_w_sus_depletion = InfectionsWithSusceptibleDepletion(
        name="test_inf_sus_depletion"
    )

    with pytest.raises(ValueError, match="S0 must match Rt batch shape exactly"):
        Inf_w_sus_depletion.sample(
            Rt=Rt, I0=I0, gen_int=gen_int, S0=jnp.array([10]), population=pop
        )

    with pytest.raises(
        ValueError, match="population must match Rt batch shape exactly"
    ):
        Inf_w_sus_depletion.sample(
            Rt=Rt, I0=I0, gen_int=gen_int, S0=S0, population=jnp.array([10])
        )
