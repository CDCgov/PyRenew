# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from typing import NamedTuple

import jax.numpy as jnp
import numpyro as npro
import pyrenew.latent.infection_functions as inf
from numpy.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


class InfectionsRtFeedbackSample(NamedTuple):
    """
    A container for holding the output from the InfectionsRtFeedback.

    Attributes
    ----------
    infections : ArrayLike | None, optional
        The estimated latent infections. Defaults to None.
    rt : ArrayLike | None, optional
        The adjusted reproduction number. Defaults to None.
    """

    infections: ArrayLike | None = None
    rt: ArrayLike | None = None

    def __repr__(self):
        return f"InfectionsSample(infections={self.infections}, rt={self.rt})"


class InfectionsRtFeedback(RandomVariable):
    r"""
    Latent infections

    This class samples infections given Rt, initial infections, and generation
    interval.

    Parameters
    ----------
    infection_feedback_strength : RandomVariable
        Infection feedback strength.
    infection_feedback_pmf : RandomVariable
        Infection feedback pmf.
    infections_mean_varname : str, optional
        Name to be assigned to the deterministic variable in the model.
        Defaults to "latent_infections".

    Notes
    -----
    This function implements the following renewal process (reproduced from
    :func:`pyrenew.latent.infection_functions.sample_infections_with_feedback`):

    .. math::

        I(t) & = \mathcal{R}(t)\sum_{\tau=1}^{T_g}I(t - \tau)g(\tau)

        \mathcal{R}(t) & = \mathcal{R}^u(t)\exp\left(-\gamma(t)\
            \sum_{\tau=1}^{T_f}I(t - \tau)f(\tau)\right)

    where :math:`\mathcal{R}(t)` is the reproductive number, :math:`\gamma(t)`
    is the infection feedback strength, :math:`T_g` is the max-length of the
    generation interval, :math:`\mathcal{R}^u(t)` is the raw reproduction
    number, :math:`f(t)` is the infection feedback pmf, and :math:`T_f`
    is the max-length of the infection feedback pmf.
    """

    def __init__(
        self,
        infection_feedback_strength: RandomVariable,
        infection_feedback_pmf: RandomVariable,
        infections_mean_varname: str = "latent_infections",
    ) -> None:
        """
        Default constructor for Infections class.

        Parameters
        ----------
        infection_feedback_strength : RandomVariable
            Infection feedback strength.
        infection_feedback_pmf : RandomVariable
            Infection feedback pmf.
        infections_mean_varname : str, optional
            Name to be assigned to the deterministic variable in the model.
            Defaults to "latent_infections".

        Returns
        -------
        None
        """
        self.infection_feedback_strength = infection_feedback_strength
        self.infection_feedback_pmf = infection_feedback_pmf
        self.infections_mean_varname = infections_mean_varname

        return None

    @staticmethod
    def validate() -> None:  # numpydoc ignore=GL08
        return None

    def sample(
        self,
        Rt: ArrayLike,
        I0: ArrayLike,
        gen_int: ArrayLike,
        **kwargs,
    ) -> tuple:
        """
        Samples infections given Rt, initial infections, and generation
        interval.

        Parameters
        ----------
        Rt : ArrayLike
            Reproduction number.
        I0 : ArrayLike
            Initial infections.
        gen_int : ArrayLike
            Generation interval.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        InfectionsRtFeedback
            Named tuple with "infections".
        """

        gen_int_rev = jnp.flip(gen_int)

        # Adjusting sizes
        if gen_int_rev.size > Rt.size:
            n_lead = gen_int_rev.size - Rt.size
            Rt = jnp.hstack([Rt, jnp.zeros(n_lead)])
        else:
            n_lead = Rt.size - gen_int_rev.size
            gen_int_rev = jnp.hstack([jnp.zeros(n_lead), gen_int_rev])

        n_lead = gen_int_rev.size - 1
        I0_vec = jnp.hstack([jnp.zeros(n_lead), I0])

        # Sampling inf feedback strength and adjusting the shape
        inf_feedback_strength, *_ = self.infection_feedback_strength.sample(
            **kwargs,
        )
        n_lead = Rt.size - inf_feedback_strength.size
        inf_feedback_strength = jnp.hstack(
            [inf_feedback_strength, jnp.zeros(n_lead)]
        )

        # Sampling inf feedback and adjusting the shape
        inf_feedback_pmf, *_ = self.infection_feedback_pmf.sample(**kwargs)
        n_lead = Rt.size - inf_feedback_pmf.size
        inf_feedback_pmf = jnp.hstack([inf_feedback_pmf, jnp.zeros(n_lead)])

        all_infections, Rt_adj = inf.sample_infections_with_feedback(
            I0=I0_vec,
            Rt_raw=Rt,
            infection_feedback_strength=inf_feedback_strength,
            generation_interval_pmf=gen_int_rev,
            infection_feedback_pmf=inf_feedback_pmf,
        )

        npro.deterministic("Rt_adjusted", Rt_adj)

        return InfectionsRtFeedbackSample(
            infections=all_infections,
            rt=Rt_adj,
        )
