# numpydoc ignore=GL08

from typing import NamedTuple

import jax.numpy as jnp
from numpy.typing import ArrayLike

import pyrenew.arrayutils as au
import pyrenew.latent.infection_functions as inf
from pyrenew.metaclass import RandomVariable


class InfectionsRtFeedbackSample(NamedTuple):
    """
    A container for holding the output from the InfectionsWithFeedback.

    Attributes
    ----------
    post_initialization_infections : ArrayLike | None, optional
        The estimated latent infections. Defaults to None.
    rt : ArrayLike | None, optional
        The adjusted reproduction number. Defaults to None.
    """

    post_initialization_infections: ArrayLike | None = None
    rt: ArrayLike | None = None

    def __repr__(self):
        return f"InfectionsSample(post_initialization_infections={self.post_initialization_infections}, rt={self.rt})"


class InfectionsWithFeedback(RandomVariable):
    r"""
    Latent infections

    This class computes infections, given Rt, initial infections, and generation
    interval.

    Parameters
    ----------
    infection_feedback_strength : RandomVariable
        Infection feedback strength.
    infection_feedback_pmf : RandomVariable
        Infection feedback pmf.

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
    ) -> None:
        """
        Default constructor for Infections class.

        Parameters
        ----------
        infection_feedback_strength : RandomVariable
            Infection feedback strength.
        infection_feedback_pmf : RandomVariable
            Infection feedback pmf.

        Returns
        -------
        None
        """

        self.validate(infection_feedback_strength, infection_feedback_pmf)

        self.infection_feedback_strength = infection_feedback_strength
        self.infection_feedback_pmf = infection_feedback_pmf

        return None

    @staticmethod
    def validate(
        inf_feedback_strength: any,
        inf_feedback_pmf: any,
    ) -> None:  # numpydoc ignore=GL08
        """
        Validates the input parameters.

        Parameters
        ----------
        inf_feedback_strength : RandomVariable
            Infection feedback strength.
        inf_feedback_pmf : RandomVariable
            Infection feedback pmf.

        Returns
        -------
        None
        """
        assert isinstance(inf_feedback_strength, RandomVariable)
        assert isinstance(inf_feedback_pmf, RandomVariable)

        return None

    def sample(
        self,
        Rt: ArrayLike,
        I0: ArrayLike,
        gen_int: ArrayLike,
        **kwargs,
    ) -> InfectionsRtFeedbackSample:
        """
        Samples infections given Rt, initial infections, and generation
        interval.

        Parameters
        ----------
        Rt : ArrayLike
            Reproduction number.
        I0 : ArrayLike
            Initial infections, as an array
            at least as long as the generation
            interval PMF.
        gen_int : ArrayLike
            Generation interval PMF.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        InfectionsWithFeedback
            Named tuple with "infections".
        """
        if I0.shape[0] < gen_int.size:
            raise ValueError(
                "Initial infections must be at least as long as the "
                f"generation interval. Got {I0.shape[0]} initial infections "
                f"and {gen_int.size} generation interval."
            )

        gen_int_rev = jnp.flip(gen_int)

        I0 = I0[-gen_int_rev.size :]

        # Sampling inf feedback strength
        inf_feedback_strength = jnp.atleast_1d(
            self.infection_feedback_strength(
                **kwargs,
            )
        )
        # Making sure inf_feedback_strength spans the Rt length
        if inf_feedback_strength.shape[0] == 1:
            inf_feedback_strength = au.pad_edges_to_match(
                x=inf_feedback_strength,
                y=Rt,
                axis=0,
            )[0]
        elif inf_feedback_strength.shape != Rt.shape:
            raise ValueError(
                "Infection feedback strength must be of length 1 "
                "or the same length as the reproduction number array. "
                f"Got {inf_feedback_strength.shape} "
                f"and {Rt.shape} respectively."
            )

        # Sampling inf feedback pmf
        inf_feedback_pmf = self.infection_feedback_pmf(**kwargs)

        inf_fb_pmf_rev = jnp.flip(inf_feedback_pmf)

        (
            post_initialization_infections,
            Rt_adj,
        ) = inf.compute_infections_from_rt_with_feedback(
            I0=I0,
            Rt_raw=Rt,
            infection_feedback_strength=inf_feedback_strength,
            reversed_generation_interval_pmf=gen_int_rev,
            reversed_infection_feedback_pmf=inf_fb_pmf_rev,
        )

        return InfectionsRtFeedbackSample(
            post_initialization_infections=post_initialization_infections,
            rt=Rt_adj,
        )
