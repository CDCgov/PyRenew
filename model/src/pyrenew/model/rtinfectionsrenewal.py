# -*- coding: utf-8 -*-

from collections import namedtuple
from typing import Optional

from numpy.typing import ArrayLike
from pyrenew.metaclass import Model, RandomVariable, _assert_sample_and_rtype

# Output class of the RtInfectionsRenewalModel
RtInfectionsRenewalSample = namedtuple(
    "InfectModelSample",
    ["Rt", "latent_infections", "sampled_infections"],
    defaults=[None, None, None],
)
RtInfectionsRenewalSample.__doc__ = """
A container for holding the output from RtInfectionsRenewalModel.sample().

Attributes
----------
Rt : float or None
    The reproduction number over time. Defaults to None.
latent_infections : ArrayLike or None
    The estimated latent infections. Defaults to None.
sampled_infections : ArrayLike or None
    The sampled infections. Defaults to None.

Notes
-----
"""


class RtInfectionsRenewalModel(Model):
    """Basic Renewal Model (Infections + Rt)

    The basic renewal model consists of a sampler of two steps: Sample from
    Rt and then used that to sample the infections.
    """

    def __init__(
        self,
        latent_infections: RandomVariable,
        gen_int: RandomVariable,
        I0: RandomVariable,
        Rt_process: RandomVariable,
        observed_infections: RandomVariable,
    ) -> None:
        """Default constructor

        Parameters
        ----------
        latent_infections : RandomVariable
            Infections latent process (e.g.,
            pyrenew.latent.Infections.).
        gen_int : RandomVariable
            The generation interval.
        I0 : RandomVariable
            The initial infections.
        Rt_process : RandomVariable
            The sample function of the process should return a tuple where the
            first element is the drawn Rt.
        observed_infections : RandomVariable
            Infections observation process (e.g.,
            pyrenew.observations.Poisson.).

        Returns
        -------
        None
        """

        RtInfectionsRenewalModel.validate(
            gen_int=gen_int,
            i0=I0,
            latent_infections=latent_infections,
            observed_infections=observed_infections,
            Rt_process=Rt_process,
        )

        self.gen_int = gen_int
        self.i0 = I0
        self.latent_infections = latent_infections
        self.observed_infections = observed_infections
        self.Rt_process = Rt_process

    @staticmethod
    def validate(
        gen_int,
        i0,
        latent_infections,
        observed_infections,
        Rt_process,
    ) -> None:
        """
        Verifies types and status (RV) of the generation interval, initial
        infections, latent and observed infections, and the Rt process.

        Parameters
        ----------
        latent_hospitalizations : ArrayLike
            The latent process for the hospitalizations.
        observed_hospitalizations : ArrayLike
            The observed hospitalizations.

        Returns
        -------
        None

        See Also
        --------
        _assert_sample_and_rtype : Perform type-checking and verify RV
        """
        _assert_sample_and_rtype(gen_int, skip_if_none=False)
        _assert_sample_and_rtype(i0, skip_if_none=False)
        _assert_sample_and_rtype(latent_infections, skip_if_none=False)
        _assert_sample_and_rtype(observed_infections, skip_if_none=True)
        _assert_sample_and_rtype(Rt_process, skip_if_none=False)
        return None

    def sample_rt(
        self,
        **kwargs,
    ) -> tuple:
        """
        Samples the Rt process

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample_rt calls, should there be any.

        Returns
        -------
        tuple

        Notes
        -----
        TODO: More information in Returns.
        """
        return self.Rt_process.sample(**kwargs)

    def sample_gen_int(
        self,
        **kwargs,
    ) -> tuple:
        """
        Samples the generation interval

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample_gen_int calls, should there be any.

        Returns
        -------
        tuple

        Notes
        -----
        TODO: More information in Returns.
        """
        return self.gen_int.sample(**kwargs)

    def sample_i0(
        self,
        **kwargs,
    ) -> tuple:
        """
        Samples the initial infections

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample_i0 calls, should there be any.

        Returns
        -------
        tuple

        Notes
        -----
        TODO: More information in Returns.
        """
        return self.i0.sample(**kwargs)

    def sample_infections_latent(
        self,
        **kwargs,
    ) -> tuple:
        """
        Samples the latent infections

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample_infections_latent calls, should there be any.

        Returns
        -------
        tuple

        Notes
        -----
        TODO: More information in Returns.
        """
        return self.latent_infections.sample(**kwargs)

    def sample_infections_obs(
        self,
        predicted: ArrayLike,
        observed_infections: Optional[ArrayLike] = None,
        **kwargs,
    ) -> tuple:
        """Sample number of hospitalizations

        Parameters
        ----------
        predicted : ArrayLike
            The predicted infecteds.
        observed_hospitalizations : ArrayLike, optional
            The observed values of hospital admissions, if any, for inference. Defaults to None.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample() calls, should there be any.

        Returns
        -------
        tuple

        See Also
        --------
        observed_infections.sample : For sampling observed infections

        Notes
        -----
        TODO: Include example(s) here.
        """
        return self.observed_infections.sample(
            predicted=predicted,
            obs=observed_infections,
            **kwargs,
        )

    def sample(
        self,
        n_timepoints: int,
        observed_infections: Optional[ArrayLike] = None,
        **kwargs,
    ) -> RtInfectionsRenewalSample:
        """Sample from the Basic Renewal Model

        Parameters
        ----------
        n_timepoints : int
            Number of timepoints to sample.
        observed_infections : ArrayLike, optional
            Observed infections.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, if any

        Returns
        -------
        RtInfectionsRenewalSample

        Notes
        -----
        TODO: Add See Also.
        """

        # Sampling from Rt (possibly with a given Rt, depending on
        # the Rt_process (RandomVariable) object.)
        Rt, *_ = self.sample_rt(
            n_timepoints=n_timepoints,
            **kwargs,
        )

        # Getting the generation interval
        gen_int, *_ = self.sample_gen_int(**kwargs)

        # Sampling initial infections
        i0, *_ = self.sample_i0(**kwargs)

        # Sampling from the latent process
        latent, *_ = self.sample_infections_latent(
            Rt=Rt,
            gen_int=gen_int,
            I0=i0,
            **kwargs,
        )

        # Using the predicted infections to sample from the observation process
        sampled, *_ = self.sample_infections_obs(
            predicted=latent, observed_infections=observed_infections, **kwargs
        )

        return RtInfectionsRenewalSample(
            Rt=Rt,
            latent_infections=latent,
            sampled_infections=sampled,
        )
