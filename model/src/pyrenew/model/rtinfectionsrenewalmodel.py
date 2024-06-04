# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import pyrenew.datautils as du
from numpy.typing import ArrayLike
from pyrenew.deterministic import NullObservation
from pyrenew.metaclass import Model, RandomVariable, _assert_sample_and_rtype


# Output class of the RtInfectionsRenewalModel
class RtInfectionsRenewalSample(NamedTuple):
    """
    A container for holding the output from `model.RtInfectionsRenewalModel.sample()`.

    Attributes
    ----------
    Rt : float | None, optional
        The reproduction number over time. Defaults to None.
    latent_infections : ArrayLike | None, optional
        The estimated latent infections. Defaults to None.
    sampled_infections : ArrayLike | None, optional
        The sampled infections. Defaults to None.
    """

    Rt: float | None = None
    latent_infections: ArrayLike | None = None
    sampled_infections: ArrayLike | None = None

    def __repr__(self):
        return (
            f"RtInfectionsRenewalSample(Rt={self.Rt}, "
            f"latent_infections={self.latent_infections}, "
            f"sampled_infections={self.sampled_infections})"
        )


class RtInfectionsRenewalModel(Model):
    """
    Basic Renewal Model (Infections + Rt)

    The basic renewal model consists of a sampler of two steps: Sample from
    Rt and then used that to sample the infections.
    """

    def __init__(
        self,
        latent_infections: RandomVariable,
        gen_int: RandomVariable,
        I0: RandomVariable,
        Rt_process: RandomVariable,
        observation_process: RandomVariable = None,
    ) -> None:
        """
        Default constructor

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
        observation_process : RandomVariable
            Infections observation process (e.g.,
            pyrenew.observations.Poisson.).

        Returns
        -------
        None
        """

        if observation_process is None:
            observation_process = NullObservation()

        RtInfectionsRenewalModel.validate(
            gen_int=gen_int,
            i0=I0,
            latent_infections=latent_infections,
            observation_process=observation_process,
            Rt_process=Rt_process,
        )

        self.gen_int = gen_int
        self.i0 = I0
        self.latent_infections = latent_infections
        self.observation_process = observation_process
        self.Rt_process = Rt_process

    @staticmethod
    def validate(
        gen_int: any,
        i0: any,
        latent_infections: any,
        observation_process: any,
        Rt_process: any,
    ) -> None:
        """
        Verifies types and status (RV) of the generation interval, initial
        infections, latent and observed infections, and the Rt process.

        Parameters
        ----------
        gen_int : any
            The generation interval. Expects RandomVariable.
        i0 : any
            The initial infections. Expects RandomVariable.
        latent_infections : any
            Infections latent process. Expects RandomVariable.
        observation_process : any
            Infections observation process. Expects RandomVariable.
        Rt_process : any
            The sample function of the process should return a tuple where the
            first element is the drawn Rt. Expects RandomVariable.

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
        _assert_sample_and_rtype(observation_process, skip_if_none=False)
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
        """
        return self.latent_infections.sample(**kwargs)

    def sample_infections_obs(
        self,
        predicted: ArrayLike,
        observed_infections: ArrayLike | None = None,
        name: str | None = None,
        **kwargs,
    ) -> tuple:
        """
        Sample observed infections according
        to an observation process, if one has
        been specified.

        Parameters
        ----------
        predicted : ArrayLike
            The predicted infecteds.
        observed_infections : ArrayLike | None, optional
            The observed infection values, if any, for inference. Defaults to None.
        name : str | None, optional
            Name of the random variable passed to the RandomVariable. Defaults to None.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample() calls, should there be any.

        Returns
        -------
        tuple
        """
        return self.observation_process.sample(
            predicted=predicted,
            obs=observed_infections,
            name=name,
            **kwargs,
        )

    def sample(
        self,
        n_timepoints_to_simulate: int | None = None,
        observed_infections: ArrayLike | None = None,
        padding: int = 0,
        **kwargs,
    ) -> RtInfectionsRenewalSample:
        """
        Sample from the Basic Renewal Model

        Parameters
        ----------
        n_timepoints_to_simulate : int, optional
            Number of timepoints to sample.
        observed_infections : ArrayLike | None, optional
            Observed infections. Defaults to None.
        padding : int, optional
            Number of padding timepoints to add to the beginning of the
            simulation. Defaults to 0.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, if any

        Notes
        -----
        Either `observed_admissions` or `n_timepoints_to_simulate` must be specified, not both.

        Returns
        -------
        RtInfectionsRenewalSample
        """

        if n_timepoints_to_simulate is None and observed_infections is None:
            raise ValueError(
                "Either n_timepoints_to_simulate or observed_infections "
                "must be passed."
            )
        elif (
            n_timepoints_to_simulate is not None
            and observed_infections is not None
        ):
            raise ValueError(
                "Cannot pass both n_timepoints_to_simulate and observed_infections."
            )
        elif n_timepoints_to_simulate is None:
            n_timepoints = len(observed_infections)
        else:
            n_timepoints = n_timepoints_to_simulate
        # Sampling from Rt (possibly with a given Rt, depending on
        # the Rt_process (RandomVariable) object.)
        Rt, *_ = self.sample_rt(
            duration=n_timepoints,
            **kwargs,
        )

        # Getting the generation interval
        gen_int, *_ = self.sample_gen_int(**kwargs)

        # Sampling initial infections
        i0, *_ = self.sample_i0(**kwargs)

        # Padding i0 to match gen_int
        # PADDING SHOULD BE REMOVED ONCE
        # https://github.com/CDCgov/multisignal-epi-inference/pull/124
        # is merged.
        # SEE ALSO:
        # https://github.com/CDCgov/multisignal-epi-inference/pull/123#discussion_r1612337288
        i0 = du.pad_x_to_match_y(x=i0, y=gen_int, fill_value=0.0)

        # Sampling from the latent process
        latent, *_ = self.sample_infections_latent(
            Rt=Rt,
            gen_int=gen_int,
            I0=i0,
            **kwargs,
        )

        # Using the predicted infections to sample from the observation process
        if (observed_infections is not None) and (padding > 0):
            sampled_pad = jnp.repeat(jnp.nan, padding)

            sampled_obs, *_ = self.sample_infections_obs(
                predicted=latent[padding:],
                observed_infections=observed_infections[padding:],
                **kwargs,
            )

            sampled = jnp.hstack([sampled_pad, sampled_obs])

        else:
            sampled, *_ = self.sample_infections_obs(
                predicted=latent,
                observed_infections=observed_infections,
                **kwargs,
            )

        return RtInfectionsRenewalSample(
            Rt=Rt,
            latent_infections=latent,
            sampled_infections=sampled,
        )
