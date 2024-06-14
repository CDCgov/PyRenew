# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import pyrenew.arrayutils as au
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
    observed_infections : ArrayLike | None, optional
        The sampled infections. Defaults to None.
    """

    Rt: float | None = None
    latent_infections: ArrayLike | None = None
    observed_infections: ArrayLike | None = None

    def __repr__(self):
        return (
            f"RtInfectionsRenewalSample(Rt={self.Rt}, "
            f"latent_infections={self.latent_infections}, "
            f"observed_infections={self.observed_infections})"
        )


class RtInfectionsRenewalModel(Model):
    """
    Basic Renewal Model (Infections + Rt)

    The basic renewal model consists of a sampler of two steps: Sample from
    Rt and then used that to sample the infections.
    """

    def __init__(
        self,
        latent_infections_rv: RandomVariable,
        gen_int_rv: RandomVariable,
        I0_rv: RandomVariable,
        Rt_process_rv: RandomVariable,
        infection_obs_process_rv: RandomVariable = None,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        latent_infections_rv : RandomVariable
            Infections latent process (e.g.,
            pyrenew.latent.Infections.).
        gen_int_rv : RandomVariable
            The generation interval.
        I0_rv : RandomVariable
            The initial infections.
        Rt_process_rv : RandomVariable
            The sample function of the process should return a tuple where the
            first element is the drawn Rt.
        infection_obs_process_rv : RandomVariable
            Infections observation process (e.g.,
            pyrenew.observations.Poisson.).

        Returns
        -------
        None
        """

        if infection_obs_process_rv is None:
            infection_obs_process_rv = NullObservation()

        RtInfectionsRenewalModel.validate(
            gen_int_rv=gen_int_rv,
            I0_rv=I0_rv,
            latent_infections_rv=latent_infections_rv,
            infection_obs_process_rv=infection_obs_process_rv,
            Rt_process_rv=Rt_process_rv,
        )

        self.gen_int_rv = gen_int_rv
        self.I0_rv = I0_rv
        self.latent_infections_rv = latent_infections_rv
        self.infection_obs_process_rv = infection_obs_process_rv
        self.Rt_process_rv = Rt_process_rv

    @staticmethod
    def validate(
        gen_int_rv: any,
        I0_rv: any,
        latent_infections_rv: any,
        infection_obs_process_rv: any,
        Rt_process_rv: any,
    ) -> None:
        """
        Verifies types and status (RV) of the generation interval, initial
        infections, latent and observed infections, and the Rt process.

        Parameters
        ----------
        gen_int_rv : any
            The generation interval. Expects RandomVariable.
        I0_rv : any
            The initial infections. Expects RandomVariable.
        latent_infections_rv : any
            Infections latent process. Expects RandomVariable.
        infection_obs_process_rv : any
            Infections observation process. Expects RandomVariable.
        Rt_process_rv : any
            The sample function of the process should return a tuple where the
            first element is the drawn Rt. Expects RandomVariable.

        Returns
        -------
        None

        See Also
        --------
        _assert_sample_and_rtype : Perform type-checking and verify RV
        """
        _assert_sample_and_rtype(gen_int_rv, skip_if_none=False)
        _assert_sample_and_rtype(I0_rv, skip_if_none=False)
        _assert_sample_and_rtype(latent_infections_rv, skip_if_none=False)
        _assert_sample_and_rtype(infection_obs_process_rv, skip_if_none=False)
        _assert_sample_and_rtype(Rt_process_rv, skip_if_none=False)
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
        return self.Rt_process_rv.sample(**kwargs)

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
        return self.gen_int_rv.sample(**kwargs)

    def sample_I0(
        self,
        **kwargs,
    ) -> tuple:
        """
        Samples the initial infections

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample_I0 calls, should there be any.

        Returns
        -------
        tuple
        """
        return self.I0_rv.sample(**kwargs)

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
        return self.latent_infections_rv.sample(**kwargs)

    def sample_infection_obs_process(
        self,
        observed_infections_mean: ArrayLike,
        data_observed_infections: ArrayLike | None = None,
        name: str | None = None,
        **kwargs,
    ) -> tuple:
        """
        Sample observed infections according
        to an observation process, if one has
        been specified.

        Parameters
        ----------
        observed_infections_mean : ArrayLike
            The mean of the observed infections distribution.
        data_observed_infections : ArrayLike | None, optional
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
        return self.infection_obs_process_rv.sample(
            mu=observed_infections_mean,
            obs=data_observed_infections,
            name=name,
            **kwargs,
        )

    def sample(
        self,
        n_timepoints_to_simulate: int | None = None,
        data_observed_infections: ArrayLike | None = None,
        padding: int = 0,
        **kwargs,
    ) -> RtInfectionsRenewalSample:
        """
        Sample from the Basic Renewal Model

        Parameters
        ----------
        n_timepoints_to_simulate : int, optional
            Number of timepoints to sample.
        data_observed_infections : ArrayLike | None, optional
            Observed infections. Defaults to None.
        padding : int, optional
            Number of padding timepoints to add to the beginning of the
            simulation. Defaults to 0.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, if any

        Notes
        -----
        Either `data_observed_infections` or `n_timepoints_to_simulate` must be specified, not both.

        Returns
        -------
        RtInfectionsRenewalSample
        """

        if (
            n_timepoints_to_simulate is None
            and data_observed_infections is None
        ):
            raise ValueError(
                "Either n_timepoints_to_simulate or data_observed_infections "
                "must be passed."
            )
        elif (
            n_timepoints_to_simulate is not None
            and data_observed_infections is not None
        ):
            raise ValueError(
                "Cannot pass both n_timepoints_to_simulate and data_observed_infections."
            )
        elif n_timepoints_to_simulate is None:
            n_timepoints = len(data_observed_infections)
        else:
            n_timepoints = n_timepoints_to_simulate
        # Sampling from Rt (possibly with a given Rt, depending on
        # the Rt_process (RandomVariable) object.)
        Rt, *_ = self.sample_rt(
            n_timepoints=n_timepoints,
            **kwargs,
        )

        # Getting the generation interval
        gen_int, *_ = self.sample_gen_int(**kwargs)

        # Sampling initial infections
        I0, *_ = self.sample_I0(**kwargs)
        I0_size = I0.size
        # Sampling from the latent process
        latent_infections, *_ = self.sample_infections_latent(
            Rt=Rt,
            gen_int=gen_int,
            I0=I0,
            **kwargs,
        )

        if data_observed_infections is None:
            (
                observed_infections,
                *_,
            ) = self.sample_infection_obs_process(
                observed_infections_mean=latent_infections,
                data_observed_infections=data_observed_infections,
                **kwargs,
            )
        else:
            data_observed_infections = au.pad_x_to_match_y(
                data_observed_infections,
                latent_infections,
                jnp.nan,
                pad_direction="start",
            )

            (
                observed_infections,
                *_,
            ) = self.sample_infection_obs_process(
                observed_infections_mean=latent_infections[
                    I0_size + padding :
                ],
                data_observed_infections=data_observed_infections[
                    I0_size + padding :
                ],
                **kwargs,
            )

        observed_infections = au.pad_x_to_match_y(
            observed_infections,
            latent_infections,
            jnp.nan,
            pad_direction="start",
        )

        Rt = au.pad_x_to_match_y(
            Rt, latent_infections, jnp.nan, pad_direction="start"
        )
        return RtInfectionsRenewalSample(
            Rt=Rt,
            latent_infections=latent_infections,
            observed_infections=observed_infections,
        )
