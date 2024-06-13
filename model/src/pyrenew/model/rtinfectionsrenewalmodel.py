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
    sampled_observed_infections : ArrayLike | None, optional
        The sampled infections. Defaults to None.
    """

    Rt: float | None = None
    latent_infections: ArrayLike | None = None
    sampled_observed_infections: ArrayLike | None = None

    def __repr__(self):
        return (
            f"RtInfectionsRenewalSample(Rt={self.Rt}, "
            f"latent_infections={self.latent_infections}, "
            f"sampled_observed_infections={self.sampled_observed_infections})"
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
        Rt, *_ = self.Rt_process_rv.sample(
            n_timepoints=n_timepoints,
            **kwargs,
        )

        # Getting the generation interval
        gen_int, *_ = self.sample_gen_int(**kwargs)

        # Sampling initial infections
        I0, *_ = self.I0_rv.sample(**kwargs)
        I0_size = I0.size
        # Sampling from the latent process
        latent_infections, *_ = self.latent_infections_rv.sample(
            Rt=Rt,
            gen_int=gen_int,
            I0=I0,
            **kwargs,
        )

        if observed_infections is None:
            (
                sampled_observed_infections,
                *_,
            ) = self.infection_obs_process_rv.sample(
                mu=latent_infections,
                obs=observed_infections,
                **kwargs,
            )
        else:
            observed_infections = au.pad_x_to_match_y(
                observed_infections,
                latent_infections,
                jnp.nan,
                pad_direction="start",
            )

            (
                sampled_observed_infections,
                *_,
            ) = self.infection_obs_process_rv.sample(
                mu=latent_infections[I0_size + padding :],
                obs=observed_infections[I0_size + padding :],
                **kwargs,
            )

        sampled_observed_infections = au.pad_x_to_match_y(
            sampled_observed_infections,
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
            sampled_observed_infections=sampled_observed_infections,
        )
