# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpyro as npro
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
    Rt : ArrayLike | None, optional
        The reproduction number over time. Defaults to None.
    latent_infections : ArrayLike | None, optional
        The estimated latent infections. Defaults to None.
    observed_infections : ArrayLike | None, optional
        The sampled infections. Defaults to None.
    """

    Rt: ArrayLike | None = None
    latent_infections: ArrayLike | None = None
    observed_infections: ArrayLike | None = None

    def __repr__(self):
        return (
            f"RtInfectionsRenewalSample("
            f"Rt={self.Rt}, "
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

    def sample(
        self,
        n_datapoints: int | None = None,
        data_observed_infections: ArrayLike | None = None,
        padding: int = 0,
        **kwargs,
    ) -> RtInfectionsRenewalSample:
        """
        Sample from the Basic Renewal Model

        Parameters
        ----------
        n_datapoints : int, optional
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
        Either `data_observed_infections` or `n_datapoints`
        must be specified, not both.

        Returns
        -------
        RtInfectionsRenewalSample
        """

        if n_datapoints is None and data_observed_infections is None:
            raise ValueError(
                "Either n_datapoints or data_observed_infections "
                "must be passed."
            )
        elif n_datapoints is not None and data_observed_infections is not None:
            raise ValueError(
                "Cannot pass both n_datapoints and data_observed_infections."
            )
        elif n_datapoints is None:
            n_timepoints = len(data_observed_infections) + padding
        else:
            n_timepoints = n_datapoints + padding
        # Sampling from Rt (possibly with a given Rt, depending on
        # the Rt_process (RandomVariable) object.)
        Rt, *_ = self.Rt_process_rv(
            n_steps=n_timepoints,
            **kwargs,
        )

        # Getting the generation interval
        gen_int, *_ = self.gen_int_rv(**kwargs)

        # Sampling initial infections
        I0, *_ = self.I0_rv(**kwargs)
        # Sampling from the latent process
        (
            post_initialization_latent_infections,
            *_,
        ) = self.latent_infections_rv(
            Rt=Rt,
            gen_int=gen_int,
            I0=I0,
            **kwargs,
        )

        observed_infections, *_ = self.infection_obs_process_rv(
            mu=post_initialization_latent_infections[padding:],
            obs=data_observed_infections,
            **kwargs,
        )

        all_latent_infections = jnp.hstack(
            [I0, post_initialization_latent_infections]
        )
        npro.deterministic("all_latent_infections", all_latent_infections)

        if observed_infections is not None:
            observed_infections = au.pad_x_to_match_y(
                observed_infections,
                all_latent_infections,
                jnp.nan,
                pad_direction="start",
            )

        Rt = au.pad_x_to_match_y(
            Rt,
            all_latent_infections,
            jnp.nan,
            pad_direction="start",
        )
        npro.deterministic("Rt", Rt)

        return RtInfectionsRenewalSample(
            Rt=Rt,
            latent_infections=all_latent_infections,
            observed_infections=observed_infections,
        )
