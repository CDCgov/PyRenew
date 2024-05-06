# -*- coding: utf-8 -*-

from collections import namedtuple

import jax.numpy as jnp
from numpy.typing import ArrayLike
from pyrenew.deterministic import NullObservation
from pyrenew.metaclass import Model, RandomVariable, _assert_sample_and_rtype
from pyrenew.model.rtinfectionsrenewal import RtInfectionsRenewalModel

HospModelSample = namedtuple(
    "HospModelSample",
    [
        "Rt",
        "latent_infections",
        "IHR",
        "latent_admissions",
        "sampled_admissions",
    ],
    defaults=[None, None, None, None, None],
)
HospModelSample.__doc__ = """
A container for holding the output from HospitalAdmissionsModel.sample().

Attributes
----------
Rt : float or None
    The reproduction number over time. Defaults to None.
latent_infections : ArrayLike or None
    The estimated number of new infections over time. Defaults to None.
IHR : float or None
    The infected hospitalization rate. Defaults to None.
latent_admissions : ArrayLike or None
    The estimated latent hospitalizations. Defaults to None.
sampled_admissions : ArrayLike or None
    The sampled or observed hospital admissions. Defaults to None.

Notes
-----
"""


class HospitalizationsModel(Model):
    """
    HospitalAdmissions Model (BasicRenewal + HospitalAdmissions)

    This class inherits from pyrenew.models.Model. It extends the
    basic renewal model by adding a hospital admissions module, e.g.,
    pyrenew.observations.HospitalAdmissions.
    """

    def __init__(
        self,
        latent_hospitalizations: RandomVariable,
        latent_infections: RandomVariable,
        gen_int: RandomVariable,
        I0: RandomVariable,
        Rt_process: RandomVariable,
        observation_process: RandomVariable | None = None,
        observation_process_infections: RandomVariable | None = None,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        latent_hospitalizations : RandomVariable
            Latent process for the hospitalizations.
        latent_infections : RandomVariable
            The infections latent process (passed to RtInfectionsRenewalModel).
        gen_int : RandomVariable
            Generation time (passed to RtInfectionsRenewalModel)
        I0 : RandomVariable
            Initial infections (passed to RtInfectionsRenewalModel)
        Rt_process : RandomVariable
            Rt process  (passed to RtInfectionsRenewalModel).
        observation_process : RandomVariable, optional
            Observation process for the hospitalizations.
            Defaults to None.
        observation_process_infections : RandomVariable, optional
            Observation process for the infections. Passed to the
            RtInfectionsRenewalModel. Defaults to None.

        Returns
        -------
        None

        Notes
        -----
        TODO: See Also
        """
        self.basic_renewal = RtInfectionsRenewalModel(
            gen_int=gen_int,
            I0=I0,
            latent_infections=latent_infections,
            observation_process=observation_process_infections,
            Rt_process=Rt_process,
        )

        if observation_process is None:
            observation_process = NullObservation()

        HospitalizationsModel.validate(
            latent_hospitalizations, observation_process
        )

        self.latent_hospitalizations = latent_hospitalizations
        self.observation_process = observation_process

    @staticmethod
    def validate(latent_hospitalizations, observation_process) -> None:
        """
        Verifies types and status (RV) of latent and observed hospitalizations

        Parameters
        ----------
        latent_hospitalizations : ArrayLike
            The latent process for the hospitalizations.
        observation_process : ArrayLike
            The observed hospitalizations.

        Returns
        -------
        None

        See Also
        --------
        _assert_sample_and_rtype : Perform type-checking and verify RV
        """
        _assert_sample_and_rtype(latent_hospitalizations, skip_if_none=False)
        _assert_sample_and_rtype(observation_process, skip_if_none=False)
        return None

    def sample_hospitalizations_latent(
        self,
        infections: ArrayLike,
        **kwargs,
    ) -> tuple:
        """Sample number of hospitalizations

        Parameters
        ----------
        infections : ArrayLike
            The predicted infections array.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample() calls, should there be any.

        Returns
        -------
        tuple

        See Also
        --------
        latent_hospitalizations.sample : For sampling latent hospitalizations

        Notes
        -----
        TODO: Include example(s) here.
        TODO: Cover Returns in more detail.
        """

        return self.latent_hospitalizations.sample(
            latent=infections,
            **kwargs,
        )

    def sample_hospitalizations_obs(
        self,
        predicted: ArrayLike,
        observed_hospitalizations: ArrayLike,
        name: str | None = None,
        **kwargs,
    ) -> tuple:
        """Sample number of hospitalizations

        Parameters
        ----------
        predicted : ArrayLike
            The predicted hospitalizations.
        observed_hospitalizations : ArrayLike
            The observed hospitalization data (to fit).
        name : str, optional
            Name of the random variable. Defaults to None.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample_hospitalizations_obs calls, should there be any.

        Returns
        -------
        tuple

        Notes
        -----
        TODO: Include example(s) here.
        """

        return self.observation_process.sample(
            predicted=predicted,
            obs=observed_hospitalizations,
            name=name,
            **kwargs,
        )

    def sample(
        self,
        n_timepoints: int,
        observed_hospitalizations: ArrayLike | None = None,
        observed_infections: ArrayLike | None = None,
        padding: int = 0,
        **kwargs,
    ) -> HospModelSample:
        """
        Sample from the HospitalAdmissions model

        Parameters
        ----------
        n_timepoints : int
            Number of timepoints to sample (passed to the basic renewal model).
        observed_hospitalizations : ArrayLike, optional
            The observed hospitalization data (passed to the basic renewal
            model). Defaults to None (simulation, rather than fit).
        observed_infections : ArrayLike, optional
            The observed infection data (passed to the basic renewal model).
        padding : int, optional
            Number of padding timepoints to add to the beginning of the
            simulation. Defaults to 0.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        HospModelSample

        See Also
        --------
        basic_renewal.sample : For sampling the basic renewal model
        sample_hospitalizations_latent : To sample latent hospitalization process
        sample_hospitalizations_obs : For sampling observed hospitalizations

        Notes
        -----
        TODO: Include example(s) here.
        """

        # Getting the initial quantities from the basic model
        basic_model = self.basic_renewal.sample(
            n_timepoints=n_timepoints,
            observed_infections=observed_infections,
            padding=padding,
            **kwargs,
        )

        # Sampling the latent hospitalizations
        (
            IHR,
            latent,
            *_,
        ) = self.sample_hospitalizations_latent(
            infections=basic_model.latent_infections,
            **kwargs,
        )

        # Sampling the hospitalizations
        if (observed_hospitalizations is not None) and (padding > 0):
            sampled_na = jnp.repeat(jnp.nan, padding)

            sampled_observed, *_ = self.sample_hospitalizations_obs(
                predicted=latent[padding:],
                observed_hospitalizations=observed_hospitalizations[padding:],
                **kwargs,
            )

            sampled = jnp.hstack([sampled_na, sampled_observed])

        else:
            sampled, *_ = self.sample_hospitalizations_obs(
                predicted=latent,
                observed_hospitalizations=observed_hospitalizations,
                **kwargs,
            )

        return HospModelSample(
            Rt=basic_model.Rt,
            latent_infections=basic_model.latent_infections,
            IHR=IHR,
            latent_admissions=latent,
            sampled_admissions=sampled,
        )
