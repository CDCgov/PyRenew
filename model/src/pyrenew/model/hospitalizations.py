# -*- coding: utf-8 -*-

from collections import namedtuple

from numpy.typing import ArrayLike
from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import Model, RandomVariable, _assert_sample_and_rtype
from pyrenew.model.rtinfectionsrenewal import RtInfectionsRenewalModel

HospModelSample = namedtuple(
    "HospModelSample",
    [
        "Rt",
        "infections",
        "IHR",
        "latent",
        "sampled",
    ],
    defaults=[None, None, None, None, None],
)
"""Output from HospitalizationsModel.sample()
"""


class HospitalizationsModel(Model):
    """HospitalAdmissions Model (BasicRenewal + HospitalAdmissions)

    This class inherits from pyrenew.models.Model. It extends the
    basic renewal model by adding a hospitalization module, e.g.,
    pyrenew.observations.HospitalAdmissions.
    """

    def __init__(
        self,
        latent_hospitalizations: RandomVariable,
        latent_infections: RandomVariable,
        gen_int: RandomVariable,
        I0: RandomVariable,
        Rt_process: RandomVariable,
        observed_hospitalizations: RandomVariable,
    ) -> None:
        """Default constructor

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
        observed_hospitalizations : RandomVariable, optional
            Observation process for the hospitalizations.

        Returns
        -------
        None
        """
        self.basic_renewal = RtInfectionsRenewalModel(
            gen_int=gen_int,
            I0=I0,
            latent_infections=latent_infections,
            observed_infections=DeterministicVariable((0,)),
            Rt_process=Rt_process,
        )

        HospitalizationsModel.validate(
            latent_hospitalizations, observed_hospitalizations
        )

        self.latent_hospitalizations = latent_hospitalizations
        self.observed_hospitalizations = observed_hospitalizations

    @staticmethod
    def validate(latent_hospitalizations, observed_hospitalizations) -> None:
        _assert_sample_and_rtype(latent_hospitalizations, skip_if_none=False)
        _assert_sample_and_rtype(observed_hospitalizations, skip_if_none=False)
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
            The predicted infections array
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            `sample_hospitalizations_latent()` calls, if any

        Returns
        -------
        tuple

        See Also
        --------
        latent_hospitalizations.sample : For sampling latent hospitalizations

        Notes
        -----
        TODO: Include example(s) here.
        """

        return self.latent_hospitalizations.sample(
            latent=infections,
            **kwargs,
        )

    def sample_hospitalizations_obs(
        self,
        predicted: ArrayLike,
        observed_hospitalizations: ArrayLike,
        **kwargs,
    ) -> tuple:
        """Sample number of hospitalizations

        Parameters
        ----------
        predicted : ArrayLike
            Predicted hospitalizations.
        observed_hospitalizations : ArrayLike
            The observed hospitalization data
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            `sample_hospitalizations_obs()` calls, if any

        Returns
        -------
        tuple

        See Also
        --------
        observed_hospitalizations.sample : For sampling observed hospitalizations

        Notes
        -----
        TODO: Include example(s) here.
        """

        return self.observed_hospitalizations.sample(
            predicted=predicted, obs=observed_hospitalizations, **kwargs
        )

    def sample(
        self,
        n_timepoints: int,
        observed_hospitalizations=None,
        **kwargs,
    ) -> HospModelSample:
        """
        Sample from the HospitalAdmissions model

        Parameters
        ----------
        n_timepoints : int
            Number of timepoints to sample (passed to the basic renewal model).
        observed_hospitalizations : ArrayLike or None, optional
            The observed hospitalization data (passed to the basic renewal
            model). Defaults to None.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal `sample()`
            calls, if any

        Returns
        -------
        HospModelSample

        See Also
        --------
        basic_renewal.sample : For sampling the basic renewal model
        sample_hospitalizations_latent : To sample latent hospitalizations
        sample_hospitalizations_obs : For sampling observed hospitalizations

        Notes
        -----
        TODO: Include example(s) here.
        """

        # Getting the initial quantities from the basic model
        Rt, infections, *_ = self.basic_renewal.sample(
            n_timepoints=n_timepoints,
            observed_infections=None,
            **kwargs,
        )

        # Sampling the latent hospitalizations
        (
            IHR,
            latent,
            *_,
        ) = self.sample_hospitalizations_latent(
            infections=infections,
            **kwargs,
        )

        # Sampling the hospitalizations
        sampled, *_ = self.sample_hospitalizations_obs(
            predicted=latent,
            observed_hospitalizations=observed_hospitalizations,
            **kwargs,
        )

        return HospModelSample(
            Rt=Rt,
            infections=infections,
            IHR=IHR,
            latent=latent,
            sampled=sampled,
        )
