# -*- coding: utf-8 -*-

from collections import namedtuple

from numpy.typing import ArrayLike
from pyrenew.metaclass import Model, RandomVariable, _assert_sample_and_rtype

# Output class of the RtInfectionsRenewalModel
RtInfectionsRenewalSample = namedtuple(
    "InfectModelSample",
    ["Rt", "latent", "observed"],
    defaults=[None, None, None],
)
"""Output from RtInfectionsRenewalModel.sample()"""


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
            pyrenew.latent.Infections.)
        gen_int : RandomVariable
            Generation interval.
        I0 : RandomVariable
            Initial infections.
        Rt_process : RandomVariable
            The sample function of the process should return a tuple where the
            first element is the drawn Rt.
        observed_infections : RandomVariable, optional
            Infections observation process (e.g.,
            pyrenew.observations.Poisson.) It should receive the sampled Rt
            via `random_variables`.

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
        return self.Rt_process.sample(**kwargs)

    def sample_gen_int(
        self,
        **kwargs,
    ) -> tuple:
        return self.gen_int.sample(**kwargs)

    def sample_i0(
        self,
        **kwargs,
    ) -> tuple:
        return self.i0.sample(**kwargs)

    def sample_infections_latent(
        self,
        **kwargs,
    ) -> tuple:
        return self.latent_infections.sample(**kwargs)

    def sample_infections_obs(
        self,
        latent: ArrayLike,
        observed_infections: ArrayLike = None,
        **kwargs,
    ) -> tuple:
        return self.observed_infections.sample(
            mean=latent,
            obs=observed_infections,
            **kwargs,
        )

    def sample(
        self,
        n_timepoints: int,
        observed_infections: ArrayLike = None,
        **kwargs,
    ) -> RtInfectionsRenewalSample:
        """Sample from the Basic Renewal Model

        Parameters
        ----------
        n_timepoints : int
            Number of timepoints to sample.
        observed_infections : ArrayLike, optional
            Observed infections.
        kwargs : dict
            Keyword arguments passed to the sampling methods.

        Returns
        -------
        RtInfectionsRenewalSample
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
        observed, *_ = self.sample_infections_obs(
            latent=latent, observed_infections=observed_infections, **kwargs
        )

        return RtInfectionsRenewalSample(
            Rt=Rt,
            latent=latent,
            observed=observed,
        )
