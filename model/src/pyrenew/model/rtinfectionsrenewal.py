# -*- coding: utf-8 -*-

from collections import namedtuple

from pyrenew.latent import Infections0
from pyrenew.metaclass import Model, RandomVariable, _assert_sample_and_rtype
from pyrenew.process import RtRandomWalkProcess

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
        observed_infections: RandomVariable = None,
        I0: RandomVariable = Infections0(),
        Rt_process: RandomVariable = RtRandomWalkProcess(),
    ) -> None:
        """Default constructor

        Parameters
        ----------
        latent_infections : RandomVariable
            Infections latent process (e.g.,
            pyrenew.latent.Infections.)
        gen_int : RandomVariable
            Generation interval.
        observed_infections : RandomVariable, optional
            Infections observation process (e.g.,
            pyrenew.observations.Poisson.) It should receive the sampled Rt
            via `random_variables`.
        I0 : RandomVariable
            Baseline infections. Default to Infections0.
        Rt_process : RandomVariable, optional
            The sample function of the process should return a tuple where the
            first element is the drawn Rt., by default RtRandomWalkProcess()

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
        random_variables: dict = None,
        constants: dict = None,
    ) -> tuple:
        return self.Rt_process.sample(
            random_variables=random_variables,
            constants=constants,
        )

    def sample_gen_int(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> tuple:
        return self.gen_int.sample(
            random_variables=random_variables,
            constants=constants,
        )

    def sample_i0(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> tuple:
        return self.i0.sample(
            random_variables=random_variables,
            constants=constants,
        )

    def sample_infections_latent(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> tuple:
        return self.latent_infections.sample(
            random_variables=random_variables,
            constants=constants,
        )

    def sample_infections_obs(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> tuple:
        if self.observed_infections is None:
            return (None,)

        return self.observed_infections.sample(
            random_variables=random_variables,
            constants=constants,
        )

    def sample(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> RtInfectionsRenewalSample:
        """Sample from the Basic Renewal Model

        Parameters
        ----------
        random_variables : dict, optional
            A dictionary containing `infections` and/or `Rt` (optional).
        constants : dict, optional
            A dictionary containing `n_timepoints`.

        Returns
        -------
        RtInfectionsRenewalSample
        """

        if random_variables is None:
            random_variables = dict()

        if constants is None:
            constants = dict()

        # Sampling from Rt (possibly with a given Rt, depending on
        # the Rt_process (RandomVariable) object.)
        Rt, *_ = self.sample_rt(
            random_variables=random_variables,
            constants=constants,
        )

        # Getting the generation interval
        gen_int, *_ = self.sample_gen_int(
            random_variables=random_variables,
            constants=constants,
        )

        # Sampling baseline infections
        i0, *_ = self.sample_i0(
            random_variables=random_variables,
            constants=constants,
        )

        # Sampling from the latent process
        latent, *_ = self.sample_infections_latent(
            random_variables={
                **random_variables,
                **dict(Rt=Rt, gen_int=gen_int, i0=i0),
            },
            constants=constants,
        )

        # Using the predicted infections to sample from the observation process
        observed, *_ = self.sample_infections_obs(
            random_variables={
                **random_variables,
                **dict(latent=latent),
            },
            constants=constants,
        )

        return RtInfectionsRenewalSample(
            Rt=Rt,
            latent=latent,
            observed=observed,
        )
