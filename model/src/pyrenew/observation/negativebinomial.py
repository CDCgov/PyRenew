# -*- coding: utf-8 -*-

import numbers as nums

import numpyro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


class NegativeBinomialObservation(RandomVariable):
    """Negative Binomial observation"""

    def __init__(
        self,
        concentration_prior: dist.Distribution | ArrayLike,
        concentration_suffix: str = "_concentration",
        parameter_name="negbinom_rv",
        mean_varname="mean",
        counts_varname="counts",
    ) -> None:
        """Default constructor

        Parameters
        ----------
        concentration_prior : dist.Distribution | nummbers.nums
            Numpyro distribution from which to sample the positive concentration
            parameter of the negative binomial. This parameter is sometimes
            called k, phi, or the "dispersion" or "overdispersion" parameter,
            despite the fact that larger values imply that the distribution
            becomes more Poissonian, while smaller ones imply a greater degree
            of dispersion.
        concentration_suffix : str, optional
            Suffix for the numpy variable.
        parameter_name : str, optional
            Name for the numpy variable.
        mean_varname : str, optional
            Name of the element in `random_variables` that will hold the rate
            when calling `PoissonObservation.sample()`.
        counts_varname: str, optional
            Name of the element in `random_variables` that will hold the
            observed count when calling `PoissonObservation.sample()`.

        Returns
        -------
        None
        """

        NegativeBinomialObservation.validate(concentration_prior)

        if isinstance(concentration_prior, dist.Distribution):
            self.sample_prior = lambda: numpyro.sample(
                self.parameter_name + self.concentration_suffix,
                concentration_prior,
            )
        else:
            self.sample_prior = lambda: concentration_prior

        self.parameter_name = parameter_name
        self.concentration_suffix = concentration_suffix
        self.mean_varname = mean_varname
        self.counts_varname = counts_varname

    def sample(
        self,
        random_variables: dict,
        constants: dict = None,
    ) -> tuple:
        """Sample from the negative binomial distribution

        Parameters
        ----------
        random_variables : dict, optional
            A dictionary containing the `mean` parameter, and possibly
            containing `counts`, which is passed to `obs` `numpyro.sample()`.
        constants : dict, optional
            Ignored, defaults to dict().

        Returns
        -------
        tuple
        """
        return (
            numpyro.sample(
                self.parameter_name,
                dist.NegativeBinomial2(
                    mean=random_variables.get(self.mean_varname),
                    concentration=self.sample_prior(),
                ),
                obs=random_variables.get(self.counts_varname, None),
            ),
        )

    @staticmethod
    def validate(concentration_prior) -> None:
        assert isinstance(
            concentration_prior, (dist.Distribution, nums.Number)
        )
        return None
