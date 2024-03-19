#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import numpyro
import numpyro.distributions as dist
from pyrenew.metaclasses import RandomProcess


class PoissonObservation(RandomProcess):
    """
    Poisson observation process
    """

    def __init__(
        self,
        parameter_name: str = "poisson_rv",
        rate_varname: str = "rate",
        counts_varname: str = "counts",
        eps: float = 1e-8,
    ) -> None:
        """Default Constructor

        Parameters
        ----------
        parameter_name : str, optional
            Passed to numpyro.sample.
        rate_varname : str, optional
            Name of the element in `random_variables` that will hold the rate
            when calling `PoissonObservation.sample()`.
        counts_varname : str, optional
            Name of the element in `random_variables` that will hold the
            observed count when calling `PoissonObservation.sample()`.

        Returns
        -------
        None
        """

        self.parameter_name = parameter_name
        self.rate_varname = rate_varname
        self.counts_varname = counts_varname
        self.eps = eps

        return None

    def sample(
        self,
        random_variables: dict,
        constants: dict = None,
    ) -> tuple:
        """Sample from the Poisson process

        Parameters
        ----------
        random_variables : dict, optional
            A dictionary containing the rate parameter passed to
            `numpyro.distributions.Poisson()`, and possible containing `counts`
            passed to `obs` in `numpyro.sample()`.
        constants : dict, optional
            Ignored.

        Returns
        -------
        tuple
        """
        return (
            numpyro.sample(
                self.parameter_name,
                dist.Poisson(
                    rate=random_variables.get(self.rate_varname) + self.eps
                ),
                obs=random_variables.get(self.counts_varname, None),
            ),
        )

    @staticmethod
    def validate():
        None
