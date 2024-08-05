# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable, SampledValue
from pyrenew.process.ar import ARProcess
from pyrenew.process.differencedprocess import FirstDifferencedProcess


class FirstDifferencedARProcess(FirstDifferencedProcess):
    """
    Class for a stochastic process
    with an AR(n) process on the first
    differences (i.e. the rate of change).
    """

    def __init__(
            self,
            name: str,
            autoreg: ArrayLike,
            noise_sd: float,
            ar_process_suffix = "_diff_ar"
            **kwargs,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        name : str
            Name for the random variable
        autoreg : ArrayLike
            Array of AR process autoregressive coefficient(s),
            passed to `class:`pyrenew.process.ARprocess`.
            Length of the vector will determine the order
            of the process.
        noise_sd : float
            s.d. of the AR process zero-mean Normal step,
            distribution, passed to `class`:pyrenew.process.ARProcess`
        ar_process_suffix: str
            Suffix to add to self.name when naming the underlying
            ARProcess RandomVariable. Default "_diff_ar".
        **kwargs :
            Additional keyword arguments passed to the
            parent class constructor

        Returns
        -------
        None
        """
        super().__init__(
            name=name,
            fundamental_process=ARProcess(
                self.name + ar_process_suffix,
                jnp.atleast_1d(autoreg),
                jnp.atleast_1d(noise_sd)

        self.rate_of_change_proc = ARProcess(
            "arprocess", 0, jnp.array([autoreg]), noise_sd
        )
        self.name = name
