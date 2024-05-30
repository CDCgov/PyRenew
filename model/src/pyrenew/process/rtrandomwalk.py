# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import numpyro as npro
import numpyro.distributions as dist
from pyrenew.metaclass import RandomVariable
from pyrenew.process.simplerandomwalk import SimpleRandomWalkProcess
from pyrenew.transform import AbstractTransform, LogTransform


class RtRandomWalkProcess(RandomVariable):
    r"""Rt Randomwalk Process

    Notes
    -----

    The process is defined as follows:

    .. math::

            Rt(0) &\sim \text{Rt0_dist} \\
            Rt(t) &\sim \text{Rt_transform}(\text{Rt_transformed_rw}(t))
    """

    def __init__(
        self,
        Rt0_dist: dist.Distribution = dist.TruncatedNormal(
            loc=1.2, scale=0.2, low=0
        ),
        Rt_transform: AbstractTransform = LogTransform(),
        Rt_rw_dist: dist.Distribution = dist.Normal(0, 0.025),
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        Rt0_dist : dist.Distribution, optional
            Initial distribution of Rt, defaults to
            dist.TruncatedNormal( loc=1.2, scale=0.2, low=0 )
        Rt_transform : AbstractTransform, optional
            Transformation applied to the sampled Rt0, defaults
            to LogTransform().
        Rt_rw_dist : dist.Distribution, optional
            Randomwalk process, defaults to dist.Normal(0, 0.025)

        Returns
        -------
        None
        """
        RtRandomWalkProcess.validate(Rt0_dist, Rt_transform, Rt_rw_dist)

        self.Rt0_dist = Rt0_dist
        self.Rt_transform = Rt_transform
        self.Rt_rw_dist = Rt_rw_dist

        return None

    @staticmethod
    def validate(
        Rt0_dist: dist.Distribution,
        Rt_transform: AbstractTransform,
        Rt_rw_dist: dist.Distribution,
    ) -> None:
        """
        Validates Rt0_dist, Rt_transform, and Rt_rw_dist.

        Parameters
        ----------
        Rt0_dist : dist.Distribution, optional
            Initial distribution of Rt, expected dist.Distribution
        Rt_transform : any
            Transformation applied to the sampled Rt0, expected
            AbstractTransform
        Rt_rw_dist : any
            Randomwalk process, expected dist.Distribution.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If Rt0_dist or Rt_rw_dist are not dist.Distribution or if
            Rt_transform is not AbstractTransform.
        """
        assert isinstance(Rt0_dist, dist.Distribution)
        assert isinstance(Rt_transform, AbstractTransform)
        assert isinstance(Rt_rw_dist, dist.Distribution)

    def sample(
        self,
        n_timepoints: int,
        **kwargs,
    ) -> tuple:
        """
        Generate samples from the process

        Parameters
        ----------
        n_timepoints : int
            Number of timepoints to sample.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        tuple
            With a single array of shape (duration,).
        """

        Rt0 = npro.sample("Rt0", self.Rt0_dist)

        Rt0_trans = self.Rt_transform(Rt0)
        Rt_trans_proc = SimpleRandomWalkProcess(self.Rt_rw_dist)
        Rt_trans_ts, *_ = Rt_trans_proc.sample(
            duration=n_timepoints,
            name="Rt_transformed_rw",
            init=Rt0_trans,
        )

        Rt = npro.deterministic("Rt", self.Rt_transform.inverse(Rt_trans_ts))

        return (Rt,)
