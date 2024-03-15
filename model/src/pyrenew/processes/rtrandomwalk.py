import numpyro as npro
import numpyro.distributions as dist
from pyrenew.metaclasses import RandomProcess
from pyrenew.processes.simplerandomwalk import SimpleRandomWalkProcess
from pyrenew.transform import AbstractTransform, LogTransform


class RtRandomWalkProcess(RandomProcess):
    """Rt Randomwalk Process"""

    def __init__(
        self,
        Rt0_dist: dist.Distribution = dist.TruncatedNormal(
            loc=1.2, scale=0.2, low=0
        ),
        Rt_transform: AbstractTransform = LogTransform(),
        Rt_rw_dist: dist.Distribution = dist.Normal(0, 0.025),
    ) -> None:
        """Default constructor

        :param Rt0_dist: Baseline distributiono of Rt, defaults to
            dist.TruncatedNormal( loc=1.2, scale=0.2, low=0 )
        :type Rt0_dist: dist.Distribution, optional
        :param Rt_transform: Transformation applied to the sampled Rt0, defaults
            to LogTransform()
        :type Rt_transform: AbstractTransform, optional
        :param Rt_rw_dist: Randomwalk process, defaults to dist.Normal(0, 0.025)
        :type Rt_rw_dist: dist.Distribution, optional
        :return: _description_
        :rtype: _type_
        """
        self.validate(Rt0_dist, Rt_transform, Rt_rw_dist)

        self.Rt0_dist = Rt0_dist
        self.Rt_transform = Rt_transform
        self.Rt_rw_dist = Rt_rw_dist

        return None

    @staticmethod
    def validate(Rt0_dist, Rt_transform, Rt_rw_dist):
        assert isinstance(Rt0_dist, dist.Distribution)
        assert isinstance(Rt_transform, AbstractTransform)
        assert isinstance(Rt_rw_dist, dist.Distribution)

    def sample(self, obs: dict = dict(), data: dict = dict()):
        """Generate samples from the process

        :param data: A dictionary containing
        :param data: A dictionary containing `n_timepoints` and optionally
            `Rt0`.
        :type data: _type_
        :return: _description_
        :rtype: _type_
        """
        n_timepoints = data.get("n_timepoints")

        Rt0 = npro.sample("Rt0", self.Rt0_dist, obs=obs.get("Rt0", None))

        Rt0_trans = self.Rt_transform(Rt0)
        Rt_trans_proc = SimpleRandomWalkProcess(self.Rt_rw_dist)
        Rt_trans_ts = Rt_trans_proc.sample(
            duration=n_timepoints, name="Rt_transformed_rw", init=Rt0_trans
        )

        Rt = npro.deterministic("Rt", self.Rt_transform.inverse(Rt_trans_ts))

        return Rt
