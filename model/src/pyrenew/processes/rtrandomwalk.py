import numpyro as npro
import numpyro.distributions as dist
from pyrenew.metaclasses import RandomProcess
from pyrenew.processes.simplerandomwalk import SimpleRandomWalkProcess
from pyrenew.transform import AbstractTransform, LogTransform


class RtRandomWalkProcess(RandomProcess):
    def __init__(
        self,
        Rt0_dist: dist.Distribution = dist.TruncatedNormal(
            loc=1.2, scale=0.2, low=0
        ),
        Rt_transform: AbstractTransform = LogTransform(),
        Rt_rw_dist: dist.Distribution = dist.Normal(0, 0.025),
    ) -> None:
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

    def sample(self, data):
        n_timepoints = data.get("n_timepoints")

        Rt0 = npro.sample("Rt0", self.Rt0_dist, obs=data.get("Rt0", None))

        Rt0_trans = self.Rt_transform(Rt0)
        Rt_trans_proc = SimpleRandomWalkProcess(self.Rt_rw_dist)
        Rt_trans_ts = Rt_trans_proc.sample(
            duration=n_timepoints, name="Rt_transformed_rw", init=Rt0_trans
        )

        Rt = npro.deterministic("Rt", self.Rt_transform.inverse(Rt_trans_ts))

        return Rt
