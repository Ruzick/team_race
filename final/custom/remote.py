from . import Data
from tournament.runner import Match, TeamRunner
from tournament.utils import (BaseRecorder, DataRecorder, MultiRecorder,
                              VideoRecorder, StateRecorder)

# TODO: Wrap TeamRunner and Team in ray if possible
try:
    import ray

    @ray.remote
    class RayMatch(Data.Match):
        pass

    @ray.remote
    class RayTeamRunner(Data.TeamRunner):
        pass

    @ray.remote
    class RayStateRecorder(StateRecorder):
        pass

    @ray.remote
    class RayDataRecorder(DataRecorder):
        pass

    @ray.remote
    class RayVideoRecorder(VideoRecorder):
        pass

    RayMatchException = ray.exceptions.RayTaskError

    get = ray.get
    init = ray.init

except ImportError:
    ray = None