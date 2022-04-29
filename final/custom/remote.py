from custom.Data import Match, TeamRunner
# from custom.runner_det import Match, TeamRunner
from custom.utils import (BaseRecorder, DataRecorder, MultiRecorder,
                              VideoRecorder, StateRecorder)

# TODO: Wrap TeamRunner and Team in ray if possible
try:
    import ray

    @ray.remote
    class RayMatch(Match):
        pass

    @ray.remote
    class RayTeamRunner(TeamRunner):
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