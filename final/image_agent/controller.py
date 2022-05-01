from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from image_agent.detections import TeamDetections


class Controller(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def act(self,
            team_id: int,
            team_state: List[Dict[str, Any]],
            team_images: List[np.ndarray],
            team_detections: TeamDetections,
            puck_global_coords: Tuple[float, float, float],
            *args: Any) -> List[Dict[str, Any]]:
        '''
        Given as much information as we can get, outputs the actions for each
        player in the team.
        '''
