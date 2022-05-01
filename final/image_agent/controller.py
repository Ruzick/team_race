from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

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
            team_puck_global_coords: List[Optional[np.ndarray]],
            *args: Any) -> List[Dict[str, Any]]:
        '''
        Given as much information as we can get, outputs the actions for each
        player in the team.

        :param team_id: int id for the team. RED is 0, BLUE is 1
        :param team_state: List[Dict[str, Any]] list of the state of each player.
            Info about player state can be found in player.py
        :param team_detections: TeamDetections holds information about
            all objects detected in each player's image
        :param team_puck_global_coords: List[Optional[np.ndarray]] list of where
            each player thinks the puck is, based on what they saw in the image.
            None means that the player thinks they did not see the puck.
        :param *args Any anything else you might need for choosing your action
            you can pass in here
        '''
