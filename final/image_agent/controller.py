import enum
from abc import ABCMeta, abstractmethod
from enum import auto
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

        :return List[Dict[str, Any]] list of the actions for all players. Each
            dict corresponds to the action of 1 player.
        '''

    @abstractmethod
    def get_kart_types(self, team: int, num_players: int) -> List[str]:
        '''
        Get the kart types for each player in the given team.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie',
                 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok',
                 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne',
                 'tux', 'wilber', 'xue'. Default: 'tux'
        '''


class ControllerType(enum.Enum):
    JURGEN = auto()
    ANA = auto()
