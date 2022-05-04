from os import path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.jit import ScriptModule

from image_agent.controller import Controller
from image_agent.detections import TeamDetections

GOAL_TEAM1 = [
    [-10.449999809265137, 0.07000000029802322, -64.5],
    [10.449999809265137, 0.07000000029802322, -64.5]
]
GOAL_TEAM2 = [
    [10.460000038146973, 0.07000000029802322, 64.5],
    [-10.510000228881836, 0.07000000029802322, 64.5]
]
GOALS = [GOAL_TEAM1, GOAL_TEAM2]


class JurgenController(Controller):
    def __init__(self, device: torch.device):
        super().__init__()
        model: ScriptModule = torch.jit.load(
            path.join(path.dirname(path.abspath(__file__)), 'jurgen_agent.pt'))
        self.model = model.to(device)
        self.device = device

    def act(self,
            team_id: int,
            team_state: List[Dict[str, Any]],
            team_images: List[np.ndarray],
            team_detections: TeamDetections,
            team_puck_global_coords: List[Optional[np.ndarray]],
            *args: Any) -> List[Dict[str, Any]]:

        actions: List[Dict[str, Any]] = []
        for i_player, player_state in enumerate(team_state):
            soccer_state = get_soccer_state(team_puck_global_coords, i_player)
            model_input = extract_featuresV2(player_state, soccer_state, team_id)
            accel, steer, brake = self.model(model_input.to(self.device))
            actions.append({
                'acceleration': accel,
                'steer': steer,
                'brake': brake,
            })

        return actions

    def get_kart_types(self, team: int, num_players: int) -> List[str]:
        return ['sara_the_racer'] * num_players


def get_soccer_state(team_puck_global_coords: List[Optional[np.ndarray]], i_player: int):
    if team_puck_global_coords[i_player] is not None:
        location: List[float] = team_puck_global_coords[i_player].tolist()
    else:
        location = [0., 0., 0.]
    ball_state = {
        'location': location
    }
    goals_state = GOALS
    return {
        'ball': ball_state,
        'goal_line': goals_state
    }


def limit_period(angle: Tensor) -> Tensor:
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2


def extract_featuresV2(player_state: dict, soccer_state: dict, team_id: int) -> Tensor:
    # features of ego-vehicle
    kart_front = torch.tensor(player_state['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(player_state['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    # features of soccer
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0])

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

    # features of score-line
    goal_line_center = torch.tensor(
        soccer_state['goal_line'][(team_id+1) % 2], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

    puck_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)

    features = torch.tensor([
        kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle,
        goal_line_center[0], goal_line_center[1], kart_to_puck_angle_difference,
        puck_center[0], puck_center[1], puck_to_goal_line[0], puck_to_goal_line[1]
    ], dtype=torch.float32)

    return features
