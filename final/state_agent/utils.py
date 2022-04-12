from os import path
from typing import List
import numpy as np

from torch import Tensor
from torch.jit import ScriptModule
import torch

from state_agent.model import StateModel


MODEL_FILENAME = 'state_agent.pt'


def limit_period(angle):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2


def extract_features(player_id: int,
                     player_state: List[dict],
                     soccer_state: dict,
                     opponent_state: List[dict],
                     team_id: int
                     ) -> Tensor:
    pstate = player_state[player_id]

    # features of ego-vehicle
    kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front - kart_center) / torch.norm(kart_front - kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    # features of soccer ball
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center - kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0])

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle) / np.pi)

    # features of goal-line
    goal_line_center = torch.tensor(soccer_state['goal_line'][(team_id + 1) % 2],
                                    dtype=torch.float32)[:, [0, 2]].mean(dim=0)

    puck_to_goal_line_diff = goal_line_center - puck_center
    puck_to_goal_line_angle = torch.atan2(puck_to_goal_line_diff[1], puck_to_goal_line_diff[0])
    puck_to_goal_line_dist = torch.norm(puck_to_goal_line_diff)

    features = torch.tensor(
        [
            kart_center[0], kart_center[1],
            kart_angle, kart_to_puck_angle,
            goal_line_center[0], goal_line_center[1],  # Goal location (TODO: remove)
            kart_to_puck_angle_difference,
            puck_center[0], puck_center[1],  # Ball location
            puck_to_goal_line_angle, puck_to_goal_line_dist
        ],
        dtype=torch.float32)

    return features


def load_model(filename: str = MODEL_FILENAME) -> ScriptModule:
    return torch.jit.load(path.join(path.dirname(path.abspath(__file__)), filename))


def save_model(model: StateModel, filename: str = MODEL_FILENAME):
    torch.jit.save(model, path.join(path.dirname(path.abspath(__file__)), filename))
