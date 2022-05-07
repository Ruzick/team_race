from os import path
from typing import Any, Dict, List, Optional, Tuple

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
        self.prev_team_positions: Tuple[List[Tensor], List[Tensor]] = ([], [])
        self.prev_puck_position: Tensor = torch.tensor([0., 0.], dtype=torch.float32)

    def act(self,
            team_id: int,
            team_state: List[Dict[str, Any]],
            team_images: List[np.ndarray],
            team_detections: TeamDetections,
            team_puck_global_coords: List[Optional[np.ndarray]],
            *args: Any) -> List[Dict[str, Any]]:

        if team_puck_global_coords[0] is None or team_puck_global_coords[1] is None:
            raise RuntimeError('These should not be None')

        puck_global_coords = torch.from_numpy(team_puck_global_coords[0])[[0, 2]]
        ball_search_actions = get_ball_search_actions(team_state,
                                                      puck_global_coords,
                                                      self.prev_puck_position)
        self.prev_puck_position = puck_global_coords
        if ball_search_actions is not None:
            return ball_search_actions

        actions: List[Dict[str, Any]] = []
        for i_player, player_state in enumerate(team_state):
            soccer_state = get_soccer_state(team_puck_global_coords, i_player)
            features_tensor = extract_featuresV2(player_state, soccer_state, team_id)

            action = (get_stuck_near_ball_action(self.prev_team_positions[i_player],
                                                 features_tensor)
                      or get_action_from_model(self.model, features_tensor, self.device))
            actions.append(action)

            self.prev_team_positions[i_player].append(features_tensor[0:2])
            if len(self.prev_team_positions[i_player]) >= 20:
                del self.prev_team_positions[i_player][:-20]

        return actions

    def get_kart_types(self, team: int, num_players: int) -> List[str]:
        return ['sara_the_racer'] * num_players


def get_action_from_model(model: ScriptModule, features_tensor: Tensor, device: torch.device
                          ) -> Dict[str, Any]:
    accel, steer, brake = model(features_tensor.to(device))
    return {
        'acceleration': accel,
        'steer': steer,
        'brake': brake,
    }


def get_stuck_near_ball_action(prev_player_positions: List[Tensor],
                               features_tensor: Tensor
                               ) -> Optional[Dict[str, Any]]:
    if len(prev_player_positions) < 10:
        return None

    prev_positions_tensor = torch.stack(prev_player_positions)
    prev_positions_start_dist: Tensor = torch.norm((prev_positions_tensor
                                                    - prev_positions_tensor[0:1, :]),
                                                   dim=-1)
    if prev_positions_start_dist.max() > 5:
        return None

    ball_position = features_tensor[[7, 8]]
    prev_positions_ball_dist: Tensor = torch.norm(prev_positions_tensor - ball_position[None, :],
                                                  dim=-1)
    if prev_positions_ball_dist.max() > 5:
        return None

    kart_to_puck_angle_difference = features_tensor[6]
    if abs(kart_to_puck_angle_difference) < 0.05:
        # If almost point straight, go straight ahead
        print('Surprise!')
        return {
            'acceleration': 1.,
            'steer': torch.sign(kart_to_puck_angle_difference),
            'brake': False,
        }

    return {
        'acceleration': 0.,
        'steer': -1. * torch.sign(kart_to_puck_angle_difference),
        'brake': True,
    }


def get_ball_search_actions(team_state: List[Dict[str, Any]],
                            puck_global_coords: Tensor,
                            prev_puck_coords: Tensor,
                            ) -> Optional[List[Dict[str, Any]]]:
    if puck_global_coords[0] == 0. and puck_global_coords[1] == 0.:
        return None

    if puck_global_coords != prev_puck_coords:
        return None

    actions: List[Dict[str, Any]] = []
    for player_state in team_state:
        action = (get_in_goals_action(player_state)
                  or get_look_around_action(player_state))
        actions.append(action)

    return actions


def get_in_goals_action(player_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    player_location = player_state['kart']['location']
    if -64.5 <= player_location[2] and player_location[2] <= 64.5:
        return None

    return None

    # return {
    #     'acceleration': 0.,
    #     'steer': 0.,
    #     'rescue': True,
    # }


def get_look_around_action(player_state: Dict[str, Any]) -> Dict[str, Any]:
    player_front = torch.tensor(player_state['kart']['front'], dtype=torch.float32)[[0, 2]]
    player_center = torch.tensor(player_state['kart']['location'], dtype=torch.float32)[[0, 2]]
    player_direction = player_front - player_center
    player_unit_direction = player_direction / torch.norm(player_direction)
    player_dist_from_center = torch.norm(player_center)

    if player_dist_from_center > 20 and player_unit_direction.dot(player_center) < 0:
        # Away from center and pointing inwards means accelerate and steer
        return {
            'acceleration': .5,
            'steer': 1.,
            'brake': False,
            'drift': True
        }

    # In the center or pointing outwards means reverse and steer
    return {
        'acceleration': 0.,
        'steer': -1.,
        'brake': True,
    }


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
