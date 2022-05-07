from os import path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.jit import ScriptModule

from image_agent.controller import Controller
from image_agent.detections import TeamDetections

DISABLE_BALL_SEARCH_ACTION = False
DISABLE_STUCK_NEAR_BALL_ACTION = False
DISABLE_AFTER_GOAL_ACTION = False
DISABLE_AVOID_PLAYER_INTERFERENCE_ACTION = False
DISABLE_ADD_MOMENTUM = False

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
        self.prev_frame_search_actions: Optional[List[Dict[str, Any]]] = None

    def act(self,
            team_id: int,
            team_state: List[Dict[str, Any]],
            team_images: List[np.ndarray],
            team_detections: TeamDetections,
            team_puck_global_coords: List[Optional[np.ndarray]],
            *args: Any) -> List[Dict[str, Any]]:
        # print('Frame:', args[0])

        if team_puck_global_coords[0] is None or team_puck_global_coords[1] is None:
            raise RuntimeError('These should not be None')

        puck_global_coords = torch.from_numpy(team_puck_global_coords[0])[[0, 2]]
        ball_search_actions = get_ball_search_actions(team_state,
                                                      puck_global_coords,
                                                      self.prev_puck_position,
                                                      self.prev_frame_search_actions)
        self.prev_puck_position = puck_global_coords
        if ball_search_actions is not None:
            self.prev_frame_search_actions = ball_search_actions
            return ball_search_actions

        self.prev_frame_search_actions = None

        actions: List[Dict[str, Any]] = []
        for i_player, player_state in enumerate(team_state):
            soccer_state = get_soccer_state(team_puck_global_coords,
                                            i_player,
                                            player_state,
                                            self.prev_puck_position)
            features_tensor = extract_featuresV2(player_state, soccer_state, team_id)

            action = (get_stuck_near_ball_action(self.prev_team_positions[i_player],
                                                 features_tensor)
                      or get_after_goal_action(team_state, i_player, features_tensor)
                      or get_avoid_interference_action(team_state, i_player, features_tensor)
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
    if DISABLE_STUCK_NEAR_BALL_ACTION:
        return None

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
        # If puck is almost straight ahead, let Jurgen take over
        # print('Surprise!')
        return None
        # return {
        #     'acceleration': 1.,
        #     'steer': torch.sign(kart_to_puck_angle_difference),
        #     'brake': False,
        # }

    return {
        'acceleration': 0.,
        'steer': -1. * torch.sign(kart_to_puck_angle_difference),
        'brake': True,
    }


def get_after_goal_action(_: List[Dict[str, Any]],
                          i_player: int,
                          features_tensor: Tensor
                          ) -> Optional[Dict[str, Any]]:
    if DISABLE_AFTER_GOAL_ACTION:
        return None

    if i_player == 1:
        return None

    # other_player_pos = torch.tensor(team_state[1]['kart']['center'], dtype=torch.float32)[[0, 2]]
    # if abs(other_player_pos[1]) < 10:
    #     return None

    puck_center = features_tensor[7:9]
    if puck_center[0] != 0. or puck_center[1] != 0.:
        return None

    return {
        'acceleration': 0.,
        'steer': 0.,
        'brake': False
    }


def get_avoid_interference_action(team_state: List[Dict[str, Any]],
                                  i_player: int,
                                  features_tensor: Tensor
                                  ) -> Optional[Dict[str, Any]]:
    if DISABLE_AVOID_PLAYER_INTERFERENCE_ACTION:
        return None

    player_pos = torch.tensor(team_state[i_player]['kart']['location'],
                              dtype=torch.float32)[[0, 2]]
    i_other = (i_player + 1) % 2
    other_pos = torch.tensor(team_state[i_other]['kart']['location'], dtype=torch.float32)[[0, 2]]
    if torch.norm(other_pos - player_pos) > 4:
        return None

    puck_center = features_tensor[7:9]
    player_to_puck_direction = player_pos - puck_center
    other_to_puck_direction = other_pos - puck_center
    player_to_puck_dist = torch.norm(player_to_puck_direction)
    other_to_puck_dist = torch.norm(other_to_puck_direction)
    if player_to_puck_dist < other_to_puck_dist:
        return None

    if other_to_puck_dist > 3:
        return None

    # other_to_puck_angle = torch.atan2(other_to_puck_direction[1], other_to_puck_direction[0])
    # other_front = torch.tensor(team_state[i_other]['kart']['front'], dtype=torch.float32)[[0, 2]]
    # other_direction = other_front - other_pos
    # other_angle = torch.atan2(other_direction[1], other_direction[0])

    # if abs(other_angle - other_to_puck_angle) > torch.pi / 6:
    #     return None

    return {
        'acceleration': 0.,
        'steer': 0.,
        'brake': True
    }


def get_ball_search_actions(team_state: List[Dict[str, Any]],
                            puck_global_coords: Tensor,
                            prev_puck_coords: Tensor,
                            prev_frame_search_actions: Optional[List[Dict[str, Any]]]
                            ) -> Optional[List[Dict[str, Any]]]:
    if DISABLE_BALL_SEARCH_ACTION:
        return None

    if puck_global_coords[0] == 0. and puck_global_coords[1] == 0.:
        return None

    if not torch.equal(puck_global_coords, prev_puck_coords):
        return None

    if prev_frame_search_actions is not None:
        return prev_frame_search_actions

    actions: List[Dict[str, Any]] = []
    for player_state in team_state:
        action = (get_in_goals_action(player_state)
                  or get_look_around_action(player_state, prev_puck_coords))
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


def get_look_around_action(player_state: Dict[str, Any], prev_puck_coords: Tensor
                           ) -> Dict[str, Any]:
    player_front = torch.tensor(player_state['kart']['front'], dtype=torch.float32)[[0, 2]]
    player_center = torch.tensor(player_state['kart']['location'], dtype=torch.float32)[[0, 2]]
    player_direction = player_front - player_center
    # player_unit_direction = player_direction / torch.norm(player_direction)
    player_angle = torch.atan2(player_direction[1], player_direction[0])

    # features of soccer
    puck_center = prev_puck_coords
    player_to_puck_direction = puck_center - player_center
    player_to_puck_angle = torch.atan2(player_to_puck_direction[1], player_to_puck_direction[0])

    player_to_puck_angle_difference = limit_period((player_angle - player_to_puck_angle) / np.pi)

    # In the center or pointing outwards means reverse and steer
    return {
        'acceleration': 0.,
        'steer': -1. * torch.sign(player_to_puck_angle_difference),
        'brake': True,
    }

    # player_front = torch.tensor(player_state['kart']['front'], dtype=torch.float32)[[0, 2]]
    # player_center = torch.tensor(player_state['kart']['location'], dtype=torch.float32)[[0, 2]]
    # player_direction = player_front - player_center
    # player_unit_direction = player_direction / torch.norm(player_direction)
    # player_dist_from_center = torch.norm(player_center)

    # if player_dist_from_center > 20 and player_unit_direction.dot(player_center) < 0:
    #     # Away from center and pointing inwards means accelerate and steer
    #     return {
    #         'acceleration': .5,
    #         'steer': 1.,
    #         'brake': False,
    #         'drift': True
    #     }

    # # In the center or pointing outwards means reverse and steer
    # return {
    #     'acceleration': 0.,
    #     'steer': -1.,
    #     'brake': True,
    # }


def get_soccer_state(team_puck_global_coords: List[Optional[np.ndarray]],
                     i_player: int,
                     player_state: Dict[str, Any],
                     prev_puck_position: Tensor):
    location: List[float] = team_puck_global_coords[i_player].tolist()
    x_velocity = location[0] - float(prev_puck_position[0])
    y_velocity = location[2] - float(prev_puck_position[1])
    if not DISABLE_ADD_MOMENTUM and not (x_velocity == 0. and y_velocity == 0.):
        x_velocity = location[0] - float(prev_puck_position[0])
        y_velocity = location[2] - float(prev_puck_position[1])

        player_front = torch.tensor(player_state['kart']['front'], dtype=torch.float32)[[0, 2]]
        player_center = torch.tensor(player_state['kart']['location'], dtype=torch.float32)[[0, 2]]
        player_direction = player_front - player_center

        puck_center = torch.tensor([location[0], location[2]], dtype=torch.float32)
        velocity = puck_center - prev_puck_position
        # if torch.norm(velocity) >= 3:
        #     print('puck may have teleported!', velocity)
        velocity_normal = torch.tensor([-velocity[1], velocity[0]], dtype=torch.float32)

        unit_velocity_normal = velocity_normal / torch.norm(velocity_normal)
        unit_player_direction = player_direction / torch.norm(player_direction)
        velocity_multiplier = abs(float(unit_velocity_normal.dot(unit_player_direction)))
        velocity_multiplier *= 3
        velocity_multiplier /= max(float(torch.norm(velocity)), 1.)

        # player_front = torch.tensor(player_state['kart']['front'], dtype=torch.float32)[[0, 2]]
        # player_center = torch.tensor(player_state['kart']['location'], dtype=torch.float32)[[0, 2]]
        # player_direction = player_front - player_center
        # # player_unit_direction = player_direction / torch.norm(player_direction)
        # player_angle = torch.atan2(player_direction[1], player_direction[0])

        # # features of soccer
        # puck_center = torch.tensor([location[0], location[2]], dtype=torch.float32)
        # player_to_puck_direction = puck_center - player_center
        # player_to_puck_angle = torch.atan2(player_to_puck_direction[1],
        #                                    player_to_puck_direction[0])

        # player_to_puck_angle_difference = limit_period((player_angle - player_to_puck_angle)
        #                                                / torch.pi)
        # velocity_multiplier = 2 * (0.5 - abs(0.5 - abs(player_to_puck_angle_difference)))

        location[0] += x_velocity * velocity_multiplier
        location[2] += y_velocity * velocity_multiplier

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
