from os import path
from typing import List

import torch
from torch import Tensor
from torch.jit import ScriptModule

MODEL_FILENAME = 'state_agent.pt'


def limit_period(angle):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2


def get_ball_to_goal_tensor(ball_center: Tensor, goal_center: Tensor) -> Tensor:
    ball_to_goal_diff = goal_center - ball_center
    ball_to_goal_angle = torch.atan2(ball_to_goal_diff[1], ball_to_goal_diff[0])
    ball_to_goal_dist = torch.norm(ball_to_goal_diff)
    return torch.tensor([ball_to_goal_angle, ball_to_goal_dist], dtype=torch.float32)


def get_kart_to_ball_tensor(kart_state: dict, ball_center: Tensor) -> Tensor:
    kart_front = torch.tensor(kart_state['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(kart_state['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = kart_front - kart_center
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    kart_to_puck_direction = ball_center - kart_center
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0])

    # kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle) / np.pi)
    kart_to_puck_distance = torch.norm(kart_to_puck_direction)

    return torch.tensor(
        [kart_angle, kart_to_puck_angle, kart_to_puck_distance],
        dtype=torch.float32)


def state_to_tensor(team_id: int,
                    player_state: List[dict],
                    opponent_state: List[dict],
                    soccer_state: dict,
                    ) -> Tensor:

    # features of soccer ball
    ball_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]

    # features of goal-lines
    goal_centers = (
        torch.tensor(
            soccer_state['goal_line'][(team_id + i) % 2],
            dtype=torch.float32
        )[:, [0, 2]].mean(dim=0)
        for i in range(2)
    )
    goal_tensors = [
        get_ball_to_goal_tensor(ball_center, goal_center)
        for goal_center in goal_centers
    ]

    # features of karts
    player_and_opponent_kart_states = (
        state['kart'] for state in player_state + opponent_state
    )
    kart_tensors = [
        get_kart_to_ball_tensor(kart_state, ball_center)
        for kart_state in player_and_opponent_kart_states
    ]

    features_tensor = torch.cat(
        [
            torch.tensor([team_id], dtype=torch.float32),  # 1 dim
            *goal_tensors,  # 2 goals * (angle, dist) -> 4 dim
            *kart_tensors,  # 4 players * (kart angle, kart to ball angle, ball dist) -> 12 dim
        ])

    return features_tensor


def get_kart_tensor_jurgen(kart_state: dict, ball_center: Tensor) -> Tensor:
    kart_front = torch.tensor(kart_state['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(kart_state['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = kart_front - kart_center
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    kart_to_puck_direction = ball_center - kart_center
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0])

    # kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle) / np.pi)
    # kart_to_puck_distance = torch.norm(kart_to_puck_direction)

    return torch.tensor(
        [*kart_center, kart_angle, kart_to_puck_angle],
        dtype=torch.float32)


def state_to_tensor_jurgen_by_player(team_id: int,
                                     player_state: dict,
                                     _: List[dict],  # opponent_state
                                     soccer_state: dict,
                                     ) -> Tensor:

    # features of soccer ball
    ball_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]

    # features of goal-lines
    attacking_goal_center = torch.tensor(
        soccer_state['goal_line'][(team_id + 1) % 2],
        dtype=torch.float32
    )[:, [0, 2]].mean(dim=0)
    ball_to_goal_direction = (attacking_goal_center - ball_center)
    ball_to_goal_unit_vector = ball_to_goal_direction / torch.linalg.norm(ball_to_goal_direction)

    # features of karts
    kart_tensor = get_kart_tensor_jurgen(player_state['kart'], ball_center)
    kart_to_ball_angle_difference = limit_period((kart_tensor[2] - kart_tensor[3]) / torch.pi)

    return torch.tensor([
        *kart_tensor,
        *attacking_goal_center,
        kart_to_ball_angle_difference,
        *ball_center,
        *ball_to_goal_unit_vector
    ], dtype=torch.float32)


def state_to_tensor_jurgen(team_id: int,
                           team_state: List[dict],
                           opponent_state: List[dict],
                           soccer_state: dict,
                           ) -> Tensor:
    result = torch.cat([
        state_to_tensor_jurgen_by_player(team_id, player_state, opponent_state, soccer_state)
        for player_state in team_state
    ])
    return result


def load_model(filename: str = MODEL_FILENAME) -> ScriptModule:
    return torch.jit.load(path.join(path.dirname(path.abspath(__file__)), filename))


def save_model(model: ScriptModule, filename: str = MODEL_FILENAME):
    torch.jit.save(model, path.join(path.dirname(path.abspath(__file__)), filename))


def copy_parameters(src_model: ScriptModule, dest_model: ScriptModule):
    state_dict = src_model.state_dict()
    dest_model.load_state_dict(state_dict)
