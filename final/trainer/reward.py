from enum import IntFlag, auto
from typing import List

import torch
from torch import Tensor


SCORE_UTILITY_MULTIPLIER = 10000


class RewardCriterion(IntFlag):
    '''
    Determines what factors should be considered when giving an agent a reward
    for its actions.
    '''
    # No reward
    NONE = auto()

    # Reward an agent for being closer to the ball
    PLAYER_TO_BALL_DIST = auto()

    # Reward for getting ball near opponent's goal and away from team's goal
    BALL_TO_GOAL_DIST = auto()

    # Reward for scoring and penality for getting scored against
    SCORE = auto()


class RewardCriteria:
    '''
    Holds many RewardCriterion objects.
    '''
    def __init__(self, *criteria: RewardCriterion):
        self.criteria = RewardCriterion.NONE
        for criterion in criteria:
            self.criteria |= criterion

    def __contains__(self, criterion: RewardCriterion):
        return (self.criteria & criterion) == criterion


def get_match_rewards(match_data: List[dict], team_id: int, reward_criteria: RewardCriteria
                      ) -> List[float]:
    rewards = torch.zeros(len(match_data) - 1, dtype=torch.float32)
    if RewardCriterion.PLAYER_TO_BALL_DIST in reward_criteria:
        rewards += _get_player_to_ball_dist_rewards(match_data, team_id)
    if RewardCriterion.BALL_TO_GOAL_DIST in reward_criteria:
        rewards += _get_ball_to_goal_dist_rewards(match_data, team_id)
    if RewardCriterion.SCORE in reward_criteria:
        rewards += _get_score_rewards(match_data, team_id)

    return list(rewards.numpy())


def _get_player_to_ball_dist_rewards(match_data: List[dict], team_id: int) -> Tensor:
    utilities = torch.tensor([
        torch.tensor(_players_dist_from_goals(frame_data[f'team{team_id + 1}_state'],
                                              frame_data['soccer_state'])
                     ).sum().item()
        for frame_data in match_data
    ], dtype=torch.float32)

    rewards = utilities[1:] - utilities[:-1]
    return rewards


def _players_dist_from_goals(team_state: dict, soccer_state: dict) -> List[float]:
    # features of soccer ball
    ball_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]

    # features of goal-lines
    player_centers = (
        torch.tensor(
            player_state['kart']['location'],
            dtype=torch.float32
        )[[0, 2]]
        for player_state in team_state
    )

    return [
        torch.linalg.norm(player_center - ball_center).item() for player_center in player_centers
    ]


def _get_ball_to_goal_dist_rewards(match_data: List[dict], team_id: int) -> Tensor:
    utilities = torch.tensor([
        (
            torch.tensor(_dist_from_goals(frame_data['soccer_state'], team_id))
            * torch.tensor([-1, 1])
        ).sum().item()
        for frame_data in match_data
    ], dtype=torch.float32)

    rewards = utilities[1:] - utilities[:-1]
    return rewards


def _dist_from_goals(soccer_state: dict, team_id: int) -> List[float]:
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

    return [
        torch.linalg.norm(goal_center - ball_center).item() for goal_center in goal_centers
    ]


def _get_score_rewards(match_data: List[dict], team_id: int) -> Tensor:
    utilities = torch.tensor([
        (frame_data['soccer_state']['score'][team_id]
         - frame_data['soccer_state']['score'][(team_id + 1) % 2]
         ) * SCORE_UTILITY_MULTIPLIER
        for frame_data in match_data
    ], dtype=torch.float32)

    rewards = utilities[1:] - utilities[:-1]
    return rewards
