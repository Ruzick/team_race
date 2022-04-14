from typing import List, Optional, Tuple, Union

import torch

from torch import Tensor

from state_agent.player import Team
from torch.jit import ScriptModule
from torch.utils.data import Dataset
from state_agent.utils import state_to_tensor
from tournament.runner import Match, TeamRunner
from tournament.utils import (BaseRecorder, DataRecorder, MultiRecorder,
                              VideoRecorder)

SCORE_UTILITY_MULTIPLIER = 10000


class FramesDataset(Dataset):
    def __init__(self,
                 matches_data: List[List[dict]],
                 red_matches_rewards: List[List[float]],
                 blue_matches_rewards: List[List[float]],
                 transform=None):
        self.transform = transform

        self.data: List[Tuple[Tensor, List[dict], float]] = []
        for match_data, red_rewards in zip(matches_data, red_matches_rewards):
            for frame_data, reward in zip(match_data, red_rewards):
                features_tensor = state_to_tensor(0,
                                                  frame_data['team1_state'],
                                                  frame_data['team2_state'],
                                                  frame_data['soccer_state'])
                actions = frame_data['actions'][0::2]
                self.data.append((features_tensor, actions, reward))

        for match_data, blue_rewards in zip(matches_data, blue_matches_rewards):
            for frame_data, reward in zip(match_data, blue_rewards):
                features_tensor = state_to_tensor(1,
                                                  frame_data['team2_state'],
                                                  frame_data['team1_state'],
                                                  frame_data['soccer_state'])
                actions = frame_data['actions'][1::2]
                self.data.append((features_tensor, actions, reward))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform is not None:
            data = self.transform(*data)
        return data


def generate_data(team_or_dir1: Union[ScriptModule, str],
                  team_or_dir2: Union[ScriptModule, str],
                  num_matches: int,
                  use_red_data: bool = True,
                  use_blue_data: bool = True,
                  video_path: Optional[str] = None,
                  num_frames: Optional[int] = None,
                  initial_ball_location: Optional[Tuple[float, float]] = None,
                  initial_ball_velocity: Optional[Tuple[float, float]] = None
                  ) -> FramesDataset:

    if not use_red_data and not use_blue_data:
        raise RuntimeError("At least one team's data must be used")

    if isinstance(team_or_dir1, str):
        team1_runner = TeamRunner(team_or_dir1)
    else:
        team1_runner = TeamRunner(Team(team_or_dir1))

    if isinstance(team_or_dir2, str):
        team2_runner = TeamRunner(team_or_dir2)
    else:
        team2_runner = TeamRunner(Team(team_or_dir2))

    match = Match()
    matches_data = [
        play_match(match, team1_runner, team2_runner, video_path,
                   num_frames, initial_ball_location, initial_ball_velocity)
        for _ in range(num_matches)
    ]
    del match

    red_matches_rewards: List[List[float]] = [[]]
    if use_red_data:
        red_matches_rewards = [
            get_match_rewards(match_data, 0)
            for match_data in matches_data
        ]

    blue_matches_rewards: List[List[float]] = [[]]
    if use_blue_data:
        blue_matches_rewards = [
            get_match_rewards(match_data, 1)
            for match_data in matches_data
        ]

    return FramesDataset(matches_data, red_matches_rewards, blue_matches_rewards)


def play_match(match: Match,
               team1_runner: TeamRunner,
               team2_runner: TeamRunner,
               video_path: Optional[str] = None,
               num_frames: Optional[int] = None,
               initial_ball_location: Optional[Tuple[float, float]] = None,
               initial_ball_velocity: Optional[Tuple[float, float]] = None
               ) -> List[dict]:

    recorders: List[BaseRecorder] = []
    if video_path is not None:
        recorders.append(VideoRecorder(video_path))
    data_recorder = DataRecorder()
    recorders.append(data_recorder)

    optional_arguments_dict = {}
    if num_frames is not None:
        optional_arguments_dict['max_frames'] = num_frames
    if initial_ball_location is not None:
        optional_arguments_dict['initial_ball_location'] = initial_ball_location
    if initial_ball_velocity is not None:
        optional_arguments_dict['initial_ball_velocity'] = initial_ball_velocity

    match.run(team1_runner,
              team2_runner,
              num_player=2,
              record_fn=MultiRecorder(*recorders),
              **optional_arguments_dict)

    return data_recorder.data()


def dist_from_goals(soccer_state: dict, team_id: int) -> List[float]:
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
        torch.norm(goal_center - ball_center) for goal_center in goal_centers
    ]


def get_match_rewards(match_data: List[dict], team_id: int) -> List[float]:
    dist_from_goal_utilities = torch.tensor([
        (
            torch.tensor(dist_from_goals(frame_data['soccer_state'], team_id))
            * torch.tensor([-1, 1])
        ).sum().item()
        for frame_data in match_data
    ])

    score_utilites = torch.tensor([
        (frame_data['soccer_state']['score'][team_id]
         - frame_data['soccer_state']['score'][(team_id + 1) % 2]
         ) * SCORE_UTILITY_MULTIPLIER
        for frame_data in match_data
    ])

    utilities = dist_from_goal_utilities + score_utilites
    rewards = utilities[1:] - utilities[:-1]
    return list(rewards.numpy())
