from typing import List, Optional, Tuple, Union

import torch
from state_agent.player import Team
from state_agent.utils import state_to_tensor
from torch import Tensor
from torch.jit import ScriptModule
from torch.utils.data import Dataset
from tournament.runner import Match, TeamRunner
from tournament.utils import (BaseRecorder, DataRecorder, MultiRecorder,
                              VideoRecorder)

from trainer.reward import RewardCriteria, get_match_rewards


class FramesDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data: List[Tuple[Tensor, List[dict], float]] = []

    def add_data(self,
                 matches_data: List[List[dict]],
                 red_matches_rewards: List[List[float]],
                 blue_matches_rewards: List[List[float]]):
        data: List[Tuple[Tensor, List[dict], float]] = []
        for match_data, red_rewards in zip(matches_data, red_matches_rewards):
            state_tensor: Optional[Tensor] = None
            for frame_data, next_frame_data, reward in zip(match_data,
                                                           match_data[1:],
                                                           red_rewards):
                if state_tensor is None:
                    state_tensor = state_to_tensor(0,
                                                   frame_data['team1_state'],
                                                   frame_data['team2_state'],
                                                   frame_data['soccer_state'])
                next_state_tensor = state_to_tensor(0,
                                                    next_frame_data['team1_state'],
                                                    next_frame_data['team2_state'],
                                                    next_frame_data['soccer_state'])
                actions = self.action_dicts_to_tensor(frame_data['actions'][0::2])
                data.append((state_tensor, actions, reward, next_state_tensor))

                state_tensor = next_state_tensor

        for match_data, blue_rewards in zip(matches_data, blue_matches_rewards):
            state_tensor: Optional[Tensor] = None
            for frame_data, next_frame_data, reward in zip(match_data,
                                                           match_data[1:],
                                                           blue_rewards):
                if state_tensor is None:
                    state_tensor = state_to_tensor(1,
                                                   frame_data['team2_state'],
                                                   frame_data['team1_state'],
                                                   frame_data['soccer_state'])
                next_state_tensor = state_to_tensor(1,
                                                    next_frame_data['team2_state'],
                                                    next_frame_data['team1_state'],
                                                    next_frame_data['soccer_state'])
                actions = self.action_dicts_to_tensor(frame_data['actions'][1::2])
                data.append((state_tensor, actions, reward, next_state_tensor))

        self.data += data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform is not None:
            data = self.transform(*data)
        return data

    def discard_to_max_size(self, max_size: int):
        if max_size >= len(self):
            return

        random_indices = torch.randperm(len(self))[:max_size]
        self.data = [self.data[idx.item()] for idx in random_indices]

    @staticmethod
    def action_dicts_to_tensor(action_dicts: List[dict]) -> Tensor:
        action_lists = [
            [
                action_dict['acceleration'],
                action_dict['steer'],
                action_dict['brake'],
            ]
            for action_dict in action_dicts
        ]
        return torch.tensor(action_lists, dtype=torch.float32).flatten()


def merge_datasets(*frame_datasets: FramesDataset) -> FramesDataset:
    merged_dataset = FramesDataset(frame_datasets[0].transform)
    for frame_dataset in frame_datasets:
        merged_dataset.data += frame_dataset.data

    return merged_dataset


def generate_data(team_or_dir1: Union[ScriptModule, str],
                  team_or_dir2: Union[ScriptModule, str],
                  num_matches: int,
                  reward_criteria: RewardCriteria,
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
            get_match_rewards(match_data, 0, reward_criteria)
            for match_data in matches_data
        ]

    blue_matches_rewards: List[List[float]] = [[]]
    if use_blue_data:
        blue_matches_rewards = [
            get_match_rewards(match_data, 1, reward_criteria)
            for match_data in matches_data
        ]

    dataset = FramesDataset()
    dataset.add_data(matches_data, red_matches_rewards, blue_matches_rewards)
    return dataset


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
