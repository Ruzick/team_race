import argparse
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from state_agent.model import StateModel
from state_agent.player import Team
from state_agent.utils import save_model, state_to_tensor
from torch.jit import ScriptModule
from tournament.runner import Match, TeamRunner
from tournament.utils import (BaseRecorder, DataRecorder, MultiRecorder,
                              VideoRecorder)


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


def generate_data(team_or_dir1: Union[ScriptModule, str],
                  team_or_dir2: Union[ScriptModule, str],
                  num_matches: int,
                  video_path: Optional[str] = None,
                  num_frames: Optional[int] = None,
                  initial_ball_location: Optional[Tuple[float, float]] = None,
                  initial_ball_velocity: Optional[Tuple[float, float]] = None
                  ) -> List[List[dict]]:
    """
    :return: List[List[dict]]  dict contains the relevant state at the given time
                               Inner list contains all the state for a match
                               Outer list contains all match states
    """
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

    return matches_data


def train(args: argparse.Namespace):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)

    model: ScriptModule = torch.jit.script(StateModel(
        not args.no_flip_for_blue
    )).to(device)
    model.device = device

    # TODO: Start training here

    # Just trying 1 player on 1 example to check nothing is breaking
    # Nothing is getting learned!
    matches_data = generate_data(model, 'jurgen_agent', 1)
    frame_data = matches_data[0][0]
    input_tensor = state_to_tensor(
        0,
        frame_data['team1_state'],
        frame_data['soccer_state'],
        frame_data['team2_state'],
        1)  # Blue
    print('Input tensor', input_tensor)
    output: Tensor = model(input_tensor)
    output.sum().backward()

    save_model(model)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-flip-for-blue', action='store_true')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
