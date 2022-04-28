
import argparse

import torch
from state_agent.human_model import HumanModel
from state_agent.utils import save_model, state_to_tensor_human
from torch.jit import ScriptModule
from tournament.runner import Match

from trainer.algorithm import AlgorithmImpl
from trainer.data import generate_data
from trainer.reward import RewardCriteria, RewardCriterion

LOG_DIR = 'log/'
MAX_EPOCH_SAMPLES = 1000
MAX_DATASET_SIZE = 20000
GAMMA = 0.99
TAU = 0.001
LR = 0.01
NOISE_STANDARD_DEVIATION = 0.1


class HUMAN(AlgorithmImpl):
    def init_subparser(self, _: argparse.ArgumentParser):
        pass

    def train(self, args: argparse.Namespace):
        train(args)


def get_model_generator(model_class, **kwargs):
    return lambda: torch.jit.script(model_class(**kwargs))


def train(_: argparse.Namespace):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)

    human_model_generator = get_model_generator(
        HumanModel,
        device=device,
    )

    human_model: ScriptModule = human_model_generator().to(device)

    match = Match()

    reward_criteria = RewardCriteria(RewardCriterion.PLAYER_TO_BALL_DIST,
                                     RewardCriterion.SCORE)

    # generate_data(
    #     match, human_model, human_model, 1, reward_criteria,
    #     # video_path='human-test.mp4',
    #     state_to_tensor_fn=state_to_tensor_human)
    generate_data(
        match, human_model, human_model, 1, reward_criteria,
        video_path='human-test.mp4',
        state_to_tensor_fn=state_to_tensor_human)

    save_model(human_model)
