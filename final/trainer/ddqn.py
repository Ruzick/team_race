import argparse
from os import path
from typing import Optional

import torch
import torch.utils.tensorboard as tb
from state_agent.ddqn_model import DQNModel, DQNPlayerModel
from state_agent.utils import copy_parameters, save_model
from torch import Tensor
from torch.jit import ScriptModule
from torch.utils.data import DataLoader
from tournament.runner import Match

from trainer.algorithm import AlgorithmImpl
from trainer.data import FramesDataset, generate_data, merge_datasets
from trainer.reward import RewardCriteria, RewardCriterion

LOG_DIR = 'log/'
MAX_EPOCH_SAMPLES = 1000
MAX_DATASET_SIZE = 20000
GAMMA = 0.99
TAU = 0.001
LR = 0.01
NOISE_STANDARD_DEVIATION = 0.1


class DDQN(AlgorithmImpl):
    def init_subparser(self, subparser: argparse.ArgumentParser):
        subparser.add_argument('-g', '--gamma', type=float, default=GAMMA)
        subparser.add_argument('-lr', '--learning-rate', type=float, default=LR)
        subparser.add_argument('-mds', '--max-dataset-size', type=int, default=MAX_DATASET_SIZE)
        subparser.add_argument('--tau', type=float, default=TAU)
        subparser.add_argument('-N', '--max-epoch-samples', type=int, default=MAX_EPOCH_SAMPLES)

    def train(self, args: argparse.Namespace):
        train(args)


def get_model_generator(model_class, **kwargs):
    return lambda: torch.jit.script(model_class(**kwargs))


def disable_grad(model: ScriptModule):
    for param in model.parameters():
        param.requires_grad = False


def get_video_path(i_epoch: int, video_epochs_interval: int, suffix: str) -> Optional[str]:
    if video_epochs_interval <= 0:
        return None

    if (i_epoch + 1) % video_epochs_interval != 0:
        return None

    return f'video_{i_epoch}_{suffix}.mp4'


def train(args: argparse.Namespace):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)

    train_logger = tb.SummaryWriter(path.join(LOG_DIR, 'train'), flush_secs=1)

    dqn_model_generator = get_model_generator(
        DQNModel,
        device=device,
        flip_for_blue=not args.no_flip_for_blue,
    )

    dqn_model: ScriptModule = dqn_model_generator().to(device)
    dqn_player_model = DQNPlayerModel(dqn_model)

    target_dqn_model: ScriptModule = dqn_model_generator().to(device)
    copy_parameters(dqn_model, target_dqn_model)
    disable_grad(target_dqn_model)

    dqn_loss = torch.nn.MSELoss(reduction='sum')

    dqn_optimizer = torch.optim.Adam(dqn_model.parameters())

    dataset = FramesDataset()

    reward_criteria = RewardCriteria(RewardCriterion.PLAYER_TO_BALL_DIST)
    match = Match()

    for i_epoch in range(args.epochs):
        print(f'Starting epoch {i_epoch} with dataset size {len(dataset)}')

        dataset = merge_datasets(
            dataset,
            generate_data(match, 'jurgen_agent', dqn_player_model, 1, reward_criteria,
                          use_red_data=False, use_blue_data=True,
                          video_path=get_video_path(i_epoch, args.video_epochs_interval, 'blue')),
            generate_data(match, dqn_player_model, 'jurgen_agent', 1, reward_criteria,
                          use_red_data=True, use_blue_data=False,
                          video_path=get_video_path(i_epoch, args.video_epochs_interval, 'red')),
            generate_data(match, dqn_player_model, dqn_player_model, 1, reward_criteria,
                          video_path=get_video_path(i_epoch, args.video_epochs_interval, 'both'))
        )
        dataset.discard_to_max_size(args.max_dataset_size)
        data_loader = DataLoader(dataset, args.batch_size, shuffle=True)

        dqn_optimizer.zero_grad()

        epoch_q_value_loss: Tensor = torch.tensor(0.)

        n_epoch_samples: int = 0
        for state_batch, action_batch, reward_batch, next_state_batch in data_loader:
            state_batch: Tensor
            action_batch: Tensor
            reward_batch: Tensor
            next_state_batch: Tensor

            n_batch_samples = min(state_batch.size(0), args.max_epoch_samples - n_epoch_samples)
            if n_batch_samples <= 0:
                break
            if n_batch_samples < state_batch.size(0):
                state_batch = state_batch[:n_batch_samples]
                action_batch = action_batch[:n_batch_samples]
                reward_batch = reward_batch[:n_batch_samples]
                next_state_batch = next_state_batch[:n_batch_samples]

            next_state_best_action_batch = dqn_player_model.get_best_action(dqn_model,
                                                                            next_state_batch)

            target_q_value_best_action_input = torch.cat(
                [next_state_batch, next_state_best_action_batch], dim=-1)
            target_q_value_best_action_batch: Tensor = target_dqn_model(
                target_q_value_best_action_input)

            target_q_value_actual_action_batch = (reward_batch
                                                  + args.gamma * target_q_value_best_action_batch)

            pred_q_value_actual_action_input = torch.cat(
                [state_batch, action_batch], dim=-1)
            pred_q_value_actual_action_batch: Tensor = dqn_model(
                pred_q_value_actual_action_input)

            q_value_loss: Tensor = dqn_loss(target_q_value_actual_action_batch,
                                            pred_q_value_actual_action_batch)

            epoch_q_value_loss += q_value_loss
            n_epoch_samples += state_batch.size(0)

        epoch_q_value_loss /= n_epoch_samples
        epoch_q_value_loss.backward()

        dqn_optimizer.step()

        log(train_logger, epoch_q_value_loss, i_epoch)

        for param, target_param in zip(dqn_model.parameters(),
                                       target_dqn_model.parameters()):
            target_param.data.copy_(param.data * args.tau + target_param.data * (1.0 - args.tau))

    save_model(dqn_model)


def log(logger: tb.SummaryWriter, value_loss: float, global_step: int):
    logger.add_scalar('q_value_loss', value_loss, global_step=global_step)
