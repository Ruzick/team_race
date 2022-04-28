import argparse
from os import path
from typing import Optional

import torch
import torch.utils.tensorboard as tb
from state_agent.ddqn_model import DQNModel, DQNNoOpModel, DQNPlayerModel
from state_agent.utils import (copy_parameters, load_model, save_model,
                               state_to_tensor_ddqn)
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
EPSILON = 0.1
RANDOM_EPOCHS = 20


class DDQN(AlgorithmImpl):
    def init_subparser(self, subparser: argparse.ArgumentParser):
        subparser.add_argument('-f', '--num-frames', type=int, default=None)
        subparser.add_argument('-g', '--gamma', type=float, default=GAMMA)
        subparser.add_argument('-lr', '--learning-rate', type=float, default=LR)
        subparser.add_argument('-re', '--random-epochs', type=int, default=RANDOM_EPOCHS)
        subparser.add_argument('-eps', '--epsilon', type=float, default=EPSILON)
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


def save_dqn_player_model(model: ScriptModule, i_epoch: Optional[int] = None):
    suffix = '' if i_epoch is None else f'_{i_epoch}'
    filename = f'dqn_player{suffix}.pt'
    save_model(model, filename)


def load_dqn_player_model(i_epoch: Optional[int] = None) -> ScriptModule:
    suffix = '' if i_epoch is None else f'_{i_epoch}'
    filename = f'dqn_player{suffix}.pt'
    return load_model(filename)


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
    dqn_player_model: ScriptModule = torch.jit.script(DQNPlayerModel(dqn_model))
    noop_player_model: ScriptModule = torch.jit.script(DQNNoOpModel())

    target_dqn_model: ScriptModule = dqn_model_generator().to(device)
    copy_parameters(dqn_model, target_dqn_model)
    disable_grad(target_dqn_model)

    dqn_loss = torch.nn.MSELoss()

    dqn_optimizer = torch.optim.Adam(dqn_model.parameters())

    dataset = FramesDataset(state_to_tensor_ddqn)

    reward_criteria = RewardCriteria(RewardCriterion.DDQN_CUSTOM)

    match = Match()

    global_step: int = 0

    for i_epoch in range(args.epochs):
        print(f'Starting epoch {i_epoch} with dataset size {len(dataset)}')

        if i_epoch < args.random_epochs:
            dqn_player_model.epsilon = 1.
        else:
            dqn_player_model.epsilon = args.epsilon

        generated_dataset = generate_data(
            match, dqn_player_model, noop_player_model, 1, reward_criteria,
            num_frames=args.num_frames,
            state_to_tensor_fn=state_to_tensor_ddqn,
            video_path=get_video_path(i_epoch, args.video_epochs_interval, 'red'))
        mean_generated_reward = torch.tensor([
            data_entry[2] for data_entry in generated_dataset.data
        ]).mean()
        train_logger.add_scalar('mean_generated_reward',
                                mean_generated_reward,
                                global_step=global_step)

        dataset = merge_datasets(
            dataset,
            # generate_data(match, 'jurgen_agent', dqn_player_model, 1, reward_criteria,
            #               num_frames=args.num_frames, use_red_data=False, use_blue_data=True,
            #              video_path=get_video_path(i_epoch, args.video_epochs_interval, 'blue')),
            # generate_data(match, dqn_player_model, 'jurgen_agent', 1, reward_criteria,
            #               num_frames=args.num_frames, use_red_data=True, use_blue_data=False,
            #               video_path=get_video_path(i_epoch, args.video_epochs_interval, 'red')),
            generated_dataset
        )
        dataset.discard_to_max_size(args.max_dataset_size)
        data_loader = DataLoader(dataset, args.batch_size, shuffle=True)

        num_epoch_batches: int = 0
        num_epoch_samples: int = 0
        for state_batch, action_batch, reward_batch, next_state_batch in data_loader:
            state_batch: Tensor
            action_batch: Tensor
            reward_batch: Tensor
            next_state_batch: Tensor

            next_state_best_action_batch = dqn_player_model.get_best_action(next_state_batch)

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

            dqn_optimizer.zero_grad()
            q_value_loss.backward()
            dqn_optimizer.step()

            num_epoch_batches += 1
            num_epoch_samples += state_batch.size(0)
            if num_epoch_samples >= args.max_epoch_samples:
                break

            mean_pred_q_value = pred_q_value_actual_action_batch.mean()

            log(train_logger, q_value_loss, mean_pred_q_value, global_step)
            global_step += 1

        for param, target_param in zip(dqn_model.parameters(),
                                       target_dqn_model.parameters()):
            target_param.data.copy_(param.data * args.tau + target_param.data * (1.0 - args.tau))

    save_dqn_player_model(dqn_player_model)


def log(logger: tb.SummaryWriter, value_loss: float, mean_q_value: float, global_step: int):
    logger.add_scalar('q_value_loss', value_loss, global_step=global_step)
    logger.add_scalar('mean_q_value', mean_q_value, global_step=global_step)
