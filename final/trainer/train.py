import argparse
from os import path
from typing import Optional

import torch
import torch.utils.tensorboard as tb
from state_agent.model import ActorModel, CriticModel
from state_agent.utils import copy_parameters, save_model
from torch import Tensor
from torch.jit import ScriptModule
from torch.utils.data import ConcatDataset, DataLoader

from trainer.data import FramesDataset, generate_data


LOG_DIR = 'log/'


BATCH_SIZE = 128
EPOCHS = 100000
MAX_EPOCH_SAMPLES = 1000
GAMMA = 0.99
TAU = 0.001
ACTOR_LR = 0.01
CRITIC_LR = 0.01
VIDEO_EPOCHS_INTERVAL = 10


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

    actor_model_generator = get_model_generator(
        ActorModel,
        device=device,
        flip_for_blue=not args.no_flip_for_blue
    )
    critic_model_generator = get_model_generator(
        CriticModel,
        device=device,
        flip_for_blue=not args.no_flip_for_blue
    )

    actor_model: ScriptModule = actor_model_generator().to(device)
    critic_model: ScriptModule = critic_model_generator().to(device)

    target_actor_model: ScriptModule = actor_model_generator().to(device)
    target_critic_model: ScriptModule = critic_model_generator().to(device)
    copy_parameters(actor_model, target_actor_model)
    copy_parameters(critic_model, target_critic_model)
    disable_grad(target_actor_model)
    disable_grad(target_critic_model)

    critic_loss = torch.nn.MSELoss(reduction='sum')

    actor_optimizer = torch.optim.Adam(actor_model.parameters())
    critic_optimizer = torch.optim.Adam(critic_model.parameters())

    dataset = FramesDataset([[]], [[]], [[]])

    for i_epoch in range(args.epochs):
        dataset = ConcatDataset([
            generate_data('jurgen_agent', actor_model, 1, use_red_data=False,
                          video_path=get_video_path(i_epoch, args.video_epochs_interval, 'blue')),
            generate_data(actor_model, 'jurgen_agent', 1, use_blue_data=False,
                          video_path=get_video_path(i_epoch, args.video_epochs_interval, 'red')),
            generate_data(actor_model, actor_model, 1,
                          video_path=get_video_path(i_epoch, args.video_epochs_interval, 'both'))
        ])
        data_loader = DataLoader(dataset, args.batch_size, shuffle=True)

        print(f'Starting epoch {i_epoch} with dataset size {len(dataset)}')

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        epoch_value_loss: Tensor = torch.tensor(0.)
        epoch_utility: Tensor = torch.tensor(0.)

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

            value_input_actual_action = torch.cat((state_batch, action_batch), dim=-1)
            value_batch_actual_action: Tensor = critic_model(value_input_actual_action)

            pred_action_batch: Tensor = torch.squeeze(actor_model(state_batch), dim=-1)
            value_input_pred_action = torch.cat((state_batch, pred_action_batch), dim=-1)
            value_batch_pred_action: Tensor = critic_model(value_input_pred_action)

            pred_next_action_batch: Tensor = torch.squeeze(target_actor_model(next_state_batch),
                                                           dim=-1)
            value_input_pred_next_action = torch.cat(
                (next_state_batch, pred_next_action_batch), dim=-1)
            value_batch_pred_next_action: Tensor = target_critic_model(
                value_input_pred_next_action)

            approx_value_actual_action = reward_batch + args.gamma * value_batch_pred_next_action
            state_value_loss: Tensor = critic_loss(approx_value_actual_action,
                                                   value_batch_actual_action)

            epoch_value_loss += state_value_loss
            epoch_utility += value_batch_pred_action.sum()

            n_epoch_samples += state_batch.size(0)

        epoch_value_loss /= n_epoch_samples
        epoch_utility /= n_epoch_samples
        epoch_value_loss.backward()
        epoch_utility.backward()

        actor_optimizer.step()
        critic_optimizer.step()

        log(train_logger, epoch_value_loss, epoch_utility, i_epoch)

        for param, target_param in zip(actor_model.parameters(),
                                       target_actor_model.parameters()):
            target_param.data.copy_(param.data * args.tau + target_param.data * (1.0 - args.tau))

        for param, target_param in zip(critic_model.parameters(),
                                       target_critic_model.parameters()):
            target_param.data.copy_(param.data * args.tau + target_param.data * (1.0 - args.tau))

    save_model(actor_model)


def log(logger: tb.SummaryWriter, value_loss: float, utility: float, global_step: int):
    logger.add_scalar('value_loss', value_loss, global_step=global_step)
    logger.add_scalar('utility', utility, global_step=global_step)


def main():
    parser = argparse.ArgumentParser()

    # Training args
    parser.add_argument('-e', '--epochs', type=int, default=EPOCHS)
    parser.add_argument('-g', '--gamma', type=float, default=GAMMA)
    parser.add_argument('-alr', '--actor-learning-rate', type=float, default=ACTOR_LR)
    parser.add_argument('-clr', '--critic-learning-rate', type=float, default=CRITIC_LR)
    parser.add_argument('--tau', type=float, default=TAU)
    parser.add_argument('-bs', '--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('-N', '--max-epoch-samples', type=int, default=MAX_EPOCH_SAMPLES)

    parser.add_argument('--no-flip-for-blue', action='store_true')

    parser.add_argument('-vel', '--video-epochs-interval', type=int, default=VIDEO_EPOCHS_INTERVAL)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
