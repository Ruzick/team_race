import argparse
from os import path
from typing import Callable, List, Optional

import torch
import torch.utils.tensorboard as tb
from state_agent.dagger_model import DaggerModel
from state_agent.utils import load_model, save_model, state_to_tensor_jurgen
from torch import Tensor
from torch.jit import ScriptModule
from torch.utils.data import DataLoader
from tournament.runner import Match

from trainer.algorithm import AlgorithmImpl
from trainer.data import FramesDataset, generate_data, merge_datasets
from trainer.reward import RewardCriteria, RewardCriterion

LOG_DIR = 'log/'
MAX_EPOCH_SAMPLES = 1000000
MAX_DATASET_SIZE = 50000
LR = 0.01


class Dagger(AlgorithmImpl):
    def init_subparser(self, subparser: argparse.ArgumentParser):
        subparser.add_argument('-f', '--num-frames', type=int, default=None)
        subparser.add_argument('-lr', '--learning-rate', type=float, default=LR)
        subparser.add_argument('-mds', '--max-dataset-size', type=int, default=MAX_DATASET_SIZE)
        subparser.add_argument('-N', '--max-epoch-samples', type=int, default=MAX_EPOCH_SAMPLES)

    def train(self, args: argparse.Namespace):
        train(args)


def get_model_generator(model_class, **kwargs):
    return lambda: torch.jit.script(model_class(**kwargs))


def save_dagger_player_model(model: ScriptModule, i_epoch: Optional[int] = None):
    suffix = '' if i_epoch is None else f'_{i_epoch}'
    filename = f'dagger_player{suffix}.pt'
    save_model(model, filename)


def load_dagger_player_model(i_epoch: Optional[int] = None) -> ScriptModule:
    suffix = '' if i_epoch is None else f'_{i_epoch}'
    filename = f'dagger_player{suffix}.pt'
    return load_model(filename)


def load_jurgen_model() -> ScriptModule:
    return load_model('jurgen_agent.pt')


def get_video_path(i_epoch: int, video_epochs_interval: int, suffix: Optional[str]
                   ) -> Optional[str]:
    if video_epochs_interval <= 0:
        return None

    if (i_epoch + 1) % video_epochs_interval != 0:
        return None

    return f'video_{i_epoch}_{suffix}.mp4'


def generate_dagger_data(match: Match,
                         dagger_model: ScriptModule,
                         target_model: ScriptModule,
                         opponents: List[str],
                         num_matches: int,
                         i_epoch: int,
                         video_epochs_interval: int = -1,
                         num_frames: Optional[int] = None) -> FramesDataset:
    reward_criteria = RewardCriteria(RewardCriterion.NONE)

    datasets = [
        generate_data(
            match, dagger_model, opponent, num_matches, reward_criteria,
            use_blue_data=False, num_frames=num_frames,
            state_to_tensor_fn=state_to_tensor_jurgen,
            video_path=get_video_path(
                i_epoch, video_epochs_interval, f'{opponent}-blue'),
            verbose=False)
        for opponent in opponents
    ] + [
        generate_data(
            match, opponent, dagger_model, num_matches, reward_criteria,
            use_red_data=False, num_frames=num_frames,
            state_to_tensor_fn=state_to_tensor_jurgen,
            video_path=get_video_path(
                i_epoch, video_epochs_interval, f'{opponent}-red'),
            verbose=False)
        for opponent in opponents
    ]

    dataset = merge_datasets(*datasets)
    for i_entry, entry in enumerate(dataset.data):
        state = entry[0].to(dagger_model.device)
        half_last_dim = state.size(-1) // 2
        action: Tensor = torch.cat([
            *target_model(state[:half_last_dim]),
            *target_model(state[half_last_dim:])
        ]).squeeze().detach().clone()
        dataset.data[i_entry] = (entry[0], action, entry[1], entry[2])

    return dataset


def adjust_target_output(target_output: Tensor) -> Tensor:
    # return ((target_output
    #          + torch.tensor([0., 1., 0.], dtype=torch.float32).to(target_output.device))
    #         * torch.tensor([1., 0.5, 1.], dtype=torch.float32).to(target_output.device))
    return (target_output
            + torch.tensor([0., 1., 0.], dtype=torch.float32).to(target_output.device)[None, :])


def adjust_model_output(model_output: Tensor, _: Tensor) -> Tensor:
    return model_output
    # # Set acceleration to 0 if target asks for braking, so that acceleration is not punished.
    # modifier = (adjusted_target_output[:, -1] != 0).detach().clone().type(torch.float32)
    # model_accel = modifier * model_output[:, 0]
    # return torch.cat([model_accel[:, None], model_output[:, 1:]], dim=-1)


def get_loss_fn() -> Callable[[Tensor, Tensor], Tensor]:
    # accel_loss_fn = torch.nn.BCEWithLogitsLoss()
    steer_loss_fn = torch.nn.CrossEntropyLoss()
    brake_loss_fn = torch.nn.BCEWithLogitsLoss()

    def loss_fn(model_output: Tensor, target: Tensor) -> Tensor:
        adjusted_target = adjust_target_output(target)
        model_output = adjust_model_output(model_output, adjusted_target)

        # accel_loss: Tensor = accel_loss_fn(model_output[:, 0], adjusted_target[:, 0])
        steer_loss: Tensor = steer_loss_fn(model_output[:, 0:3],
                                           adjusted_target[:, 1].type(torch.long))
        brake_loss: Tensor = brake_loss_fn(model_output[:, 3], adjusted_target[:, 2])
        # return accel_loss + steer_loss + brake_loss
        return steer_loss + brake_loss

    return loss_fn


def train(args: argparse.Namespace):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)

    train_logger = tb.SummaryWriter(path.join(LOG_DIR, 'train'), flush_secs=1)

    dagger_model_generator = get_model_generator(
        DaggerModel,
        device=device,
    )

    dagger_model: ScriptModule = dagger_model_generator().to(device)
    target_model: ScriptModule = load_jurgen_model().to(device)

    loss_fn = get_loss_fn()

    dagger_optimizer = torch.optim.Adam(dagger_model.parameters())

    dataset = FramesDataset(state_to_tensor_jurgen)

    match = Match()
    opponents = ['jurgen_agent', 'geoffrey_agent', 'yann_agent', 'yoshua_agent']

    global_step: int = 0
    best_epoch_loss: float = 1000000.

    for i_epoch in range(args.epochs):
        print(f'Starting epoch {i_epoch} with dataset size {len(dataset)}')

        dagger_model.eval()
        generated_dataset = generate_dagger_data(
            match, dagger_model, target_model, opponents, 1, i_epoch,
            args.video_epochs_interval, args.num_frames)
        dagger_model.train()

        # dataset = merge_datasets(
        #     dataset,
        #     # generate_data(match, 'jurgen_agent', dqn_player_model, 1, reward_criteria,
        #     #               num_frames=args.num_frames, use_red_data=False, use_blue_data=True,
        #     #              video_path=get_video_path(i_epoch, args.video_epochs_interval, 'blue')),
        #     # generate_data(match, dqn_player_model, 'jurgen_agent', 1, reward_criteria,
        #     #               num_frames=args.num_frames, use_red_data=True, use_blue_data=False,
        #     #               video_path=get_video_path(i_epoch, args.video_epochs_interval, 'red')),
        #     generated_dataset
        # )
        dataset = generated_dataset
        dataset.discard_to_max_size(args.max_dataset_size)
        data_loader = DataLoader(dataset, args.batch_size, shuffle=True)

        num_epoch_batches: int = 0
        num_epoch_samples: int = 0
        epoch_losses: List[Tensor] = []
        for state_batch, action_batch, _, _ in data_loader:
            state_batch: Tensor
            action_batch: Tensor
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)

            model_output = dagger_model(state_batch)
            half_dim_size = model_output.size(-1) // 2
            model_p1_output = model_output[:, :half_dim_size]
            model_p2_output = model_output[:, half_dim_size:]

            half_dim_size = action_batch.size(-1) // 2
            target_p1_output = action_batch[:, :half_dim_size]
            target_p2_output = action_batch[:, half_dim_size:]

            batch_loss: Tensor = (loss_fn(model_p1_output, target_p1_output)
                                  + loss_fn(model_p2_output, target_p2_output))

            dagger_optimizer.zero_grad()
            batch_loss.backward()
            dagger_optimizer.step()

            epoch_losses.append(batch_loss)
            num_epoch_batches += 1
            num_epoch_samples += state_batch.size(0)
            if num_epoch_samples >= args.max_epoch_samples:
                break

            log(train_logger, batch_loss, global_step)
            global_step += 1

        epoch_loss = float(torch.stack(epoch_losses).mean().item())
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            print(f'New best epoch loss: {best_epoch_loss}')
            save_dagger_player_model(dagger_model, i_epoch)

        save_dagger_player_model(dagger_model)

    dagger_model.eval()
    save_dagger_player_model(dagger_model)


def log(logger: tb.SummaryWriter, train_loss: float, global_step: int):
    logger.add_scalar('train_loss', train_loss, global_step=global_step)
