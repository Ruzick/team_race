import argparse

import torch
from state_agent.model import StateModel
from state_agent.utils import save_model
from torch import Tensor
from torch.jit import ScriptModule
from torch.utils.data import DataLoader

from trainer.data import generate_data


BATCH_SIZE = 128


def train(args: argparse.Namespace):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)

    model: ScriptModule = torch.jit.script(StateModel(
        not args.no_flip_for_blue
    )).to(device)
    model.device = device

    # TODO: Start training here

    # Just trying both players on 1 example to check nothing is breaking
    # Nothing is getting learned!
    dataset = generate_data('jurgen_agent', model, 2, use_red_data=False, use_blue_data=True)
    data_loader = DataLoader(dataset, args.batch_size, shuffle=False)

    for features_batch, actions_batch, rewards_batch in data_loader:
        print('Feature tensor', features_batch[0])
        print(features_batch.size(0), len(actions_batch))
        output: Tensor = model(features_batch)
        output.sum().backward()

    save_model(model)


def main():
    parser = argparse.ArgumentParser()

    # Training args
    parser.add_argument('-bs', '--batch-size', type=int, default=BATCH_SIZE)

    parser.add_argument('--no-flip-for-blue', action='store_true')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
