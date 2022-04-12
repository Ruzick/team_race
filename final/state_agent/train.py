import argparse

import torch

from state_agent.utils import save_model

from .model import StateModel


def main():
    _ = argparse.ArgumentParser()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)

    model = torch.jit.script(StateModel()).to(device)
    model.device = device

    # Do training here

    save_model(model)


if __name__ == '__main__':
    main()
