import argparse

from trainer.algorithm import (Algorithm, AlgorithmImpl, all_algorithms)
from trainer.ddpg import DDPG
from trainer.ddqn import DDQN

LOG_DIR = 'log/'


BATCH_SIZE = 128
EPOCHS = 100000
VIDEO_EPOCHS_INTERVAL = 0


def get_impl(algorithm: Algorithm) -> AlgorithmImpl:
    if algorithm is Algorithm.DDPG:
        return DDPG()
    if algorithm is Algorithm.DDQN:
        return DDQN()

    raise NotImplementedError(f'Algorithm {algorithm} is not yet implemented')


def main():
    parser = argparse.ArgumentParser()

    # Shared args. Algorithm-specific args should go in algorithm subparser
    parser.add_argument('-e', '--epochs', type=int, default=EPOCHS)
    parser.add_argument('-vel', '--video-epochs-interval', type=int, default=VIDEO_EPOCHS_INTERVAL)
    parser.add_argument('-bs', '--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--no-flip-for-blue', action='store_true')

    subparsers = parser.add_subparsers(required=True,
                                       help='Training algorithm (look at algorithm.py)')
    for algorithm in all_algorithms():
        algorithm_impl = get_impl(algorithm)
        subparser = subparsers.add_parser(algorithm.name)
        algorithm_impl.init_subparser(subparser)
        subparser.set_defaults(train=algorithm_impl.train)

    args = parser.parse_args()
    args.train(args)


if __name__ == '__main__':
    main()
