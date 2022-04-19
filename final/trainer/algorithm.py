import argparse
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class Algorithm(Enum):
    DDPG = 'DDPG'
    DDQN = 'DDQN'
    HUMAN = 'HUMAN'


def all_algorithms() -> List[Algorithm]:
    return [
        Algorithm.DDPG,
        Algorithm.DDQN,
        Algorithm.HUMAN,
    ]


class AlgorithmImpl(ABC):
    @abstractmethod
    def init_subparser(self, subparser: argparse.ArgumentParser):
        pass

    @abstractmethod
    def train(self, args: argparse.Namespace):
        pass
