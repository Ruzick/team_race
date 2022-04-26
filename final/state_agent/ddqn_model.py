import itertools
from typing import List

import torch
from torch import Tensor, nn
from torch.jit import ScriptModule


DEFAULT_DEVICE: torch.device = torch.device('cpu')
FLIP_FOR_BLUE: bool = True
EPSILON: float = 0.1


class DQNModel(nn.Module):
    def __init__(self,
                 device: torch.device = DEFAULT_DEVICE,
                 flip_for_blue: bool = FLIP_FOR_BLUE):
        super().__init__()
        # TODO: Choose a decent network architecture
        self.network = torch.nn.Sequential(
            torch.nn.Linear(22, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        self.device = device
        self.flip_for_blue = flip_for_blue

    def forward(self, input_tensor) -> Tensor:
        original_device = input_tensor.device
        input_tensor = torch.atleast_2d(input_tensor.to(self.device))

        if input_tensor.size(-1) == 23:
            # Input actions have not been made discrete, do this now
            input_tensor = self.make_input_action_discrete(input_tensor)

        # This assumes the first component of each input tensor is the team id
        # If you violate that assumption, you will need to change this
        is_blue = input_tensor[:, 0] == 1
        if self.flip_for_blue:
            input_tensor = self.flip_input_for_blue(input_tensor, is_blue)

        # Remove first component of state (the team) since it is not needed
        input_tensor = input_tensor[:, 1:]

        # Remove last component of action (the team) because it is currently a dud
        input_tensor = input_tensor[:, :-1]

        network_output: Tensor = self.network(input_tensor)
        output_tensor = network_output

        return torch.squeeze(output_tensor).to(original_device)

    @staticmethod
    def make_input_action_discrete(input_tensor: Tensor) -> Tensor:
        assert input_tensor.size(-1) == 23, \
            f'Input tensor size {input_tensor.shape} does not match expected size'

        action_extraction_info = [
            # (13, -100000, 0.5),
            (13, 0.5, 100000),    # P1 accelerate
            (14, -100000, -0.5),  # P1 turn left
            # (14, -0.5, 0.5),
            (14, 0.5, 100000),    # P1 turn right
            # (15, -100000, 0.5),
            # (15, 0.5, 100000),    # P1 brake
            # (16, -100000, 0.5),
            (16, 0.5, 100000),    # P2 accelerate
            (17, -100000, -0.5),  # P2 turn left
            # (17, -0.5, 0.5),
            (17, 0.5, 100000),    # P2 turn right
            # (18, -100000, 0.5),
            # (18, 0.5, 100000),    # P2 brake
            (18, 10000, 10000),   # dud
        ]

        discretized_actions = torch.stack([
            torch.logical_and(
                input_tensor[:, input_idx] > lower_bound,
                input_tensor[:, input_idx] < upper_bound
            )
            for input_idx, lower_bound, upper_bound in action_extraction_info
        ], dim=-1)
        return torch.cat([input_tensor[:, :17], discretized_actions], dim=-1)

    @staticmethod
    def flip_input_for_blue(input_tensor: Tensor, is_blue: Tensor) -> Tensor:
        assert input_tensor.size(-1) == 24, \
            f'Input tensor size {input_tensor.shape} does not match expected size'
        should_flip_feature = torch.zeros(input_tensor.size(-1), dtype=torch.bool)
        should_flip_feature[1] = True  # ball to defence goal angle
        should_flip_feature[3] = True  # ball to attack goal angle
        should_flip_feature[5] = True  # player 1 angle
        should_flip_feature[6] = True  # player 1 to ball angle
        should_flip_feature[8] = True  # player 2 angle
        should_flip_feature[9] = True  # player 2 to ball angle
        should_flip_feature[11] = True  # opponent 1 angle
        should_flip_feature[12] = True  # opponent 1 to ball angle
        should_flip_feature[14] = True  # opponent 2 angle
        should_flip_feature[15] = True  # opponent 2 to ball angle

        flip_multiple = torch.where(should_flip_feature, -1., 1.).to(input_tensor.device)
        return torch.where(is_blue[:, None], input_tensor * flip_multiple[None, :], input_tensor)


class DQNPlayerModel(nn.Module):
    def __init__(self, dqn_model: ScriptModule, epsilon: float = EPSILON):
        super().__init__()
        self.dqn_model = dqn_model
        self.epsilon = epsilon
        self.discretized_actions: List[Tensor] = self.get_discretized_actions()
        self.action_len: int = self.discretized_actions[0].numel()

    def forward(self, state_tensor: Tensor):
        if len(state_tensor.squeeze().shape) >= 2:
            raise RuntimeError('This model does not support batching for forward')

        state_tensor = torch.atleast_2d(state_tensor)
        is_blue = state_tensor[0, 0].item() != 0

        # action: Tensor
        if self.training and torch.rand(1) < self.epsilon:
            action = self.get_random_action()
        else:
            action = self.get_best_action(state_tensor).squeeze()

        return torch.cat([
            self.to_output_tensor(action[0:self.action_len // 2], is_blue),
            self.to_output_tensor(action[self.action_len // 2:self.action_len], is_blue),
        ])

    def get_best_action(self, state_batch: Tensor) -> Tensor:
        dqn_outputs: List[Tensor] = []
        for discretized_action in self.discretized_actions:
            broadcasted_discretized_action = torch.broadcast_to(
                discretized_action, (state_batch.size(0), discretized_action.numel()))
            dqn_input = torch.cat([state_batch, broadcasted_discretized_action], dim=-1)
            dqn_output: Tensor = self.dqn_model.forward(dqn_input)
            dqn_outputs.append(dqn_output)

        best_action_indices = torch.argmax(torch.stack(dqn_outputs), dim=0)
        return torch.stack(self.discretized_actions)[best_action_indices]

    def get_random_action(self) -> Tensor:
        index: int = int(torch.randint(len(self.discretized_actions), (1,)).item())
        return self.discretized_actions[index]

    @staticmethod
    def get_discretized_actions() -> Tensor:
        indices = torch.arange(7)  # Add a dud input to avoid confusion with continuous action
        indices_of_each_action = [
            [-1, 0],     # P1 reverse or accelerate
            [-1, 1, 2],  # P1 no steer, turn left or turn right
            [-1, 3],     # P2 reverse or accelerate
            [-1, 4, 5],  # P2 no steer, turn left or turn right
        ]
        one_hots_of_each_action = [
            [
                (indices == index).type(torch.float32)
                for index in indices_of_action
            ]
            for indices_of_action in indices_of_each_action
        ]

        discretized_actions: List[Tensor] = [
            sum(one_hots_of_action_permutation)
            for one_hots_of_action_permutation in itertools.product(*one_hots_of_each_action)
        ]

        assert len(discretized_actions) == 36, 'Unexpected number of discretized actions'
        return discretized_actions

    @staticmethod
    def to_output_tensor(discrete_action_fragment: Tensor, is_blue: bool) -> Tensor:
        assert discrete_action_fragment.numel() >= 3, \
            'Unexpected number of elements in action fragment'

        output_tensor = torch.zeros(3)

        # Acceleration
        # assert sum(discrete_action_fragment[0]) <= 1, \
        #     'Both accelerate and reverse are set'
        if discrete_action_fragment[0] == 1:
            output_tensor[0] = 1  # Accelerate
        elif discrete_action_fragment[0] == 0:
            output_tensor[2] = 1  # Reverse
        else:
            raise RuntimeError('Unxpected acceleration/reverse')

        # Steer
        assert sum(discrete_action_fragment[1:3]) <= 1, \
            'Steer set to both -1 and 1'
        if discrete_action_fragment[1] == 1:
            output_tensor[1] = -1
        elif discrete_action_fragment[2] == 1:
            output_tensor[1] = 1

        if is_blue:
            output_tensor[1] *= -1

        return output_tensor
