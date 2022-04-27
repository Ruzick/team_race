import torch
from torch import Tensor, nn


DEFAULT_DEVICE: torch.device = torch.device('cpu')


class DaggerModel(nn.Module):
    def __init__(self,
                 device: torch.device = DEFAULT_DEVICE):
        super().__init__()
        # TODO: Choose a decent network architecture
        self.network = torch.nn.Sequential(
            torch.nn.Linear(11, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Linear(32, 3),
        )
        self.device = device
        self.eval()

    def forward(self, input_tensor) -> Tensor:
        original_device = input_tensor.device
        input_tensor = torch.atleast_2d(input_tensor.to(self.device))

        half_last_dim = input_tensor.size(-1) // 2
        network_output_1: Tensor = self.network(input_tensor[..., :half_last_dim])
        network_output_2: Tensor = self.network(input_tensor[..., half_last_dim:])
        p1_output = self.to_output_tensor(network_output_1)
        p2_output = self.to_output_tensor(network_output_2)

        return torch.cat([p1_output, p2_output], dim=-1).to(original_device)

    def to_output_tensor(self, action_fragment: Tensor) -> Tensor:
        assert action_fragment.size(-1) == 3, \
            'Unexpected number of elements in action fragment'

        if self.training:
            return action_fragment

        assert action_fragment.size(0) == 1, 'batching is only supported in training'
        action_fragment = action_fragment.squeeze()

        brake = torch.round(action_fragment[2])

        return torch.stack([
            torch.sigmoid(action_fragment[0])
            if brake != 0 else torch.tensor(0., dtype=torch.float32).to(action_fragment.device),
            2 * torch.sigmoid(action_fragment[1]) - 1,
            torch.round(action_fragment[2])
        ])
