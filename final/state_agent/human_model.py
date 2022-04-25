from torch import Tensor, nn
import torch


DEFAULT_DEVICE: torch.device = torch.device('cpu')


class HumanModel(nn.Module):
    def __init__(self,
                 device: torch.device = DEFAULT_DEVICE):
        super().__init__()
        self.device = device

    def forward(self, input_tensor) -> Tensor:
        original_device = input_tensor.device
        input_tensor = input_tensor.to(self.device)

        steering_multiplier = 10.

        output_tensor: Tensor = torch.tensor([
            1., steering_multiplier * float(input_tensor[5].item()), 0.,
            1., steering_multiplier * float(input_tensor[7].item()), 0.,
        ], dtype=torch.float32)

        output_tensor = self.clip_output(output_tensor)
        return torch.squeeze(output_tensor).to(original_device)

    @staticmethod
    def clip_output(output_tensor: Tensor) -> Tensor:
        upper_bound = torch.tensor(
            [1., 1., 1., 1., 1., 1.], dtype=torch.float32
        ).to(output_tensor.device)
        lower_bound = torch.tensor(
            [0., -1., 0., 0., -1., 0.], dtype=torch.float32
        ).to(output_tensor.device)

        return output_tensor.maximum(lower_bound).minimum(upper_bound)
