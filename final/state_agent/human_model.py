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
        input_tensor = torch.atleast_2d(input_tensor.to(self.device))

        output_tensor: Tensor = torch.tensor([
            1., -1 * input_tensor[3].item(), 0.,
            1., -1 * input_tensor[5].item(), 0.,
        ], dtype=torch.float32)

        return torch.squeeze(output_tensor).to(original_device)
