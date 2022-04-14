from torch import Tensor, nn
import torch


FLIP_FOR_BLUE: bool = True


class StateModel(nn.Module):
    def __init__(self,
                 flip_for_blue: bool = FLIP_FOR_BLUE):
        super().__init__()
        self.device = torch.device('cpu')
        # TODO: Choose a decent network architecture
        self.network = torch.nn.Linear(13, 6)
        self.flip_for_blue = flip_for_blue

    def forward(self, input_tensor: Tensor):
        input_tensor = torch.atleast_2d(input_tensor.to(self.device))

        # This assumes the first component of each input tensor is the team id
        # If you violate that assumption, you will need to change this
        is_blue = input_tensor[:, 0] == 1
        if self.flip_for_blue:
            input_tensor = flip_input_for_blue(input_tensor, is_blue)

        # TODO: Build a decent model
        network_output: Tensor = self.network(input_tensor)
        output_tensor = network_output

        if self.flip_for_blue:
            output_tensor = flip_output_for_blue(output_tensor, is_blue)

        return torch.squeeze(output_tensor)


def flip_input_for_blue(input_tensor: Tensor, is_blue: Tensor) -> Tensor:
    assert input_tensor.size(-1) == 13, \
           f'Input tensor size {input_tensor.shape} does not match expected size'
    should_flip_feature = torch.zeros(input_tensor.size(-1), dtype=torch.bool)
    should_flip_feature[1] = True  # ball to defence goal angle
    should_flip_feature[3] = True  # ball to attack goal angle

    flip_multiple = torch.where(should_flip_feature, -1., 1.).to(input_tensor.device)
    return torch.where(is_blue[:, None], input_tensor * flip_multiple[None, :], input_tensor)


def flip_output_for_blue(output_tensor: Tensor, _: Tensor) -> Tensor:
    assert output_tensor.size(-1) == 6, \
           f'Output tensor size {output_tensor.shape} does not match expected size'

    # Currently logic does not require any flipping for output
    return output_tensor
