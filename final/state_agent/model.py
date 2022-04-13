from torch import Tensor, nn
import torch


FLIP_FOR_BLUE: bool = True


class StateModel(nn.Module):
    def __init__(self,
                 flip_for_blue: bool = FLIP_FOR_BLUE):
        super().__init__()
        self.device = torch.device('cpu')
        # TODO: Choose a decent network architecture
        self.network = torch.nn.Linear(12, 3)
        self.flip_for_blue = flip_for_blue

    def forward(self, input_tensor: Tensor):
        input_tensor = torch.atleast_2d(input_tensor.to(self.device))

        # This assumes the first component of each input tensor is the team id
        # If you violate that assumption, you will need to change this
        is_blue = input_tensor[:, 0] == 1
        if self.flip_for_blue:
            print('Before', input_tensor)
            input_tensor = flip_input_for_blue(input_tensor, is_blue)
            print('After', input_tensor)

        # TODO: Build a decent model
        network_output: Tensor = self.network(input_tensor)
        output_tensor = network_output

        if self.flip_for_blue:
            output_tensor = flip_output_for_blue(output_tensor, is_blue)

        return torch.squeeze(output_tensor)


def flip_input_for_blue(input_tensor: Tensor, is_blue: Tensor) -> Tensor:
    should_flip_feature = torch.tensor([
        False,  # team_id
        False,  # kart_center[0]
        True,   # kart_center[1]
        True,   # kart_angle
        True,   # kart_to_puck_angle
        False,  # goal_line_center[0]
        True,   # goal_line_center[1]
        True,   # kart_to_puck_angle_difference
        False,  # puck_center[0]
        True,   # puck_center[1]
        True,   # puck_to_goal_line_angle
        False,  # puck_to_goal_line_dist
    ])
    flip_multiple = torch.where(should_flip_feature, -1., 1.)
    return torch.where(is_blue[:, None], input_tensor * flip_multiple[None, :], input_tensor)


def flip_output_for_blue(output_tensor: Tensor, is_blue: Tensor) -> Tensor:
    should_flip_feature = torch.tensor([
        False,  # acceleration
        True,   # steer
        False,  # brake
    ])
    flip_multiple = torch.where(should_flip_feature, -1., 1.)
    return torch.where(is_blue[:, None], output_tensor * flip_multiple[None, :], output_tensor)
