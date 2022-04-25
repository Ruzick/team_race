from torch import Tensor, nn
import torch


DEFAULT_DEVICE: torch.device = torch.device('cpu')


def limit_period(angle: float):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2


class HumanModel(nn.Module):
    def __init__(self,
                 device: torch.device = DEFAULT_DEVICE):
        super().__init__()
        self.device = device

    def forward(self, input_tensor) -> Tensor:
        original_device = input_tensor.device
        input_tensor = input_tensor.to(self.device)

        output_tensor: Tensor = torch.tensor([
            0.5, self.get_steering(input_tensor, 0), 0.,
            1., self.get_steering(input_tensor, 1), 0.,
        ], dtype=torch.float32)

        output_tensor = self.clip_output(output_tensor)
        return torch.squeeze(output_tensor).to(original_device)

    @staticmethod
    def get_steering(input_tensor, i_player: int):
        kart_angle_index = 5 + 3 * i_player
        kart_to_ball_angle_index = kart_angle_index + 1
        kart_to_ball_relative_angle = limit_period(float(
            input_tensor[kart_angle_index] - input_tensor[kart_to_ball_angle_index]
        ) / torch.pi)

        # is_blue = float(input_tensor[0]) != 0
        # ball_to_goal_angle_multiplier = -1. if is_blue else 1.
        # ball_to_goal_angle = ball_to_goal_angle_multiplier * float(input_tensor[3]) != 0

        steering_multiplier = 10000.

        steer = steering_multiplier * kart_to_ball_relative_angle
        return steer

    @staticmethod
    def clip_output(output_tensor: Tensor) -> Tensor:
        upper_bound = torch.tensor(
            [1., 1., 1., 1., 1., 1.], dtype=torch.float32
        ).to(output_tensor.device)
        lower_bound = torch.tensor(
            [0., -1., 0., 0., -1., 0.], dtype=torch.float32
        ).to(output_tensor.device)

        return output_tensor.maximum(lower_bound).minimum(upper_bound)
