from typing import Tuple

import torch
from torch import Tensor, nn

from state_agent.utils import load_model

DEFAULT_DEVICE: torch.device = torch.device('cpu')


def limit_period(angle: float):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2


class HumanModel(nn.Module):
    def __init__(self,
                 device: torch.device = DEFAULT_DEVICE):
        super().__init__()
        self.device = device
        self.jurgen_model = load_model('jurgen_agent.pt')

    def forward(self, input_tensor) -> Tensor:
        original_device = input_tensor.device
        input_tensor = input_tensor.to(self.device)

        output_tensor: Tensor = torch.tensor([
            *self.get_player_action(input_tensor, 0),
            *self.get_player_action(input_tensor, 1),
        ], dtype=torch.float32)

        output_tensor = self.clip_output(output_tensor)
        return torch.squeeze(output_tensor).to(original_device)

    def get_player_action(self, input_tensor, i_player: int) -> Tuple[float, float, float]:
        jurgen_input = input_tensor[11 * i_player: 11 * (i_player + 1)]
        human_input = input_tensor[22:]

        jurgen_output = self.jurgen_model(jurgen_input)
        if jurgen_output[2] == 0:
            return (float(jurgen_output[0].item()),
                    float(jurgen_output[1].item()),
                    float(jurgen_output[2].item()))

        kart_velocity_angle_index = 5 + 4 * i_player
        kart_signed_speed_index = kart_velocity_angle_index + 1
        kart_signed_speed = float(human_input[kart_signed_speed_index])
        kart_to_ball_angle_index = kart_signed_speed_index + 1
        kart_to_ball_relative_angle = limit_period(float(
            human_input[kart_velocity_angle_index] - human_input[kart_to_ball_angle_index]
        ) / torch.pi)

        # is_blue = float(input_tensor[0]) != 0
        # ball_to_goal_angle_multiplier = -1. if is_blue else 1.
        # ball_to_goal_angle = ball_to_goal_angle_multiplier * float(input_tensor[3]) != 0

        steering_multiplier = 10000.

        steer = steering_multiplier * kart_to_ball_relative_angle
        return 1., steer, 0.

    @staticmethod
    def clip_output(output_tensor: Tensor) -> Tensor:
        upper_bound = torch.tensor(
            [1., 1., 1., 1., 1., 1.], dtype=torch.float32
        ).to(output_tensor.device)
        lower_bound = torch.tensor(
            [0., -1., 0., 0., -1., 0.], dtype=torch.float32
        ).to(output_tensor.device)

        return output_tensor.maximum(lower_bound).minimum(upper_bound)
