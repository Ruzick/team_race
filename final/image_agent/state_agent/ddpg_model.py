from torch import Tensor, nn
import torch


DEFAULT_DEVICE: torch.device = torch.device('cpu')
FLIP_FOR_BLUE: bool = True


class ActorModel(nn.Module):
    def __init__(self,
                 device: torch.device = DEFAULT_DEVICE,
                 flip_for_blue: bool = FLIP_FOR_BLUE,
                 noise_std_dev: float = 0.):
        super().__init__()
        # TODO: Choose a decent network architecture
        self.network = torch.nn.Sequential(
            torch.nn.Linear(13, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 6),
        )
        self.flip_for_blue = flip_for_blue
        self.device = device
        self.noise_std_dev = noise_std_dev
        self.use_noise = False
        self.discretize_action = False

    def forward(self, input_tensor: Tensor):
        original_device = input_tensor.device
        input_tensor = torch.atleast_2d(input_tensor.to(self.device))

        # This assumes the first component of each input tensor is the team id
        # If you violate that assumption, you will need to change this
        is_blue = input_tensor[:, 0] == 1
        if self.flip_for_blue:
            input_tensor = self.flip_input_for_blue(input_tensor, is_blue)

        network_output: Tensor = self.network(input_tensor)
        output_tensor = (torch.sigmoid(network_output)
                         * torch.tensor([1., 2., 1., 1., 2., 1.]).to(self.device)
                         - torch.tensor([0., 1., 0., 0., 1., 0.]).to(self.device))

        if self.training and self.use_noise:
            output_tensor = self.apply_noise(output_tensor)

        if self.flip_for_blue:
            output_tensor = self.flip_output_for_blue(output_tensor, is_blue)

        if not self.training or self.discretize_action:
            output_tensor[:, 0] = torch.floor(output_tensor[:, 0] * 2.999) / 2
            output_tensor[:, 2] = torch.round(output_tensor[:, 2])
            output_tensor[:, 3] = torch.floor(output_tensor[:, 3] * 2.999) / 2
            output_tensor[:, 5] = torch.round(output_tensor[:, 5])

        return torch.unsqueeze(torch.squeeze(output_tensor), -1).to(original_device)

    def apply_noise(self, output_tensor: Tensor) -> Tensor:
        if self.noise_std_dev <= 0.:
            return output_tensor

        noisy_output = (output_tensor + torch.normal(torch.zeros(6),
                                                     self.noise_std_dev * torch.ones(6)
                                                     ).to(self.device))

        noisy_output = torch.maximum(
            noisy_output,
            torch.tensor([0., -1., 0., 0., -1., 0.]).to(noisy_output.device))
        noisy_output = torch.minimum(
            noisy_output,
            torch.tensor([1., 1., 1., 1., 1., 1.]).to(noisy_output.device))
        return noisy_output

    @staticmethod
    def flip_input_for_blue(input_tensor: Tensor, is_blue: Tensor) -> Tensor:
        assert input_tensor.size(-1) == 13, \
            f'Input tensor size {input_tensor.shape} does not match expected size'
        should_flip_feature = torch.zeros(input_tensor.size(-1), dtype=torch.bool)
        should_flip_feature[1] = True  # ball to defence goal angle
        should_flip_feature[3] = True  # ball to attack goal angle
        should_flip_feature[5] = True  # player 1 to ball angle
        should_flip_feature[7] = True  # player 2 to ball angle
        should_flip_feature[9] = True  # opponent 1 to ball angle
        should_flip_feature[11] = True  # opponent 2 to ball angle

        flip_multiple = torch.where(should_flip_feature, -1., 1.).to(input_tensor.device)
        return torch.where(is_blue[:, None], input_tensor * flip_multiple[None, :], input_tensor)

    @staticmethod
    def flip_output_for_blue(output_tensor: Tensor, is_blue: Tensor) -> Tensor:
        assert output_tensor.size(-1) == 6, \
            f'Output tensor size {output_tensor.shape} does not match expected size'

        should_flip_feature = torch.zeros(output_tensor.size(-1), dtype=torch.bool)
        should_flip_feature[1] = True  # player 1 steer
        should_flip_feature[4] = True  # player 2 steer

        flip_multiple = torch.where(should_flip_feature, -1., 1.).to(output_tensor.device)
        return torch.where(is_blue[:, None], output_tensor * flip_multiple[None, :], output_tensor)


class CriticModel(nn.Module):
    def __init__(self,
                 device: torch.device = DEFAULT_DEVICE,
                 flip_for_blue: bool = FLIP_FOR_BLUE):
        super().__init__()
        # TODO: Choose a decent network architecture
        self.network = torch.nn.Sequential(
            torch.nn.Linear(19, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
        self.device = device
        self.flip_for_blue = flip_for_blue

    def forward(self, input_tensor) -> Tensor:
        original_device = input_tensor.device
        input_tensor = torch.atleast_2d(input_tensor.to(self.device))

        # This assumes the first component of each input tensor is the team id
        # If you violate that assumption, you will need to change this
        is_blue = input_tensor[:, 0] == 1
        if self.flip_for_blue:
            input_tensor = self.flip_input_for_blue(input_tensor, is_blue)

        network_output: Tensor = self.network(input_tensor)
        output_tensor = network_output

        return torch.squeeze(output_tensor).to(original_device)

    @staticmethod
    def flip_input_for_blue(input_tensor: Tensor, is_blue: Tensor) -> Tensor:
        assert input_tensor.size(-1) == 19, \
            f'Input tensor size {input_tensor.shape} does not match expected size'
        should_flip_feature = torch.zeros(input_tensor.size(-1), dtype=torch.bool)
        should_flip_feature[1] = True  # ball to defence goal angle
        should_flip_feature[3] = True  # ball to attack goal angle

        flip_multiple = torch.where(should_flip_feature, -1., 1.).to(input_tensor.device)
        return torch.where(is_blue[:, None], input_tensor * flip_multiple[None, :], input_tensor)