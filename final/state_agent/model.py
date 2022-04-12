from torch import Tensor, nn
import torch


class StateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def forward(self, features: Tensor):
        features = features.to(self.device)
        acceleration = 1.0
        steer = 0.
        brake = float(False)
        return torch.tensor([acceleration, steer, brake])
