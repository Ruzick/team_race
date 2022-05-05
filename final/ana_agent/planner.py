from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from custom import dense_transforms
from custom.model_det import Detector, load_model
from torch import nn


from image_agent.detections import DetectionType, TeamDetections


default_pucks = [np.array([0, 1, 0]), np.array([0, 1, 0])]


class ImageModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.detector: Detector = load_model().to(device)
        self.last_pucks = default_pucks
        # self.memory = Memory()
        # What frame of the current match
        self.i_frame = 0

    def forward(self, team_id: int, team_state: List[Dict[str, Any]], team_images: List[np.ndarray]
                ) -> List[Dict[str, Any]]:
        '''
        Given the state of all players of the team and the viewpoint images of
        each player, generates the action to play.
        '''
        team_detections = TeamDetections()

        for player_image in team_images:
            image_tensor = dense_transforms.ToTensor()(player_image)[
                0].to(self.device)
            player_detections = self.detector.detect(image_tensor)
            team_detections.add_player_detections(player_detections)



        return [team_detections.get_all_detections(0, DetectionType.PUCK), team_detections.get_all_detections(1, DetectionType.PUCK)]

