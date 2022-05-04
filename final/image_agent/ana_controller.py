from typing import Any, Dict, List, Optional

import numpy as np
import torch

from image_agent.controller import Controller
from image_agent.detections import TeamDetections


class AnaController(Controller):
    def __init__(self, device: torch.device):
        super().__init__()
        # TODO: Add code here
        # Feel free to not use the device

    def act(self,
            team_id: int,
            team_state: List[Dict[str, Any]],
            team_images: List[np.ndarray],
            team_detections: TeamDetections,
            team_puck_global_coords: List[Optional[np.ndarray]],
            *args: Any) -> List[Dict[str, Any]]:
        # Refer to controller.py for documentation about this method.
        # TODO: Implement this
        raise NotImplementedError('AnaController act function is not yet implemented')

    def get_kart_types(self, team: int, num_players: int) -> List[str]:
        # Refer to controller.py for documentation about this method.
        # TODO: Implement this
        raise NotImplementedError('AnaController get_kart_types function is not yet implemented')
