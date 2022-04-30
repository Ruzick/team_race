from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from custom import dense_transforms
from custom.model_det import Detector, load_model
from torch import nn

from image_agent.controller import Controller
from image_agent.detections import DetectionType, TeamDetections


class ImageModel(nn.Module):
    def __init__(self, device: torch.device, controller: Controller):
        super().__init__()
        self.device = device
        self.detector: Detector = load_model().to(device)
        self.controller = controller

    def forward(self, team_id: int, team_state: List[Dict[str, Any]], team_images: List[np.ndarray]
                ) -> List[Dict[str, Any]]:
        '''
        Given the state of all players of the team and the viewpoint images of
        each player, generates the action to play.
        '''
        team_detections = TeamDetections()

        for player_image in team_images:
            image_tensor = dense_transforms.ToTensor()(player_image)[0].to(self.device)

            player_detections = self.detector.detect(image_tensor)
            team_detections.add_player_detections(player_detections)

        puck_global_coords = get_puck_global_coords(team_state, team_detections)
        return self.controller.act(
            team_id, team_state, team_images, team_detections, puck_global_coords)


def get_puck_global_coords(team_state: List[Dict[str, Any]],
                           team_detections: TeamDetections
                           ) -> Optional[Tuple[float, float, float]]:

    detected_puck_coords: List[np.ndarray] = []
    for i_player in range(len(team_detections.detections)):
        if not team_detections.did_player_see_puck[i_player]:
            continue

        player_image_puck_coords = team_detections.get_all_detections(
            i_player, DetectionType.PUCK)[0]
        player_global_puck_coords = compute_puck_global_coords(
            player_image_puck_coords[0],
            player_image_puck_coords[1],
            team_state[i_player])

        detected_puck_coords.append(np.array(player_global_puck_coords))

    if len(detected_puck_coords) == 0:
        return None

    return tuple(np.stack(detected_puck_coords, axis=0).mean(axis=0))


def compute_puck_global_coords(x_puck: float, y_puck: float, player_state: Dict[str, Any]
                               ) -> Tuple[float, float, float]:
    '''
    Given the state of a player and the coordinates of the puck as seen from that
    player's perspective, returns the approximate global
    coordinates of the puck.
    '''
    # aim_from_puck = np.array([(y_puck / 400 * 2) - 1, -(x_puck / 300 * 2) + 1])
    aim_from_puck = np.array([(x_puck / 400 * 2) - 1, -(y_puck / 300 * 2) + 1])
    # aim_from_puck = np.array([x_puck / 400, -y_puck / 300])
    proj = np.array(player_state['camera']['projection']).T
    view = np.array(player_state['camera']['view']).T
    proj_compose_view = proj @ view
    # We need to assume that the puck resides in a plane, or else we cannot
    # deduce the global coordinates. Since the field is flat, this plane is
    # the (x, z)-plane elevated by some constant. The below constant appears to
    # be highly accurate barring a few frames at the start of the game.
    puck_global_y = 0.36983
    res = (-1. * proj_compose_view[:, 1] * puck_global_y
           - proj_compose_view[:, 3])
    mat = np.stack([
        proj_compose_view[:, 0],
        proj_compose_view[:, 2],
        np.array([0., 0., -1., 0.]),
        np.array([-aim_from_puck[0], -aim_from_puck[1], 0., -1.]),
    ], axis=-1)
    puck_global_x, puck_global_z, _, _ = np.linalg.inv(mat) @ res
    return (puck_global_x, puck_global_y, puck_global_z)
