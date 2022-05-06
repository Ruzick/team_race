from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from .custom import dense_transforms
from .custom.model_det import Detector, load_model
from torch import nn

from image_agent.controller import Controller
from image_agent.detections import DetectionType, TeamDetections

# Contains the number of frames for which to ignore the image after a goal.
# During this time, the puck is assumed to be at [0., 0., 0.]. This helps
# because the puck location is incorrectly calculated for some time after a goal.
IGNORE_IMAGE_AFTER_GOAL_FRAMES = 17

default_pucks = [np.array([0, 1, 0]), np.array([0, 1, 0])]


class ImageModel(nn.Module):
    def __init__(self, device: torch.device, controller: Controller):
        super().__init__()
        self.device = device
        self.detector: Detector = load_model().to(device)
        self.controller = controller
        self.last_pucks = default_pucks
        self.last_avg_velocities: List[Tuple[float, float, float]] = [
            (0., 0., 0.)
        ] * (IGNORE_IMAGE_AFTER_GOAL_FRAMES + 3)
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

        team_puck_global_coords = get_team_puck_global_coords(
            team_state, team_detections, self.last_avg_velocities)

        team_puck_global_coords = team_last_known(
            team_puck_global_coords, self.last_pucks)
        self.last_pucks = team_puck_global_coords

        actions = self.controller.act(
            team_id, team_state, team_images, team_detections,
            team_puck_global_coords, self.i_frame)

        self.i_frame += 1
        avg_velocities: List[np.ndarray] = np.stack([
            np.array(player_state['kart']['velocity'])
            for player_state in team_state
        ], axis=0).mean(0)
        self.last_avg_velocities = self.last_avg_velocities[1:] + [avg_velocities]

        return actions


def players_last_known(puck_coords, last_known):
    '''Each player updates NONE with their last known good detect'''
    return [lk if pc is None else pc for pc, lk in zip(puck_coords, last_known)]


def team_last_known(puck_coords, last_known):
    '''Player gets other's coord when NONE (else last known good)'''

    if puck_coords[0] is None and puck_coords[1] is None:
        # return [last_known[1], last_known[0]]
        return last_known

    return [puck_coords[(i+1) % 2] if pc is None else pc for i, pc in enumerate(puck_coords)]


def direction_vector(puck_coords, last_known):
    '''[WIP] Predict next point from last couple points'''
    filtered_coords = []

    filtered = team_last_known(puck_coords, last_known)
    for c, p1, p0 in zip(puck_coords, filtered, last_known):
        if c is None:
            m = np.nan_to_num((p1-p0) / (np.linalg.norm(p1-p0)+0.001))
            # print(p0)
            # print(p1)
            # print(p1-p0)
            # print(np.linalg.norm(p1-p0))
            # filtered_coords.append(p1 + 4*m)
            filtered_coords.append(p1 + 15*m)
        else:
            filtered_coords.append(p1)

    return filtered_coords


def carrot_on_a_stick(puck_coords, team_state):
    '''[WIP] Make the kart think the puck is in front of them to the right/left in order to turn'''
    # print(puck_coords)
    if puck_coords[0] is None and puck_coords[1] is None:
        # place point 30deg to front left
        front = np.array(team_state[0]['kart']['front'])
        center = np.array(team_state[0]['kart']['location'])
        mag = np.nan_to_num(np.linalg.norm(front - center))
        m = (front + (front - center)/mag) + 2*np.array([1, 0, 1])
        left90_1 = np.flip(m)
        left90_1[0] = left90_1[0]*-1

        front = np.array(team_state[1]['kart']['front'])
        center = np.array(team_state[1]['kart']['location'])
        mag = np.linalg.norm(front - center)
        m = (front + (front - center)/mag) + 2*np.array([1, 0, 1])
        left90_2 = np.flip(m)
        left90_2[2] = left90_2[2]*-1

        return [left90_1, left90_2]

    # else, share information
    _coords = []
    if puck_coords[0] is None:
        _coords.append(puck_coords[1])
    else:
        _coords.append(puck_coords[0])

    if puck_coords[1] is None:
        _coords.append(puck_coords[0])
    else:
        _coords.append(puck_coords[1])

    return _coords


def get_team_puck_global_coords(team_state: List[Dict[str, Any]],
                                team_detections: TeamDetections,
                                last_avg_velocities: List[np.ndarray]
                                ) -> List[Optional[np.ndarray]]:
    # If players were not moving recently, assume a goal was scored recently.
    # Assume the puck is in the middle of the field.
    if were_players_still(last_avg_velocities):
        return [
            np.array([0., 0., 0.]),
            np.array([0., 0., 0.])
        ]

    player_puck_coords: List[Optional[np.ndarray]] = []
    for i_player in range(len(team_detections.detections)):
        if not team_detections.did_player_see_puck[i_player]:
            player_puck_coords.append(None)
            continue

        player_image_puck_coords = team_detections.get_all_detections(
            i_player, DetectionType.PUCK)[0]
        player_global_puck_coords = compute_puck_global_coords(
            player_image_puck_coords[0],
            player_image_puck_coords[1],
            team_state[i_player])

        player_puck_coords.append(np.array(player_global_puck_coords))

    return player_puck_coords


def were_players_still(last_avg_velocities: List[np.ndarray]) -> bool:
    consecutive_still_frames = 0
    for avg_velocity in last_avg_velocities:
        if consecutive_still_frames >= 3:
            break

        if np.count_nonzero(avg_velocity) > 0:
            consecutive_still_frames = 0
        else:
            consecutive_still_frames += 1

    return consecutive_still_frames >= 3


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
