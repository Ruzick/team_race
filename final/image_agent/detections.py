import enum
from typing import List, Tuple


class DetectionType(enum.IntEnum):
    PUCK = 0
    PLAYER = 1


class TeamDetections():
    '''
    Utility class for working with object detections
    '''
    def __init__(self):
        # The following is a representation of the detections object
        # player -> detected_object_type -> object_instance -> (x, y)
        self.detections: List[List[List[Tuple[int, int]]]] = []
        # For each player, indicates if they saw the puck
        self.did_player_see_puck: List[bool] = []

    def add_player_detections(self, player_detections: List[List[Tuple[bool, int, int]]]):
        # The following is a representation of a player_detections object
        # detected_object_type -> object_instance -> (was_puck_found, x, y)

        # Remove was_puck_found, since we won't need it
        cleaned_player_detections = [
            [
                (x, y)
                for _, x, y in object_type_detections
            ]
            for object_type_detections in player_detections
        ]

        did_player_see_puck = (len(player_detections) > 0
                               and len(player_detections[DetectionType.PUCK]) > 0)

        self.detections.append(cleaned_player_detections)
        self.did_player_see_puck.append(did_player_see_puck)

    def get_all_detections(self, player_num: int, detection_type: DetectionType
                           ) -> List[Tuple[int, int]]:
        return self.detections[player_num][detection_type]
