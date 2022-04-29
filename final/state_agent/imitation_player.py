from typing import List, Optional
import numpy as np
import torch
from torch.jit import ScriptModule
from .utils import load_model, state_to_tensor

class Team:
    agent_type = 'state'

    def __init__(self, model: Optional[ScriptModule] = None):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team : int =0
        self.num_players: int = 2

        if model is not None:
            self.model: ScriptModule = model
        else:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.model: ScriptModule = load_model().to(device)

    def new_match(self, team: int, num_players: int) -> List[str]:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        karts =  [ 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue']
        self.team, self.num_players = team, num_players
        self.kart = np.random.choice(karts, num_players,replace = False )
        return  list(self.kart)# ['tux'] * num_players

    def act(self, player_state, opponent_state, soccer_state):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             You can ignore the camera here.
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param opponent_state: same as player_state just for other team

        :param soccer_state: dict  Mostly used to obtain the puck location
                             ball:  Puck information
                               - location: float3 world location of the puck

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # TODO: Change me. I'm just cruising straight

        features = state_to_tensor(self.team, player_state, opponent_state,
                                   soccer_state)
        p1_accel, p1_steer, p1_brake,  p2_accel, p2_steer, p2_brake = self.model(features)
        p1_accel, p1_steer, p1_brake,  p2_accel, p2_steer, p2_brake = torch.sigmoid(p1_accel), p1_steer, torch.nn.Threshold(0.8, bool(p1_brake), 0),  torch.sigmoid(p2_accel)*2, p2_steer, torch.nn.Threshold(0.8, bool(p2_brake), 0)
        return  [
          {
                'acceleration': p1_accel,
                'steer': p1_steer,
                'brake': p1_brake,
            },
            {
                'acceleration':p2_accel,
                'steer': p2_steer,
                'brake': p2_brake,
            },
        ]
        # [dict(acceleration=1, steer=0)] * self.num_players