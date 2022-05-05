from matplotlib import transforms
import numpy as np
import random 
import torch
from custom._model_det import load_model
from custom.dense_transforms import CenterToHeatmap, ToTensor, Compose


#*********************************************************************************************************
#A d v i s o r y:
#
# T h i s  C o n t r o l l e r  i s  b e i n g  a d d e d  f o r  d o c u m e n t a t i o n  p u r p o s e s  o n l y.
# 
#D o  n o t  p l a c e  b e t s. (It only runs with a modified runner.py version)
#***********************************************************************************************************


Goals = [[0, 0.07000000029802322, -64.5], [0, 0.07000000029802322, 64.5]]

def limit_period(angle):
    # turn angle into -1 to 1 
    return angle - torch.floor(angle / 2 + 0.5) * 2 

class Team:
    agent_type = 'image'


    acceleration = [0]*2
    steer = [0]*2
    brake = [False]*2

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.model = load_model()
        self.transform=Compose([
        ToTensor(),
        CenterToHeatmap()
    ])

    def new_match(self, team: int, num_players: int) -> list:
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
        self.team, self.num_players = team, num_players
        print('team: ',self.team)
        print('num_pl:',self.num_players)
        car = random.choice(['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'])
        print(car)
        return car* num_players

    def act(self, player_state ,opponent_state, soccer_state, player_image,  ball_location):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        puck_l, kart_l ,goal_l = self.model.detect(self.transform(player_image[0])[0])


        # TODO: Change me. I'm just cruising straight
        self.drift = [False]*2
        self.rescue = [False]*2
        self.fire = [False]*2
        self.nitro = [False]*2

        for i in range(2):
          if i ==0 :
            kart_front = torch.tensor(player_state[i]['kart']['front'], dtype=torch.float32)[[0, 2]]
            kart_center = torch.tensor(player_state[i]['kart']['location'], dtype=torch.float32)[[0, 2]]
            kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
            kart_angle = torch.atan2(kart_direction[1], kart_direction[0])
            #first lets set a perimeter 
            if kart_center[1].item() < -64.5 : #no longer gets stuck at own net
              # #do one turn around in direction of goley own team
              if kart_direction[1].item() >= 0.0  : #if kart starts towards opponents goal but in our net
                self.acceleration[i] = 1.0 
              # else: #turn around if towards our goal
              self.steer[i] = 1
              self.brake[i] = True
            elif kart_center[1].item() > 64.5  : #avoid geting stuck at oponent
              self.steer[i] = -1 #just turn
              self.brake[i] = True
            else:
              # features of soccer 
              puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
              kart_to_puck_direction = (puck_center - kart_center[i]) / torch.norm(puck_center-kart_center[i])
              # features of score-line 
              self.drift[i] = False
              self.rescue[i] = False
              self.fire[i] = True
              self.nitro[i] = False
              proj = np.array(player_state[i]['camera']['projection']).T
              view = np.array(player_state[i]['camera']['view']).T
              v = view @ np.float16(list(ball_location) + [1])
              #special
              p = proj @ view @ np.float16(list(ball_location) + [1])
              aim = np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1) #image coordinates of puck
              aim_from_puck = np.array([(puck_l[1][0] / 400 * 2) - 1, -(puck_l[1][1]/ 300 * 2) + 1])
              proj_compose_view = proj @ view
              puck_global_y = 0.36983
              res = (-1. * proj_compose_view[:, 1] * puck_global_y - proj_compose_view[:, 3])
              mat = np.stack([proj_compose_view[:, 0],proj_compose_view[:, 2],np.array([0., 0., -1., 0.]),np.array([-aim_from_puck[0], -aim_from_puck[1], 0., -1.]),], axis=-1)
              puck_global_x, puck_global_z, _, _ = np.linalg.inv(mat) @ res
              puck = torch.Tensor([puck_global_x, puck_global_y, puck_global_z])
              p2 = proj @ view @ np.float16(list(puck) + [1])
              aim2 = np.clip(np.array([p2[0] / p2[-1], -p2[1] / p2[-1]]), -1, 1) #image coordinates of puck

              x,y = aim2[0], aim2[1]
              if np.linalg.norm(player_state[i]['kart']['velocity']) < 10:
                self.acceleration[i] = 1.0 
              else:
                self.acceleration[i] = 0.3
              if x<0 :
                self.steer[i]=-1
                if abs(puck_center[0]- Goals[0][1]) <20 and abs(puck_center[1] -Goals[0][2]) <20: #avoid self goal
                  self.steer=-1
                  self.brake[i]=True
              elif x>0 :
                self.steer[i]=1
                if abs(puck_center[0] - Goals[0][1]) <30 and abs(puck_center[1] -Goals[0][2]) <30 : #fuss ball it out when close to self goal
                  self.steer[i] = 1
                if abs(puck_center[0] - Goals[0][1]) <20 and abs(puck_center[1] -Goals[0][2]) <20:
                  self.acceleration[i] = 0
                  self.steer[i]=1
                  self.brake[i]=True
              else: #puck behind
                self.steer[i] = 1
                self.brake[i] = True
          if i ==1:  #moves in center line straigh line
            proj = np.array(player_state[i]['camera']['projection']).T
            view = np.array(player_state[i]['camera']['view']).T
            self.steer[i]=0
            self.nitro[i]= True
            self.fire[i]=True
              
            if np.linalg.norm(player_state[i]['kart']['velocity']) < 15:
              self.acceleration[i] = 1.0 
            else:
              self.acceleration[i] = 0.2
            kart_front = torch.tensor(player_state[i]['kart']['front'], dtype=torch.float32)[[0, 2]]
            kart_center = torch.tensor(player_state[i]['kart']['location'], dtype=torch.float32)[[0, 2]]
            kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    
            if kart_center[1].item() < -60 : #no longer gets stuck at own net
            # #do one turn around in direction of goley own team
              if kart_direction[1].item() > 0.0  : #if kart starts towards opponents goal but in our net
                self.acceleration[i] = 1.0 
              else: #turn around if towards our goal
                self.acceleration[i] = 0
                self.brake[i] = True
            elif kart_center[1].item() > 62  : #avoid geting stuck at oponent
              self.acceleration[i] = 0 #just turn
              self.brake[i] = True
            else:
              self.steer[i]=0

        return [{'fire':self.fire[0],'drift':self.drift[0], 'nitro':self.nitro[0], 'acceleration':self.acceleration[0], 'steer':self.steer[0], 'brake':self.brake[0], 'rescue':self.rescue[0]}, {'drift':self.drift[1], 'nitro':self.nitro[1],'acceleration':self.acceleration[1], 'steer':self.steer[1], 'brake':self.brake[1],'rescue':self.rescue[1]}]