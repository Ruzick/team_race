import numpy as np
import torch
from typing import Any, Dict, List
from ana_agent.planner import ImageModel

class Team:
    agent_type = 'image'

    acceleration = [0]*2
    steer = [0]*2
    brake = [False]*2
    drift=[False]*2

    True_neg = 0
    True_pos= 0 
    False_neg = 0
    False_pos = 0

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = ImageModel(device).to(device)
        self.frame=0
        self.last_location = [0, 0]
        self.last_kart1_velocities = [[0, 0, 0]]*3
        self.last_kart2_velocities = [[0, 0, 0]]*3

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

        if self.team == 0:
          self.Goal1 = [0, 0.07000000029802322, 64.5]
          self.Goal2= [0, 0.07000000029802322, -64.5]
          self.id = 1
        else: 
          self.Goal1 = [0, 0.07000000029802322, -64.5]
          self.Goal2= [0, 0.07000000029802322, 64.5]
          self.id = -1


        return ['xue'] * num_players
        

    def act(self, player_state, player_image):
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

        puck_location = self.model.forward(self.team, player_state, player_image)

        self.last_kart1_velocities.append(player_state[0]['kart']['velocity'])
        self.last_kart1_velocities.pop(0)
        self.last_kart2_velocities.append(player_state[1]['kart']['velocity'])
        self.last_kart2_velocities.pop(0)

        if  np.linalg.norm(self.last_kart1_velocities[0]) == 0 and np.linalg.norm(self.last_kart1_velocities[1]) == 0 and np.linalg.norm(self.last_kart1_velocities[2]) == 0 and np.linalg.norm(self.last_kart1_velocities[0]) == 0 and np.linalg.norm(self.last_kart2_velocities[1])==0 and np.linalg.norm(self.last_kart2_velocities[2])==0:
          self.frame = 0
        else:
          self.frame +=1

       



        for i in range(2):

          proj = np.array(player_state[i]['camera']['projection']).T
          view = np.array(player_state[i]['camera']['view']).T

          kart_location = np.array(player_state[i]['kart']['location']).T
          p = proj @ view @ np.array(list(kart_location) + [1])
          k = np.array([p[0] / p[-1], -p[1] / p[-1]])
 
          if len(puck_location[i]) !=0:

              x_global, y_global = puck_location[i][0]
              x = x_global/400*2 - 1
              y = y_global/300*2 - 1

              self.last_location[i]= np.sign(x)



              if np.abs(x-k[0])<=0.1 and np.abs(y-k[1])<0.25: #if you got the puck and close to the goals 



                if self.team == 0 and abs(kart_location[0]-10) <4 and kart_location[1]>62: #if close to opponent's goal and goal to the right side
                  self.brake[i]=True
                  self.steer[i]=-1  #moves puck to the right  
                  self.drift[i]=True

                elif self.team == 0 and abs(kart_location[0]+10) <4 and kart_location[1]>62:
                  self.brake[i]=True
                  self.steer[i]=1  #moves puck to the left 
                  self.drift[i]=True

                elif self.team == 0 and kart_location[0]>0 and kart_location[1]<-62:
                  self.brake[i]=True
                  self.steer[i]=-1  #moves puck to the right  
                  self.drift[i]=True

                elif self.team == 0 and kart_location[0]<0 and kart_location[1]<-62:
                  self.brake[i]=True
                  self.steer[i]=1  #moves puck to the left 
                  self.drift[i]=True
                
                elif self.team == 1 and abs(kart_location[0]-10) <4 and kart_location[1] < -62: #if close to opponent's goal and goal to the right side
                  self.brake[i]=True
                  self.steer[i]=1  #moves puck to the right  
                  self.drift[i]=True

                elif self.team == 1 and abs(kart_location[0]+10) <4 and kart_location[1] <-62:
                  self.brake[i]=True
                  self.steer[i]=-1  #moves puck to the left 
                  self.drift[i]=True

                elif self.team == 1 and kart_location[0]>0 and kart_location[1] > 62:
                  self.brake[i]=True
                  self.steer[i]=1  #moves puck to the right  
                  self.drift[i]=True

                elif self.team == 1 and kart_location[0]<0 and kart_location[1] > 62:
                  self.brake[i]=True
                  self.steer[i]=-1  #moves puck to the left 
                  self.drift[i]=True


                else:
                  if np.linalg.norm(player_state[i]['kart']['velocity']) < 18:
                    self.acceleration[i] = 1.0 
                  else:
                    self.acceleration[i] = 0

                  v = view @ np.array(list(self.Goal1) + [1])
                  if np.dot(proj[2:3,0:3],v[0:3].reshape([-1,1])) > 0: 
                    p = proj @ view @ np.array(list(self.Goal1) + [1])
                    goal = np.array([p[0] / p[-1], -p[1] / p[-1]])   #if you got the puck and thr opponent's goal is in view 

                    if goal[0] <0: #if opponent's goal is far away but on the left, turn left gradually 

                      self.steer[i]=1 #turn the puck to the left 
                      self.brake[i]=True

                    elif goal[0] >0: #if goal is on the right, turn right gradually 

                      self.steer[i]=-1
                      self.brake[i]=True

                  
                  else: 
                    v = view @ np.array(list(self.Goal2) + [1])
                    if np.dot(proj[2:3,0:3],v[0:3].reshape([-1,1])) > 0: 
                      p = proj @ view @ np.array(list(self.Goal2) + [1])
                      goal = np.array([p[0] / p[-1], -p[1] / p[-1]])
                      if np.abs(goal[0]) <= 1 and np.abs(goal[1]) <= 1: #if your own goal is i view 

                        if goal[0] <0: #if opponent's goal is far away but on the left, turn left gradually 

                          self.steer[i]=-1 #turn puck to left 
                          self.brake[i]=True

                        elif goal[0] >0: #turn puck to right 

                          self.steer[i]=1
                          self.brake[i]=True       

              elif x>0.33 and y>0.1: #if you got the puck and close to the goal   
                self.acceleration[i] = 0 
                self.brake[i] = True
                self.steer[i] = -1 

              elif x<-0.33 and y>0.1: #if you got the puck and close to the goal   
                self.acceleration[i] = 0 
                self.brake[i] = True
                self.steer[i] = 1 

              elif x<0:
                self.steer[i]=-1
                if np.linalg.norm(player_state[i]['kart']['velocity']) < 18:
                  self.acceleration[i] = 1.0 
                else:
                  self.acceleration[i] = 0

              elif x>0:
                self.steer[i]=1
                if np.linalg.norm(player_state[i]['kart']['velocity']) < 18:
                  self.acceleration[i] = 1.0 
                else:
                  self.acceleration[i] = 0

          else: #if puck not in view turn around

            if self.frame > 30:

              self.acceleration[i] = 0 
              self.brake[i] = True
              self.steer[i] = -1 * self.last_location[i]
              #print(self.last_location[i], self.steer[i])
            else: 
              self.acceleration[i] = 1
              self.steer[i]=0


        return [{'acceleration':self.acceleration[0], 'steer':self.steer[0], 'brake':self.brake[0]}, {'acceleration':self.acceleration[1], 'steer':self.steer[1], 'brake':self.brake[1]}]