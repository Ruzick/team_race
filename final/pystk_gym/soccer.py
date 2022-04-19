import numpy as np
import gym
from gym import spaces
from gym.core import ObsType, ActType
from gym.utils import seeding
from utils import StateRecorder, VideoRecorder, AIRunner, to_native
from typing import Optional


class STKSoccer(gym.Env):
    """
    # Description

    This environment is a wrapper for PySuperTuxKart's Soccer mode
    https://pystk.readthedocs.io/en/latest/race.html


    # Action Space

    The action space is a combination of Discrete and Continous.
    `step` accepts a `dict` containing any of the following keys.

    +-----+----------------+-------+-------------------------------------------------------------------+
    | Num |     Action     | Range |                               Note                                |
    +-----+----------------+-------+-------------------------------------------------------------------+
    |   0 | "acceleration" | 0..1  |                                                                   |
    |   1 | "brake"        | bool  | Brake will reverse if you do not accelerate (good for backing up) |
    |   2 | "drift"        | bool  | (Optional. unless you want to turn faster)                        |
    |   3 | "fire"         | bool  | (Optional. you can hit the puck with a projectile)                |
    |   4 | "nitro"        | bool  | (Optional.)                                                       |
    |   5 | "rescue"       | bool  | (Optional. no clue where you will end up though.)                 |
    |   6 | "steer"        | -1..1 |                                                                   |
    +-----+----------------+-------+-------------------------------------------------------------------+


    # Observation Space
    ...

    # Rewards
    ...


    """

    metadata = {'render.modes': ['human']}
    reward_range = (-float("inf"), float("inf"))

    # Actions
    action_space: spaces.Space[ActType]
    observation_space: spaces.Space[ObsType]

    # Constants
    TEAM_RED = 0
    TEAM_BLUE = 1

    def __init__(self, use_graphics: bool = True) -> None:
        super().__init__()

        # Object vars
        self.n_teams = 2
        self.players_per_team = 2
        self.active_players = self.n_teams * self.players_per_team
        self.track_name = 'icy_soccer_field'

        # Gym vars
        # Todo:  Standardize to rest of gym envs
        self.n_actions = {
            "discrete": 5,
            "continuous": 2
        }
        # self.action_space = spaces.Tuple(
        #     spaces.Discrete(self.n_actions["discrete"]),
        #     spaces.Box(low=np.array([0.0, -1.0]),
        #                high=np.array([1.0, 1.0]))
        # )
        # self.observation_space = gym.spaces

        # Generators
        # Todo:  Randomize ball
        self.get_ball_position = lambda: (0, 1, 0)
        self.get_ball_velocity = lambda: (0, 0, 0)

        # Setup PySTK
        import pystk
        self._pystk = pystk
        self.match = None
        self.state = None

        self._use_graphics = use_graphics
        if use_graphics:
            graphics_config = self._pystk.GraphicsConfig.hd()
            graphics_config.screen_width = 400
            graphics_config.screen_height = 300
        else:
            graphics_config = self._pystk.GraphicsConfig.none()

        self._pystk.init(graphics_config)

        # Renderer
        self.recorder = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = {}):
        """ Resets the environment to an initial state and returns an initial
        observation.

        `options` of form:
            options = {"teams": [[{"is_ai": False, "kart": "tux"}, ...], [...], ...}}

        """
        # Todo:  Backfill teams with AI if not specified?
        opt_n_teams = len(options["teams"])
        if opt_n_teams != self.n_teams:
            raise f"Expected {self.n_teams} teams. Got: {opt_n_teams}"

        for i, team in enumerate(options["teams"]):
            if len(team) != self.players_per_team:
                raise f"Expected {self.players_per_team} players per team. Found {len(team)} in team {i}"

        # Stop currently running match if there is one
        if self.match is not None:
            self.match.stop()
            del self.match

        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        RaceConfig = self._pystk.RaceConfig
        self.race_config = RaceConfig(
            track=self.track_name, mode=RaceConfig.RaceMode.SOCCER,
            num_kart=self.n_teams * self.players_per_team)
        self.race_config.players.pop()

        # Add players
        for i, team in enumerate(options["teams"]):
            for player in team:
                player_conf = self._config_player(
                    i, player["is_ai"], player["kart"])
                self.race_config.players.append(player_conf)

        # start match
        self.match = self._pystk.Race(self.race_config)
        self.match.start()
        self.match.step()

        self.state = self._pystk.WorldState()
        self.state.update()
        self.state.set_ball_location(
            self.get_ball_position(),
            self.get_ball_velocity()
        )

        return self._next_observation()

    def step(self, action: ActType):
        """ Run one timestep of the environment's dynamics.
        """
        if len(action) != (self.active_players):
            raise f"Expected `action` to be of same length as number of players ({self.active_players})"

        # Take actions
        _pystk_actions = [self._pystk.Action(**a) for a in action]
        self.match.step(_pystk_actions)

        # Return new state observations
        return self._next_observation()

    def render(self, mode: str = 'human'):
        """ Renders the environment.

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        """
        if not self.recorder:
            self._init_renderer()

        # Todo:  Implement "image" and "state" agent variations
        if mode == "rgb_array":
            pass

        # Todo:  Implement a "steaming" sink for live updates
        if self.recorder is not None:
            team1_state = [to_native(p)
                           for p in self.state.players[0::2]]
            team2_state = [to_native(p)
                           for p in self.state.players[1::2]]
            soccer_state = to_native(self.state.soccer)

            team1_images = [np.array(self.match.render_data[i].image)
                            for i in range(0, len(self.match.render_data), 2)]
            team2_images = [np.array(self.match.render_data[i].image)
                            for i in range(1, len(self.match.render_data), 2)]
            self.recorder(team1_state, team2_state, soccer_state,
                          actions=None, team1_images=team1_images, team2_images=team2_images)

    def close(self):
        """ Stop match and clean up PySTK.
        """
        if self.match is not None:
            self.match.stop()
            del self.match
        if self.recorder is not None:
            del self.recorder
        self._pystk.clean()

    def _config_player(self, team_id: int, is_ai: bool = False, kart: str = "tux"):
        PlayerConfig = self._pystk.PlayerConfig
        controller = PlayerConfig.Controller.AI_CONTROL if is_ai else PlayerConfig.Controller.PLAYER_CONTROL
        return PlayerConfig(controller=controller, team=team_id, kart=kart)

    def _init_renderer(self, video: str = "my_gym_video.mp4", state: str = ""):
        self.recorder = None

        if video:
            self.recorder = self.recorder & VideoRecorder(video)

        if state:
            self.recorder = self.recorder & StateRecorder(state)

    def _next_observation(self):
        # Todo:  Option to vectorize state
        self.state.update()

        exposed_state = {}

        # State of world
        exposed_state["soccer"] = to_native(self.state.soccer)

        # Red team status
        exposed_state["team1"] = [to_native(p)
                                  for p in self.state.players[0::2]]
        # Blue team status
        exposed_state["team2"] = [to_native(p)
                                  for p in self.state.players[1::2]],

        return exposed_state


if __name__ == "__main__":
    from tqdm import tqdm

    team1 = AIRunner()
    team2 = AIRunner()

    options = {"teams": [
        [{"is_ai": True, "kart": "tux"}, {"is_ai": True, "kart": "tux"}],
        [{"is_ai": True, "kart": "tux"}, {"is_ai": True, "kart": "tux"}],
    ]}

    with STKSoccer(use_graphics=True) as env:
        observation = env.reset(options=options)

        for _ in (pbar := tqdm(range(1000))):

            team1_state = observation.pop("team1")
            team2_state = observation.pop("team2")
            soccer_state = observation.pop("soccer")
            pbar.set_description(
                "Score:" + ",".join([str(i) for i in soccer_state["score"]]))

            team1_actions = team1.act(team1_state, team2_state, soccer_state)
            team2_actions = team2.act(team1_state, team2_state, soccer_state)

            # Assemble the actions
            actions = []
            num_player = 2
            for i in range(num_player):
                a1 = team1_actions[i] if team1_actions is not None and i < len(
                    team1_actions) else {}
                a2 = team2_actions[i] if team2_actions is not None and i < len(
                    team2_actions) else {}
                actions.append(a1)
                actions.append(a2)

            observation = env.step(actions)
            env.render()
