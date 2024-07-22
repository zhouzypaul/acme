import cv2
import ipdb
import copy
import numpy as np

from gym import spaces
from visgrid.envs import TaxiEnv as InnerTaxiEnv
from visgrid import utils

from acme.wrappers.gymnasium_wrapper import GymnasiumWrapper
from acme.wrappers.observation_action_reward import ObservationActionRewardWrapper


class TaxiEnv(InnerTaxiEnv):
    def __init__(
        self,
        max_episode_steps: int,
        goal_conditioned: bool,
        n_goal_channels: int,
        size: int = 5,
        num_passengers: int = 1
    ):
        self.size = size
        self.num_passengers = num_passengers
        self.goal_conditioned = goal_conditioned
        self._n_goal_channels = n_goal_channels
        
        super().__init__(size=size, n_passengers=num_passengers, exploring_starts=False)
        
        self.game_over = False
        self._timestep = 0
        self._max_episode_steps = max_episode_steps
        self._episode_steps = 0
        self.task_goal_features = determine_task_goal_features(self)

    def reset(self):
        obs, underlying_state = super().reset()
        self.game_over = False
        obs = self._postprocess_obs(obs)
        info = self.get_current_info({})
        print(info)
        self._episode_steps = 0
        return obs, info

    def info2vec(self, info):
        return tuple(info2goals(info))

    def _initialize_obs_space(self):
        n_channels = 3 + self._n_goal_channels if self.goal_conditioned else 3
        img_shape = self.dimensions['img_shape'] + (n_channels, )
        self.img_observation_space = spaces.Box(0, 255, img_shape, dtype=np.uint8)

        factor_obs_shape = self.state_space.nvec
        self.factor_observation_space = spaces.MultiDiscrete(factor_obs_shape, dtype=int)

        assert self.should_render is True, self.should_render
        self.set_rendering(self.should_render) # sets observation to img_observation for us.
   
    def _postprocess_obs(self, obs):
        assert obs.shape == (84, 84, 3) or obs.shape == (128, 128, 3)
        obs = np.where(obs == 1, 0., 1.) # inverts as well so mostly 0s.
        obs = obs * 255
        obs = obs.astype(np.uint8)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action.item())
        self._timestep += 1
        self._episode_steps += 1
        truncated = truncated or (self._episode_steps >= self._max_episode_steps)
        info['TimeLimit.truncated'] = truncated
        info['terminated'] = terminated
        info = self.get_current_info(info)
        obs = self._postprocess_obs(obs)
        done = terminated or truncated
        self.game_over = done
        return obs, float(reward), done, info
    
    def get_current_info(self, info, terminated=False, truncated=False):
        assert len(self.passengers) == 1
        passenger = self.passengers[0]
        active_depot = [depot for depot in self.depots.values() if depot.color == passenger.color]
        assert len(active_depot) == 1, active_depot
        active_depot = active_depot[0]

        taxi_row, taxi_col = self.agent.position
        pass_row, pass_col = passenger.position
        goal_row, goal_col = active_depot.position
        in_taxi = passenger.in_taxi

        to_return = dict(
            player_pos=(taxi_col, taxi_row),
            goal_pos=(goal_col, goal_row),
            passenger_pos=(pass_col, pass_row),
            in_taxi=in_taxi,
            player_x=taxi_col,
            player_y=taxi_row,
            passenger_x=pass_col,
            passenger_y=pass_row,
            destination_x=goal_col,
            destination_y=goal_row,
            truncated=truncated,
            terminated=terminated,
            timestep=self._timestep,
            size=self.size,
            num_passengers=self.num_passengers,
        )

        to_return.update(info)
        return to_return

    def binary2info(self, binary_vector, sparse_info: bool = False):
        """Convert a binary vector back to an info dict for the taxi environment."""
        if isinstance(binary_vector, tuple):
            binary_vector = np.asarray(binary_vector)

        info = {}
        num_grid_locations = self.size * self.size
        
        # Decode taxi position
        taxi_vector = binary_vector[:num_grid_locations]
        if sum(taxi_vector) > 0:
            taxi_pos = np.where(taxi_vector)[0][0]
            info['player_y'], info['player_x'] = divmod(taxi_pos, self.size)
        elif not sparse_info:
            info['player_y'], info['player_x'] = -1, -1
        
        # Decode passenger position
        passenger_vector = binary_vector[num_grid_locations: 2*num_grid_locations]
        if sum(passenger_vector) > 0:
            pass_pos = np.where(passenger_vector)[0][0]
            info['passenger_y'], info['passenger_x'] = divmod(pass_pos, self.size)
        elif not sparse_info:
            info['passenger_y'], info['passenger_x'] = -1, -1
        
        # Decode destination
        dest_vector = binary_vector[2*num_grid_locations:]
        if sum(dest_vector) > 0:
            dest_pos = np.where(dest_vector)[0][0]
            info['destination_y'], info['destination_x'] = divmod(dest_pos, self.size)
        elif not sparse_info:
            info['destination_y'], info['destination_x'] = -1, -1
        
        return info
 
    def _render_objects(self) -> dict:
        walls = self.grid.render(cell_width=self.dimensions['cell_width'],
                                 wall_width=self.dimensions['wall_width'])
        walls = utils.to_rgb(walls, 'almost black')

        depot_patches = np.zeros_like(walls)
        for depot in self.depots.values():
            # Only render depot you're trying to visit
            assert len(self.passengers) == 1
            if depot.color != self.passengers[0].color:
                continue

            patch = self._render_depot_patch(depot.color)
            self._add_patch(depot_patches, patch, depot.position)

        agent_patches = np.zeros_like(walls)
        patch = self._render_character_patch()
        self._add_patch(agent_patches, patch, self.agent.position)

        objects = {
            'walls': walls,
            'depots': depot_patches,
            'agent': agent_patches,
        }
        if self.hidden_goal:
            del objects['depots']

        del objects['agent']

        passenger_patches = np.zeros_like(objects['walls'])
        for p in self.passengers:
            patch = self._render_passenger_patch(p.in_taxi, p.color)
            self._add_patch(passenger_patches, patch, p.position)

        taxi_patches = np.zeros_like(objects['walls'])
        patch = self._render_taxi_patch()
        self._add_patch(taxi_patches, patch, self.agent.position)

        objects.update({
            'taxi': taxi_patches,
            'passengers': passenger_patches,
        })

        return objects


def determine_n_goal_dims(env):
    _, info = env.reset()
    # return len(info2goals(info))
    return len(info2binary(info))


def info2goals(info):
    goals = [
        info["player_x"],
        info["player_y"],
        info["passenger_x"],
        info["passenger_y"],
        info["destination_x"],
        info["destination_y"],
        info["in_taxi"],
        info["terminated"]
    ]
    return np.asarray(goals, dtype=np.int16)


def info2binary(info):
    """Convert the info dict to a binary vector for the taxi environment."""

    def _encode_position(x, y, size):
        return y * size + x

    num_grid_locations = info['size'] * info['size']
    
    binary_vector = np.zeros(3 * num_grid_locations, dtype=bool)
    
    # Encode taxi position (5x5 grid = 25 positions)
    taxi_index = _encode_position(info['player_x'], info['player_y'], info['size'])
    binary_vector[taxi_index] = True
    
    pass_index = _encode_position(info['passenger_x'], info['passenger_y'], info['size'])
    binary_vector[num_grid_locations + pass_index] = True
    
    # Encode destination
    dest_index = _encode_position(info['destination_x'], info['destination_y'], info['size'])
    binary_vector[2 * num_grid_locations + dest_index] = True
    
    return binary_vector


def goals2info(goals: np.ndarray):
    """Deserialize the info dict from a np vector of integers."""
    return {
        "player_x": goals[0],
        "player_y": goals[1],
        "passenger_x": goals[2],
        "passenger_y": goals[3],
        "destination_x": goals[4],
        "destination_y": goals[5],
        "in_taxi": bool(goals[6]),
        "terminated": bool(goals[7]),
    }


def determine_task_goal_features(env):
    _, info = env.reset()
    goals0 = info2goals(info)
    goal_vector = np.zeros_like(goals0)
    goal_vector[-1] = 1
    return goal_vector


def environment_builder(
    seed=42,
    max_steps=200,
    goal_conditioned=True,
    oar_wrapper=True,
    use_learned_goal_classifiers=False,
    n_goal_channels=1,
    grid_size=5
):
    env = TaxiEnv(max_steps,
                  goal_conditioned=goal_conditioned,
                  n_goal_channels=n_goal_channels,
                  size=grid_size)
    env.seed(seed)
    n_goal_dims = determine_n_goal_dims(env)
    env = GymnasiumWrapper(env)
    if oar_wrapper:
        env = ObservationActionRewardWrapper(env)
    return env
