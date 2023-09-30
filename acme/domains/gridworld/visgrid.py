import numpy as np

from gym.core import Wrapper, ObservationWrapper
from acme.wrappers.gym_wrapper import GymWrapper
from acme.domains.gridworld.gridenv import GridWorldEnv
from acme.wrappers.minigrid_wrapper import VisGridWrapper


class VisgridInfoWrapper(Wrapper):
  def __init__(self, env: GridWorldEnv):
    super().__init__(env)
    self._timestep = 0
    self._current_info = {}

  def reset(self):
    self._current_info = self._modify_info_dict({})
    return self.env.reset()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self._timestep += 1
    truncated = info.get('TimeLimit.truncated', False)
    terminated = done and not truncated
    info = self._modify_info_dict(info, terminated, truncated)
    done = terminated or truncated
    self._current_info = info
    return obs, reward, done, info

  def _modify_info_dict(self, info, terminated=False, truncated=False):
    visgrid_info = self.env.get_current_info()
    info['player_pos'] = visgrid_info['player_pos']
    info['player_x'] = info['player_pos'][0]
    info['player_y'] = info['player_pos'][1]
    info['truncated'] = truncated
    info['terminated'] = terminated
    info['needs_reset'] = truncated  # pfrl needs this flag
    info['TimeLimit.truncated'] = truncated  # acme needs this flag
    info['timestep'] = self._timestep # total number of timesteps in env
    return info
  
  def info2vec(self, info, keys=('player_x', 'player_y')) -> tuple:
    return tuple([int(info[key]) for key in keys])
  
  def get_info(self):
    return self._current_info 


class UnsqueezeObsWrapper(ObservationWrapper):
  def observation(self, observation):
    assert observation.shape == (84, 84), observation.shape
    return np.repeat(observation[:, :, None], repeats=3, axis=-1)


def environment_builder(size=42, max_steps_per_episode=150):
  environment = GridWorldEnv(
    size=size,
    randomize_starts=False,
    random_goal=False,
    max_steps_per_episode=max_steps_per_episode)
  environment = UnsqueezeObsWrapper(environment)
  environment = VisgridInfoWrapper(environment)
  environment = GymWrapper(environment)
  environment = VisGridWrapper(
    environment,
    num_stacked_frames=1,
    action_repeats=1,
    flatten_frame_stack=True,
    grayscaling=False, 
    pooled_frames=1,
    to_float=True,
    goal_conditioned=False,
    task_goal_features=(41, 41),
    n_color_channels=3
  )

  ts0 = environment.reset()
  
  print(f'Obs spec: {environment.observation_spec()}')

  return environment
