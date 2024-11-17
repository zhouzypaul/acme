import numpy as np
import gymnasium as gym
from acme.wrappers.gymnasium_wrapper import GymnasiumWrapper
from gymnasium.core import Wrapper, ObservationWrapper, ActionWrapper
from acme.wrappers.oar_goal import ObservationActionRewardGoalWrapper


N_OBS_DIMS = 23
N_GOALS_PER_DIM = 10
N_GOAL_DIMS = N_OBS_DIMS * N_GOALS_PER_DIM

# xy euclidean distance
GOAL_TOLERANCE = 0.1


class PusherInfoWrapper(Wrapper):
  def __init__(self, env, seed, fixed_start_state: bool = True):
    super().__init__(env)
    self._seed = seed
    self._fixed_start_state = fixed_start_state
    
    self._timestep = 0

  def reset(self):
    seed = self._seed if self._fixed_start_state else None
    obs, info = self.env.reset(seed=seed)
    info = self._modify_info_dict(info, obs)
    print(info)
    return obs, info

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    self._timestep += 1
    info = self._modify_info_dict(info, obs, terminated, truncated)
    done = terminated or truncated
    return obs, reward, done, info

  def _extract_end_effector_pos(self, obs):
    return obs[14:17]
  
  def _extract_object_pos(self, obs):
    return obs[17:20]

  def _extract_object_goal(self, obs):
    return obs[20:23]

  def _modify_info_dict(self, info, obs, terminated=False, truncated=False):
    info['timestep'] = self._timestep
    info['end_effector_pos'] = self._extract_end_effector_pos(obs)
    info['player_x'] = info['end_effector_pos'][0]
    info['player_y'] = info['end_effector_pos'][1]
    info['player_z'] = info['end_effector_pos'][2]
    info['object_pos'] = self._extract_object_pos(obs)
    info['object_x'] = info['object_pos'][0]
    info['object_y'] = info['object_pos'][1]
    info['object_z'] = info['object_pos'][2]
    info['object_goal'] = self._extract_object_goal(obs)
    info['object_goal_x'] = info['object_goal'][0]
    info['object_goal_y'] = info['object_goal'][1]
    info['object_goal_z'] = info['object_goal'][2]
    info['terminated'] = terminated
    info['truncated'] = truncated
    info['needs_reset'] = truncated
    info['TimeLimit.truncated'] = truncated
    return info

  
def info2binary(info):
  robot_x = info['player_x']
  robot_y = info['player_y']
  robot_z = info['player_z']
  obj_x = info['object_x']
  obj_y = info['object_y']
  
  binary_vector = np.zeros(N_GOAL_DIMS, dtype=bool)

  def discretize_dim(value, minval=-1.5, maxval=8., num_buckets=10):
    bucket_size = (maxval - minval) / num_buckets
    bucket_index = int((value - minval) / bucket_size)
    return bucket_index

  x_bucket = discretize_dim(robot_x, num_buckets=N_GOALS_PER_DIM)
  y_bucket = discretize_dim(robot_y, num_buckets=N_GOALS_PER_DIM)
  z_bucket = discretize_dim(robot_z, num_buckets=N_GOALS_PER_DIM)
  obj_x_bucket = discretize_dim(obj_x, num_buckets=N_GOALS_PER_DIM)
  obj_y_bucket = discretize_dim(obj_y, num_buckets=N_GOALS_PER_DIM)

  binary_vector[x_bucket] = True
  binary_vector[y_bucket + N_GOALS_PER_DIM] = True
  binary_vector[z_bucket + 2 * N_GOALS_PER_DIM] = True
  binary_vector[obj_x_bucket + 3 * N_GOALS_PER_DIM] = True
  binary_vector[obj_y_bucket + 4 * N_GOALS_PER_DIM] = True

  return binary_vector


def binary2info(binary_vector, sparse_info=False):
  d = {}
  d['player_x'] = binary_vector[:N_GOALS_PER_DIM]
  d['player_y'] = binary_vector[N_GOALS_PER_DIM:2*N_GOALS_PER_DIM]
  d['player_z'] = binary_vector[2*N_GOALS_PER_DIM:3*N_GOALS_PER_DIM]
  d['object_x'] = binary_vector[3*N_GOALS_PER_DIM:4*N_GOALS_PER_DIM]
  d['object_y'] = binary_vector[4*N_GOALS_PER_DIM:5*N_GOALS_PER_DIM]

  info = {}

  def vec2pos(vec, minval=-1.5, maxval=1.5, num_buckets=10):
    bucket = np.where(vec)[0][0]
    bucket_size = (maxval - minval) / num_buckets
    return minval + bucket * bucket_size

  for k in d:
    if d[k].sum() > 0:
      info[k] = vec2pos(d[k])
    elif not sparse_info:
      info[k] = None

  return info


def environment_builder(
  level_name='Pusher-v4',
  seed=42,
  fixed_start_state=True,
  num_goal_classifiers=100,
  use_learned_goal_classifiers=False
):
  env = gym.make(level_name)
  env = PusherInfoWrapper(env, seed, fixed_start_state)

  n_goal_dims = num_goal_classifiers if use_learned_goal_classifiers else N_GOAL_DIMS

  env = GymnasiumWrapper(env)
  
  env = ObservationActionRewardGoalWrapper(
    env,
    info2goals=info2binary,
    n_goal_dims=n_goal_dims,
    use_learned_goal_classifiers=use_learned_goal_classifiers,
  )

  return env
