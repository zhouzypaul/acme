import gym
import gym_sokoban
import random
import numpy as np
from PIL import Image
from gym.core import Wrapper
from acme import wrappers
from acme.wrappers.gymnasium_wrapper import GymnasiumWrapper
from acme.wrappers.minigrid_wrapper import MiniGridWrapper as SokobanWrapper
from acme.wrappers.oar_goal import ObservationActionRewardGoalWrapper


class SokobanInfoWrapper(Wrapper):
  def __init__(self, env, seed: int, use_sparse_rewards: bool = False):
    super(SokobanInfoWrapper, self).__init__(env)
    self._timestep = 0   
    self._seed = seed
    self._set_seed(seed)
    self._use_sparse_rewards = use_sparse_rewards

  def step(self, action):
    obs, reward, done, info = self.env.step(int(action))
    terminated = self.env.unwrapped._check_if_all_boxes_on_target()
    truncated = done and not terminated
    info['reward'] = reward
    info = self._modify_info_dict(info, terminated, truncated)
    self._timestep += 1
    rew = float(terminated) if self._use_sparse_rewards else float(reward) / 10.
    return obs, rew, done, info

  def reset(self):
    self._set_seed(self._seed)
    obs = self.env.reset()
    info = {}
    info['reward'] = 0.
    info = self._modify_info_dict(info)
    print(info['state'])
    return obs, info

  def _set_seed(self, seed):
    self.env.seed(seed)
    self.env.unwrapped.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

  def _modify_info_dict(self, info, terminated=False, truncated=False):
    info['player_pos'] = tuple(self.env.unwrapped.player_position)
    info['player_x'], info['player_y'] = info['player_pos']
    state = self.env.unwrapped.room_state.astype(np.int16)
    assert np.where(state == 5)[0][0] == info['player_x']
    assert np.where(state == 5)[1][0] == info['player_y']
    info['state'] = state
    
    if 3 in state:
      info['in_target_box_locations'] = list(zip(*np.where(state == 3)))
    if 4 in state:
      info['out_target_box_locations'] = list(zip(*np.where(state == 4)))
    if 2 in state:
      info['box_target_locations'] = list(zip(*np.where(state == 2)))

    box_indices = np.where((state == 3) | (state == 4))
    info['box_locations'] = list(zip(box_indices[0], box_indices[1]))

    # dict that maps target pos -> current pos (only updated when using the "fixed target" version)
    info['box_mapping'] = self.env.unwrapped.box_mapping

    info['truncated'] = truncated
    info['terminated'] = terminated
    info['timestep'] = self._timestep
    info['needs_reset'] = truncated  # pfrl needs this flag
    info['TimeLimit.truncated'] = truncated  # acme needs this flag
    info['reached'] = terminated

    return info


def info2goals(info):
  # state = info['state'].flatten()  # (n,)
  player_pos = np.asarray(info['player_pos'], dtype=np.int16)  # (2,)
  box_locations = np.asarray(info['box_locations'], dtype=np.int16).flatten()  # (2n,)
  reached = np.asarray([info['reached']], dtype=np.int16)  # (1,)
  goals = np.concatenate([player_pos, box_locations, reached], axis=0)  # (2n+3,)
  return goals


def determine_n_goal_dims(env):
  _, info = env.reset()
  return len(info2goals(info))


def determine_task_goal_features(env):
  env.reset()
  info = env.get_info()
  goal_vector = info2goals(info)
  task_goal = np.zeros_like(goal_vector)
  task_goal[-1] = 1
  return task_goal


def environment_builder(
  level_name='Sokoban-v0',
  seed=42,
  oarg_wrapper=True,
  goal_conditioned=True,
  to_float=False,
  use_sparse_rewards=False
):
  env = gym.make(level_name)
  env = SokobanInfoWrapper(env, seed=seed, use_sparse_rewards=use_sparse_rewards)
  n_goal_dims = determine_n_goal_dims(env)
  
  env = GymnasiumWrapper(env)

  env = SokobanWrapper(
    env,
    num_stacked_frames=1,
    action_repeats=1,
    flatten_frame_stack=True,
    grayscaling=False,
    pooled_frames=1,
    to_float=to_float,
    goal_conditioned=goal_conditioned,
    task_goal_features=determine_task_goal_features(env),
  )

  if oarg_wrapper:
    env = ObservationActionRewardGoalWrapper(env,
                                             info2goals=info2goals,
                                             n_goal_dims=n_goal_dims)
  
  env = wrappers.SinglePrecisionWrapper(env)

  return env

