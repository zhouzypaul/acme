import math

import ipdb
import pickle
import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium.core import Wrapper, ObservationWrapper, ActionWrapper
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, ReseedWrapper, PositionBonus

from gymnasium import spaces
from acme.wrappers.gymnasium_wrapper import GymnasiumWrapper
from acme.wrappers.oar_goal import ObservationActionRewardGoalWrapper
from acme.wrappers.observation_action_reward import ObservationActionRewardWrapper
from acme.wrappers.atari_wrapper import AtariWrapper


class MinigridInfoWrapper(Wrapper):
  """Include extra information in the info dict for debugging/visualizations."""

  def __init__(self, env):
    super().__init__(env)
    self._timestep = 0

    # Store the test-time start state when the environment is constructed
    self.official_start_obs, self.official_start_info = self.reset()

  def reset(self):
    obs, info = self.env.reset()
    info = self._modify_info_dict(info)
    return obs, info

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    self._timestep += 1
    info = self._modify_info_dict(info, terminated, truncated)
    done = terminated or truncated
    return obs, reward, done, info

  def _modify_info_dict(self, info, terminated=False, truncated=False):
    info['player_pos'] = tuple(self.env.agent_pos)
    info['player_x'] = self.env.agent_pos[0]
    info['player_y'] = self.env.agent_pos[1]
    info['truncated'] = truncated
    info['terminated'] = terminated
    info['needs_reset'] = truncated  # pfrl needs this flag
    info['timestep'] = self._timestep # total number of timesteps in env
    info['has_key'] = self.env.unwrapped.carrying is not None
    if info['has_key']:
      assert self.unwrapped.carrying.type == 'key', self.env.unwrapped.carrying
    info['door_open'] = determine_is_door_open(self)
    return info


class ResizeObsWrapper(ObservationWrapper):
  """Resize the observation image to be (84, 84) and compatible with Atari."""
  def observation(self, observation):
    processed_pixels = observation.astype(np.uint8, copy=False)
    if processed_pixels.shape[:2] != (84, 84):
      processed_pixels = Image.fromarray(processed_pixels).resize(
          (84, 84), Image.Resampling.BILINEAR)
      processed_pixels = np.array(processed_pixels, dtype=np.uint8)
    return processed_pixels
  
  @property
  def observation_space(self):
    new_shape = (84, 84, 3)
    return spaces.Box(
      low=np.zeros(new_shape, dtype=np.uint8),
      high=255*np.ones(new_shape, dtype=np.uint8),
      shape=new_shape,
      dtype=np.uint8
    )


class TransposeObsWrapper(ObservationWrapper):
  def observation(self, observation):
    assert len(observation.shape) == 3, observation.shape
    assert observation.shape[-1] == 3, observation.shape
    return observation.transpose((2, 0, 1))


class SparseRewardWrapper(Wrapper):
  """Return a reward of 1 when you reach the goal and 0 otherwise."""
  def step(self, action):
    # minigrid discounts the reward with a step count - undo that here
    obs, reward, terminated, truncated, info = self.env.step(action)
    return obs, float(reward > 0), terminated, truncated, info


class GrayscaleWrapper(ObservationWrapper):
  def observation(self, observation):
    processed_pixels = np.tensordot(
      observation,
      [0.299, 0.587, 1 - (0.299 + 0.587)],
      (-1, 0)
    )
    return processed_pixels
  
class ScaledStateBonus(PositionBonus):
  """Slight mod of StateBonus: scale the count-based bonus before adding."""

  def __init__(self, env, reward_scale):
    super().__init__(env)
    self.reward_scale = reward_scale

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)

    # Tuple based on which we index the counts
    # We use the position after an update
    env = self.unwrapped
    tup = tuple(env.agent_pos)

    # Get the count for this key
    pre_count = 0
    if tup in self.counts:
        pre_count = self.counts[tup]

    # Update the count for this key
    new_count = pre_count + 1
    self.counts[tup] = new_count

    bonus = 1 / math.sqrt(new_count)
    reward += (self.reward_scale * bonus)

    # Add to the info dict
    info['count'] = new_count
    info['bonus'] = bonus

    return obs, reward, terminated, truncated, info
  

class RandomStartWrapper(Wrapper):
  def __init__(self, env, start_loc_file='start_locations_4rooms.pkl'):
    """Randomly samples the starting location for the agent. We have to use the
    ReseedWrapper() because otherwiswe the layout can change between episodes.
    But when we use that wrapper, it also makes random init selection impossible.
    As a hack, I stored some randomly generated (non-collision) locations to a
    file and that is the one we load here.
    """
    super().__init__(env)
    self.n_episodes = 0
    self.start_locations = pickle.load(open(start_loc_file, 'rb'))

    # TODO(ab): This assumes that the 2nd-to-last action is unused in the env
    # Not using the last action because that terminates the episode!
    self.no_op_action = env.action_space.n - 2

  def reset(self):
    super().reset()
    rand_pos = self.start_locations[self.n_episodes % len(self.start_locations)]
    self.n_episodes += 1
    return self.reset_to(rand_pos)

  def reset_to(self, rand_pos):
    new_pos = self.env.place_agent(
      top=rand_pos,
      size=(3, 3)
    )

    # Apply the no-op to get the observation image
    obs, _, _, info = self.env.step(self.no_op_action)

    info['player_x'] = new_pos[0]
    info['player_y'] = new_pos[1]
    info['player_pos'] = new_pos
    
    return obs, info


def determine_goal_pos(env):
  """Convinence hacky function to determine the goal location."""
  from minigrid.core.world_object import Goal
  for i in range(env.grid.width):
    for j in range(env.grid.height):
      tile = env.grid.get(i, j)
      if isinstance(tile, Goal):
          return i, j
      

def determine_is_door_open(env):
  """Convinence hacky function to determine the goal location."""
  from minigrid.core.world_object import Door
  for i in range(env.grid.width):
    for j in range(env.grid.height):
      tile = env.grid.get(i, j)
      if isinstance(tile, Door):
        return tile.is_open


def info2goals(info):
  return np.array([info['player_x'], info['player_y']])


def environment_builder(
  level_name='MiniGrid-Empty-16x16-v0',
  reward_fn='sparse',
  add_count_based_bonus=True,
  exploration_reward_scale=0,
  seed=42,
  random_reset=False,
  max_steps=None,
):
  if max_steps is not None and max_steps > 0:
    env = gym.make(level_name, max_steps=max_steps)  #, goal_pos=(11, 11))
  else:
    env = gym.make(level_name)
  env = ReseedWrapper(env, seeds=[seed])  # To fix the start-goal config
  env = RGBImgObsWrapper(env) # Get pixel observations

  env = ImgObsWrapper(env) # Get rid of the 'mission' field
  if reward_fn == 'sparse':
    env = SparseRewardWrapper(env)

  # env = ResizeObsWrapper(env)
  # env = TransposeObsWrapper(env)
  # env = GrayscaleWrapper(env)
  
  if add_count_based_bonus:
    env = ScaledStateBonus(env, exploration_reward_scale)
  env = MinigridInfoWrapper(env)
  if random_reset:
    assert exploration_reward_scale == 0, exploration_reward_scale
    env = RandomStartWrapper(env)
  
  # Convert the gym environment to a dm_env
  env = GymnasiumWrapper(env)
  
  env = AtariWrapper(
    env,
    num_stacked_frames=1,
    action_repeats=1,
    flatten_frame_stack=True,
    grayscaling=False, 
    pooled_frames=1,
    to_float=True,
  )
  
  # Use the OARG Wrapper
  # env = ObservationActionRewardGoalWrapper(env, info2goals, n_goal_dims=2)
  env = ObservationActionRewardWrapper(env)
  return env
