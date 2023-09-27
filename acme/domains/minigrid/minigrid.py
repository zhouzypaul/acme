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

from acme.wrappers.observation_action_reward import ObservationActionRewardWrapper
from acme.wrappers.minigrid_wrapper import MiniGridWrapper
from minigrid.core.world_object import Lava


# NOTE: we are assuming a max number of keys and doors
N_POS_DIMS = 2  # player (x, y) location
N_KEY_DIMS = 1  # has_key or not
N_DOOR_DIMS = 7 # number of possible doors in the puzzle
N_OBJECT_DIMS = 1  # is object being carried or not
N_LAVA_DIMS = 1  # is the location seek/avoid (lava or not)
N_GOAL_DIMS = N_POS_DIMS + N_KEY_DIMS + N_DOOR_DIMS + N_OBJECT_DIMS + N_LAVA_DIMS


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
    print(self)
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
    info['player_dir'] = self.env.agent_dir
    info['truncated'] = truncated
    info['terminated'] = terminated
    info['needs_reset'] = truncated  # pfrl needs this flag
    info['TimeLimit.truncated'] = truncated  # acme needs this flag
    info['timestep'] = self._timestep # total number of timesteps in env
    
    info['is_lava'] = isinstance(
      self.grid.get(info['player_x'], info['player_y']), Lava)
    
    carrying = self.unwrapped.carrying

    if carrying is not None:
      assert carrying.type in ('key', 'ball'), carrying

    info['has_key'] = carrying is not None and carrying.type == 'key'
    info['has_ball'] = carrying is not None and carrying.type == 'ball'
    
    if info['has_key']:
      assert carrying.type == 'key', self.env.unwrapped.carrying

    door_states = get_all_door_states(self)
    for i, door_state in enumerate(door_states):
      info[f'door{i}'] = door_state
    
    return info
  
  def info2vec(self, info, keys=('player_x', 'player_y', 'player_dir')) -> tuple:
    return tuple([int(info[key]) for key in keys])


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


def determine_task_goal_features(env):
  def navigational_task_goal():
    """Task goal: (goal_x, goal_y, ...don't cares...)."""
    goal_pos = determine_goal_pos(env)
    if goal_pos:
      n_padding_dims = N_GOAL_DIMS - len(goal_pos)
      return np.concatenate((
        goal_pos,
        # NOTE: -1 features mean that they don't matter
        -np.ones((n_padding_dims,), dtype=np.int16)))
  
  def pickup_ball_task_goal():
    """Task goal: (...don't cares..., has_ball=1)."""
    n_padding_dims = N_GOAL_DIMS - 1
    return np.concatenate((
      -np.ones((n_padding_dims,), dtype=np.int16),
      np.asarray([1], dtype=np.int16)
    ))

  nav_goal = navigational_task_goal()
  task_goal_features = nav_goal if nav_goal is not None \
    else pickup_ball_task_goal()

  print(f'[MiniGrid-Environment] GoalFeatures: {task_goal_features}')
  return task_goal_features


# TODO(ab): Consider using MiniGridEnv::hash()
def info2goals(info):
  door_info = extract_door_info(info)
  door_array = [0] * N_DOOR_DIMS
  if door_info:
    door_array[:len(door_info)] = door_info
  return np.array([
    info['player_x'],
    info['player_y'],
    info['has_key'],
    *door_array,
    info['is_lava'],
    info['has_ball']
    ], dtype=np.int16)


def get_all_door_states(env):
  from minigrid.core.world_object import Door
  # State 0: open, 1: closed, 2: locked
  door_states = []
  for i in range(env.grid.width):
    for j in range(env.grid.height):
      tile = env.grid.get(i, j)
      if isinstance(tile, Door):
        obj_type, color, state = tile.encode()
        door_states.append(state)
  return door_states


def extract_door_info(info):
  return [info[var_name] for var_name in info if 'door' in var_name]


def environment_builder(
  level_name='MiniGrid-Empty-16x16-v0',  # MiniGrid-DoorKey-16x16-v0
  reward_fn='sparse',
  add_count_based_bonus=True,
  exploration_reward_scale=0,
  seed=42,
  random_reset=False,
  max_steps=None,
  goal_conditioned=True
):
  if max_steps is not None and max_steps > 0:
    env = gym.make(level_name, max_steps=max_steps)  #, goal_pos=(11, 11))
  else:
    env = gym.make(level_name)
  
  # To fix the start-goal config across episodes
  env = ReseedWrapper(env, seeds=[seed])
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
  env = MiniGridWrapper(
    env,
    num_stacked_frames=1,
    action_repeats=1,
    flatten_frame_stack=True,
    grayscaling=False, 
    pooled_frames=1,
    to_float=True,
    goal_conditioned=goal_conditioned,
    task_goal_features=determine_task_goal_features(env)
  )

  ts0 = env.reset()
  print(env)

  return env
