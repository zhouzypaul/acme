import math

import ipdb
import pickle
import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium.core import Wrapper, ObservationWrapper, ActionWrapper
from minigrid.wrappers import ImgObsWrapper, ReseedWrapper, PositionBonus

from gymnasium import spaces
from acme.wrappers.gymnasium_wrapper import GymnasiumWrapper
from acme.wrappers.oar_goal import ObservationActionRewardGoalWrapper
from acme.wrappers.observation_action_reward import ObservationActionRewardWrapper
from acme.wrappers.minigrid_wrapper import MiniGridWrapper
from minigrid.core.world_object import Lava


INCLUDE_KEY_POS = True


# NOTE: we are assuming a max number of keys and doors
N_POS_DIMS = 2  # player (x, y) location
N_KEY_DIMS = 1 + (INCLUDE_KEY_POS * 2)  # has_key or not, key_x, key_y
N_DOOR_DIMS = 7 # number of possible doors in the puzzle
N_OBJECT_DIMS = 1  # is object being carried or not
N_LAVA_DIMS = 1  # is the location seek/avoid (lava or not)
N_GOAL_DIMS = N_POS_DIMS + N_KEY_DIMS + N_DOOR_DIMS + N_OBJECT_DIMS + N_LAVA_DIMS


class MinigridInfoWrapper(Wrapper):
  """Include extra information in the info dict for debugging/visualizations."""

  def __init__(self, env):
    super().__init__(env)
    self._timestep = 0

    self.env.reset()

    # NOTE: These attributes can be queried after reset()
    self._grid_width = env.grid.width
    self._grid_height = env.grid.height

    self._num_doors = get_num_doors(env)
    self._env_has_key = does_env_have_key(env)
    self._env_has_ball = does_env_have_ball(env)

    # Calculate vector size
    self._player_pos_size = self._grid_width * self._grid_height
    self._key_pos_size = self._player_pos_size
    door_state_size = self._num_doors * 3  # Assuming 3 states: open, closed, locked
    
    # +1s for has_key and has_ball
    self.vector_size = self._player_pos_size + 1 + self._key_pos_size + door_state_size + 1

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

    key_pos = determine_key_pos(self.env)
    info['key_pos'] = (0, 0) if key_pos is None else key_pos

    door_states = get_all_door_states(self)
    for i, door_state in enumerate(door_states):
      info[f'door{i}'] = door_state

    # Static information about the environment
    info['env_width'] = self._grid_width
    info['env_height'] = self._grid_height
    info['env_num_doors'] = self._num_doors
    info['env_has_key'] = self._env_has_key
    info['env_has_ball'] = self._env_has_ball
    info['env_player_pos_size'] = self._player_pos_size
    info['env_key_pos_size'] = self._key_pos_size
    info['env_pg_vector_size'] = self.vector_size
    
    return info
  
  def binary2info(self, binary_vector, sparse_info: bool = False):
    if isinstance(binary_vector, tuple):
      binary_vector = np.asarray(binary_vector)

    info = {}
    width = self.grid.width
    height = self.grid.height
    num_doors = get_num_doors(self)

    player_pos_size = width * height
    key_pos_size = player_pos_size
    
    # Decode player position
    player_vector = binary_vector[:player_pos_size]
    if np.sum(player_vector) > 0:
      player_index = np.where(player_vector == 1)[0][0]
      info['player_y'], info['player_x'] = divmod(player_index, width)
    elif not sparse_info:
      info['player_y'], info['player_x'] = -1, -1
    
    # Decode has_key
    has_key_index = player_pos_size

    if not sparse_info or binary_vector[has_key_index]:
      info['has_key'] = binary_vector[has_key_index]

    def decode_key_pos():
      key_vector = binary_vector[has_key_index + 1:has_key_index + 1 + key_pos_size]
      if np.sum(key_vector) > 0:
        key_index = np.where(key_vector == 1)[0][0]
        key_y, key_x = divmod(key_index, width)
        return (key_x, key_y)
      return (-1, -1)  # unknown
    
    # Decode key position
    keypos = decode_key_pos()
    if INCLUDE_KEY_POS and (not sparse_info or keypos != (-1, -1)):
      info['key_pos'] = keypos
        
    # Decode door states
    door_start_index = player_pos_size + 1 + key_pos_size
    door_states = {0: 'open', 1: 'closed', 2: 'locked'}
    for i in range(num_doors):
      door_vector = binary_vector[door_start_index + i*3 : door_start_index + (i+1)*3]
      if door_vector.sum() > 0:
        door_state = np.where(door_vector == 1)[0][0]
        info[f'door{i}'] = door_states[door_state]
    
    # Decode has_ball
    if not sparse_info or binary_vector[-1]:
      info['has_ball'] = binary_vector[-1]
    
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
  

class RGBImgObsWrapper(ObservationWrapper):
  """
  Wrapper to use fully observable RGB image as observation,
  This can be used to have the agent to solve the gridworld in pixel space.
  """

  def __init__(self, env, tile_size=8):
    super().__init__(env)

    self.tile_size = tile_size

    new_image_space = spaces.Box(
      low=0,
      high=255,
      shape=(self.env.width * tile_size, self.env.height * tile_size, 3),
      dtype="uint8",
    )

    self.observation_space = spaces.Dict(
      {**self.observation_space.spaces, "image": new_image_space}
    )

  def observation(self, obs):
    rgb_img = self.get_frame(highlight=False, tile_size=self.tile_size)

    return {**obs, "image": rgb_img}
  
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
      
def determine_key_pos(env):
  """Convinence hacky function to determine the key location."""
  from minigrid.core.world_object import Key
  for i in range(env.grid.width):
    for j in range(env.grid.height):
      tile = env.grid.get(i, j)
      if isinstance(tile, Key):
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


def determine_binary_goal_features(env):
  def navigational_task_goal():
    """Task goal: (goal_x, goal_y, ...don't cares...)."""
    goal_pos = determine_goal_pos(env)
    proto_vec = np.zeros((env.vector_size,), dtype=bool)
    if goal_pos:
      pos_idx = goal_pos[1] * env.grid.width + goal_pos[0]
      proto_vec[pos_idx] = 1
      return proto_vec
  
  def pickup_ball_task_goal():
    """Task goal: (...don't cares..., has_ball=1)."""
    proto_vec = np.zeros((env.vector_size,), dtype=bool)
    proto_vec[-1] = 1
    return proto_vec

  nav_goal = navigational_task_goal()
  task_goal_features = nav_goal if nav_goal is not None \
    else pickup_ball_task_goal()

  print(f'[MiniGrid-Environment] GoalPGFeatures: {task_goal_features}')
  return task_goal_features


# TODO(ab): Consider using MiniGridEnv::hash()
def info2goals(info):
  door_info = extract_door_info(info)
  door_array = [0] * N_DOOR_DIMS
  if door_info:
    door_array[:len(door_info)] = door_info
  key_array = [info['has_key'], *info['key_pos']] if INCLUDE_KEY_POS else [info['has_key']]
  return np.array([
    info['player_x'],
    info['player_y'],
    *key_array,
    *door_array,
    info['is_lava'],
    info['has_ball']
    ], dtype=np.int16)

  
def determine_num_locations(env):
  """Get number of cells where there is no wall."""
  from minigrid.core.world_object import Wall
  num_locations = 0
  for i in range(env.grid.width):
    for j in range(env.grid.height):
      tile = env.grid.get(i, j)
      if not isinstance(tile, Wall):
        num_locations += 1
  return num_locations


def get_num_doors(env):
  from minigrid.core.world_object import Door
  num_doors = 0
  for i in range(env.grid.width):
    for j in range(env.grid.height):
      tile = env.grid.get(i, j)
      if isinstance(tile, Door):
        num_doors += 1
  return num_doors


def does_env_have_key(env):
  from minigrid.core.world_object import Key
  for i in range(env.grid.width):
    for j in range(env.grid.height):
      tile = env.grid.get(i, j)
      if isinstance(tile, Key):
        return True
  return False


def does_env_have_ball(env):
  from minigrid.core.world_object import Ball
  for i in range(env.grid.width):
    for j in range(env.grid.height):
      tile = env.grid.get(i, j)
      if isinstance(tile, Ball):
        return True
  return False


def info2binary(info):
  """Convert the info dict to a proto-goal binary vector."""
  width = info['env_width']
  num_doors = info['env_num_doors']
  vector_size = info['env_pg_vector_size']
  
  binary_vector = np.zeros((vector_size,), dtype=bool)
  
  # Encode player position
  player_index = info['player_y'] * width + info['player_x']
  binary_vector[player_index] = 1
  
  # Encode has_key
  if info['has_key']:
    assert info['env_has_key'], info
    binary_vector[info['env_player_pos_size']] = 1
  
  # Encode key position
  if INCLUDE_KEY_POS and info['env_has_key'] and not info['has_key']:
    key_index = info['env_player_pos_size'] + 1 + (info['key_pos'][1] * width) + info['key_pos'][0]
    binary_vector[key_index] = 1
  
  # Encode door states
  door_start_index = info['env_player_pos_size'] + 1 + info['env_key_pos_size']
  for i in range(num_doors):
    door_state = info[f'door{i}']
    binary_vector[door_start_index + (i * 3) + door_state] = 1
  
  # Encode has_ball
  binary_vector[-1] = 1 if info['has_ball'] else 0
  
  return binary_vector


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
  seed=0,
  random_reset=False,
  max_steps=None,
  goal_conditioned=True,
  to_float=False
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
    to_float=to_float,
    goal_conditioned=goal_conditioned,
    task_goal_features=determine_binary_goal_features(env),
    scale_dims=(104, 104),
  )
  
  # Use the OARG Wrapper
  env = ObservationActionRewardGoalWrapper(
    env,
    # info2goals,
    info2binary,
    # n_goal_dims=N_GOAL_DIMS
    n_goal_dims=env.vector_size
  )

  ts0 = env.reset()
  print(env.environment.env)
  print(f's0 = {ts0.observation.goals}')

  return env
