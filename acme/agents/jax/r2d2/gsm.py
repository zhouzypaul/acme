import time
import ipdb
import copy
import pickle
import pprint
import random
import dm_env
import itertools
import threading
import collections
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from typing import Dict, Optional, Tuple

from acme.wrappers.oar_goal import OARG
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax import variable_utils
from acme.jax import networks as networks_lib
from acme.core import VariableSource, Saveable


class GoalSpaceManager(VariableSource, Saveable):
  """Worker that maintains the skill-graph."""

  def __init__(
      self,
      environment: dm_env.Environment,
      rng_key: networks_lib.PRNGKey,
      networks: r2d2_networks.R2D2Networks,
      variable_client: Optional[variable_utils.VariableClient],
    ):
    self._environment = environment

    self._hash2obs = {}  # map goal hash to obs
    self._hash2counts = collections.defaultdict(int)
    self._count_dict_lock = threading.Lock()
    
    # Map src node -> dest node -> on policy attempt count
    self._on_policy_counts = collections.defaultdict(
      lambda : collections.defaultdict(int))
    self._on_policy_count_dict_lock = threading.Lock()

    # Extrinsic reward function
    self._hash2reward = {}
    self._hash2discount = {}
    
    # src goal hash -> dest goal hash -> value
    self._value_matrix = collections.defaultdict(
      lambda : collections.defaultdict(lambda: 1.))
    
    self._rng_key = rng_key
    self._variable_client = variable_client
    self._networks = networks
    self._n_parameter_updates = 0
    self._iteration_iterator = itertools.count()

  def get_variables(self, names):
    del names
    return self.get_value_dict(),\
          self.get_goal_dict(),\
          self.get_count_dict(),\
          self.get_on_policy_count_dict(),\
          *self.get_extrinsic_reward_dicts()

  @property
  def task_goal(self) -> OARG:
    obs = self._environment.observation_spec()
    obs_shape = (84, 84, 3)  # TODO(ab)
    return OARG(
      observation=np.zeros(obs_shape, dtype=obs.observation.dtype),
      action=np.zeros(obs.action.shape, dtype=obs.action.dtype),  # doesnt matter
      reward=np.zeros(obs.reward.shape, dtype=obs.reward.dtype),  # doesnt matter
      goals=np.asarray(self._environment.task_goal_features, dtype=obs.goals.dtype)
    )
  
  @property
  def exploration_goal(self) -> OARG:
    obs = self._environment.observation_spec()
    obs_shape = (84, 84, 3)  # TODO(ab)
    exploration_goal_feats = -1 * np.ones(
      self._environment.task_goal_features.shape, dtype=obs.goals.dtype)
    return OARG(
      observation=np.ones(obs_shape, dtype=obs.observation.dtype),
      action=np.zeros(obs.action.shape, dtype=obs.action.dtype),  # doesnt matter
      reward=np.zeros(obs.reward.shape, dtype=obs.reward.dtype),  # doesnt matter
      # TODO(ab): assign it a goal that is not achievable
      goals=exploration_goal_feats
    )
    
  @property
  def _params(self):
    return self._variable_client.params if self._variable_client else []
  
  def update_params(self, wait: bool = False):
    """Update to a more recent copy of the learner params."""
    if self._variable_client:
      self._variable_client.update(wait)
      self._n_parameter_updates += 1
      
  def goal_reward_func(self, current: OARG, goal: OARG) -> Tuple[bool, float]:
    """Is the goal achieved in the current state."""
    dims = np.where(goal.goals >= 0)
    reached = (current.goals[dims] == goal.goals[dims]).all()
    return reached, float(reached)
      
  def obs_augment_fn(
    self, obs: OARG, goal: OARG, method: str
  ) -> Tuple[OARG, bool, float]:
    new_obs = self.augment(
      obs.observation, goal.observation, method=method)
    reached, reward = self.goal_reward_func(obs, goal)
    return OARG(
      observation=new_obs,  # pursued goal
      action=obs.action,
      reward=np.array(reward, dtype=np.float32),
      goals=obs.goals  # achieved goals
    ), reached, reward
    
  @staticmethod
  def augment(
    obs: np.ndarray, goal: np.ndarray, method: str
  ) -> np.ndarray:
    assert method in ('concat', 'relabel'), method
    if method == 'concat':
      return np.concatenate((obs, goal), axis=-1)
    if method == 'relabel':
      n_goal_dims = n_obs_dims = obs.shape[-1] // 2
      return np.concatenate(
        (obs[:, :, :n_obs_dims],
         goal[:, :, :n_goal_dims]), axis=-1)
    raise NotImplementedError(method)
  
  def get_recurrent_state(self, batch_size=None):
    return self._networks.init_recurrent_state(self._rng_key, batch_size)
  
  def get_count_dict(self):
    return self._hash2counts
  
  def get_on_policy_count_dict(self):
    return self._default_dict_to_dict(self._on_policy_counts)
  
  def get_goal_dict(self):
    return self._hash2obs
  
  def get_value_dict(self) -> dict:
    """Convert to regular dict because courier cannot handle fancy dicts."""
    return self._default_dict_to_dict(self._value_matrix)
  
  def get_extrinsic_reward_dicts(self) -> Tuple[Dict, Dict]:
    return self._hash2reward, self._hash2discount
    
  def update(
    self,
    hash2obs: Dict,
    hash2count: Dict,
    edge2count: Dict,
    hash2discount: Dict
  ):
    """Update based on goals achieved by the different actors."""
    self._update_obs_dict(hash2obs)
    self._update_count_dict(hash2count)
    self._update_on_policy_count_dict(edge2count)
    self._hash2discount.update(hash2discount)
    
  def _update_count_dict(self, hash2count: Dict):
    with self._count_dict_lock:
      for goal in hash2count:
        self._hash2counts[goal] += hash2count[goal]

  def _update_obs_dict(self, hash2obs: Dict):
    for goal in hash2obs:
      oarg = self._construct_oarg(*hash2obs[goal], goal)
      self._hash2obs[goal] = oarg
      self._hash2reward[goal] = oarg.reward
        
  def _update_on_policy_count_dict(self, hash2count: Dict):
    with self._on_policy_count_dict_lock:
      for key in hash2count:
        src, dest = key
        self._on_policy_counts[src][dest] += hash2count[key]
  
  def _construct_oarg(self, obs, action, reward, goal_features) -> OARG:
    """Convert the obs, action, etc from the GSM into an OARG object.

    Args:
        obs (list): obs image in list format
        action (int): action taken when this oarg was seen
        reward (float): gc reward taken when this oarg was seen
        goal_features (tuple): goal hash in tuple format
    """
    return OARG(
      observation=np.asarray(obs, dtype=np.float32),
      action=action,
      reward=reward,
      goals=np.asarray(goal_features, dtype=np.int16)
    )
        
  def _update_value_dict(self, src_dest_pairs, values):
    """Make a probabilistic graph in goal-space."""
    for (src, dest), value in zip(src_dest_pairs, values):
      self._value_matrix[src][dest] = value.item()
        
  def _construct_obs_matrix(self):
    def get_nodes():
      hash2oar = self.get_goal_dict()
      return {k: v for k, v in list(hash2oar.items())}

    def get_exploration_node():
      exploration_oarg = self.exploration_goal
      return {tuple(exploration_oarg.goals): exploration_oarg}
    
    def get_exploitation_node():
      task_goal = self.task_goal
      return {tuple(task_goal.goals): task_goal}
    
    t0 = time.time()
    nodes = get_nodes()
    t1 = time.time()

    if len(nodes) > 50:
      sub_keys = random.sample(nodes.keys(), 50)
      nodes = {key: nodes[key] for key in sub_keys}
    
    t2 = time.time()

    dest_nodes = {
      **nodes, 
      **get_exploitation_node(), 
      **get_exploration_node()
    }

    t3 = time.time()
    
    augmented_observations = []
    actions = []
    rewards = []
    goals = []
    src_dest_pairs = []
        
    for src in nodes:
      for dest in dest_nodes:  # TODO(ab): Include explore/exploit VFs here.
        obs = nodes[src]
        goal = dest_nodes[dest]
        oarg = self.obs_augment_fn(obs, goal, 'concat')[0]
        augmented_observations.append(oarg.observation)
        actions.append(oarg.action)
        rewards.append(oarg.reward)
        goals.append(oarg.goals)
        src_dest_pairs.append((src, dest))

    t4 = time.time()
        
    if augmented_observations:
      augmented_observations = jnp.asarray(
        augmented_observations)[jnp.newaxis, ...]
      augmented_observations = jnp.asarray(augmented_observations)
      actions = jnp.asarray(actions)[jnp.newaxis, ...]
      rewards = jnp.asarray(rewards)[jnp.newaxis, ...]
      goals = jnp.asarray(goals)[jnp.newaxis, ...]
      print(f'Created OARG with {len(augmented_observations)} observations, {len(nodes)} nodes')
      
      t5 = time.time()
      print(f'[GSM-Profiling] Took {t1 - t0}s to get_nodes().')
      print(f'[GSM-Profiling] Took {t2 - t1}s to subsample nodes.')
      print(f'[GSM-Profiling] Took {t3 - t2}s to create dest_nodes.')
      print(f'[GSM-Profiling] Took {t4 - t3}s to create goal pairs.')
      print(f'[GSM-Profiling] Took {t5 - t4}s to create goal tensors.')

      return src_dest_pairs, OARG(augmented_observations, actions, rewards, goals)
    
    return None, None
  
  @staticmethod
  def _default_dict_to_dict(dd):
    d = {}
    for key, val in dd.items():
      d[key] = dict(val)
    return d
  
  @staticmethod
  def _dict_to_default_dict(nested_dict, inner_default):
    dd = collections.defaultdict(
      lambda: collections.defaultdict(inner_default))
    for key in nested_dict:
      dd[key] = collections.defaultdict(
        inner_default, nested_dict[key])
    return dd
    
  def _save_value_dict(self, iteration):   
    t0 = time.time()
    print('Saving the value matrix..')
    with open(f'value_matrix_iteration_{iteration}.pkl', 'wb+') as f:
      pickle.dump(self._default_dict_to_dict(self._value_matrix), f)
    print(f'Took {time.time() - t0}s to save value matrix.')

  def visualize_value_function(
    self,
    value_dict: Dict,
    episode: int,
    n_goal_subplots: int = 4, 
    plot_task_value_fn: bool = True,
    plot_explore_value_fn: bool = True
  ):
    assert n_goal_subplots % 2 == 0, 'Ask for even number of subplots.'

    starts = list(value_dict.keys())
    
    if len(starts) < n_goal_subplots:
      return
    
    selected_starts = random.sample(starts, k=n_goal_subplots)

    plt.figure(figsize=(14, 14))
    n_subplots = n_goal_subplots + \
      int(plot_task_value_fn) + int(plot_explore_value_fn)

    for i, start in enumerate(selected_starts):
      xs = []; ys = []; values = []
      for (x, y, key, door) in value_dict[start]:
        xs.append(x)
        ys.append(y)
        value = min(value_dict[start][(x, y, key, door)], 1)
        values.append(value)
      plt.subplot(n_subplots // 2, n_subplots // 3, i + 1)
      plt.scatter(xs, ys, c=values)
      plt.colorbar()
      plt.title(f'Start State: {start}')

    if plot_task_value_fn:
      xs = []; ys = []; values = []
      task_goal_feats = tuple(self.task_goal.goals)
      for start in starts:
        (x, y, key, door) = start
        xs.append(x)
        ys.append(y)
        values.append(value_dict[start][task_goal_feats])
      plt.subplot(n_subplots // 2, n_subplots // 3, n_goal_subplots + 1)
      plt.scatter(xs, ys, c=values)
      plt.colorbar()
      plt.title(f'Task Goal ({task_goal_feats})')

    if plot_explore_value_fn:
      xs = []; ys = []; values = []
      explore_goal_feats = tuple(self.exploration_goal.goals)
      for start in starts:
        (x, y, key, door) = start
        xs.append(x)
        ys.append(y)
        values.append(value_dict[start][explore_goal_feats])
      plt.subplot(n_subplots // 2, n_subplots // 3, n_goal_subplots + 2)
      plt.scatter(xs, ys, c=values)
      plt.colorbar()
      plt.title(f'Exploration Goal ({explore_goal_feats})')

    plt.savefig(f'plots/uvfa_plots/four_rooms_uvfa_episode_{episode}.png')
    plt.close()

  def step(self):
    t0 = time.time()
    iteration: int = next(self._iteration_iterator)

    src_dest_pairs, batch_oarg = self._construct_obs_matrix()
    if batch_oarg is not None:
      
      lstm_state = self.get_recurrent_state(batch_oarg.observation.shape[1])
      
      t1 = time.time()
      values, _ = self._networks.unroll(
        self._params,
        self._rng_key,
        batch_oarg,
        lstm_state
      )  # (1, B, |A|)
      print(f'Took {time.time() - t1}s to forward pass through {values.shape} values')
      values = values.max(axis=-1)[0]  # (1, B, |A|) -> (1, B) -> (B,)
      
      print(f'[iteration={iteration}] values={values.shape}, max={values.max()} dt={time.time() - t0}s')
      self._update_value_dict(src_dest_pairs, values)

      if iteration > 0 and iteration % 100 == 0:
        # self._save_value_dict(iteration)
        self.visualize_value_function(self._value_matrix, iteration)

      if len(src_dest_pairs) > 1000:
        # start_state_features = (1, 5, 0, 0)  # TODO(ab): get from env and pass around
        start_state_features = (2, 10, 0, 0)
        # start_state_features = (8, 16, 0, 0)
        pprint.pprint(self._value_matrix[start_state_features])

      print(f'Iteration {iteration} Goal Space Size {len(self._hash2obs)}')

  def run(self):
    for iteration in self._iteration_iterator:
      self.step()
      print(f'[GSM-RunLoop] Iteration {iteration} Goal Space Size {len(self._hash2obs)}')

  def save(self) -> Tuple[Dict]:
    t0 = time.time()
    print('[GSM] Checkpointing..')
    to_return = self.get_variables(names=[])
    assert len(to_return) == 6, len(to_return)
    print(f'[GSM] Checkpointing took {time.time() - t0}s.')
    return to_return

  def restore(self, state: Tuple[Dict]):
    t0 = time.time()
    print('About to start restoring GSM from checkpoint.')
    assert len(state) == 6, len(state)
    ipdb.set_trace()
    self._value_matrix = collections.defaultdict(
      lambda : collections.defaultdict(lambda: 1.), state[0])
    self._hash2obs = state[1]
    self._hash2counts = state[2]
    self._on_policy_counts = self._dict_to_default_dict(state[3], int)
    self._hash2reward = state[4]
    self._hash2discount = state[5]
    print(f'[GSM] Took {time.time() - t0}s to restore from checkpoint.')
