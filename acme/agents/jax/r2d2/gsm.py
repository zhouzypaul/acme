import time
import ipdb
import pickle
import pprint
import itertools
import threading
import collections
import numpy as np

from typing import Dict, Optional, Tuple

from acme.wrappers.oar_goal import OARG
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax import variable_utils
from acme.jax import networks as networks_lib


class GoalSpaceManager(object):
  """Worker that maintains the skill-graph."""

  def __init__(
      self,
      rng_key: networks_lib.PRNGKey,
      networks: r2d2_networks.R2D2Networks,
      variable_client: Optional[variable_utils.VariableClient],
    ):
    self._hash2obs = {}  # map goal hash to obs
    self._hash2counts = collections.defaultdict(int)
    self._count_dict_lock = threading.Lock()
    
    # src goal hash -> dest goal hash -> value
    self._value_matrix = collections.defaultdict(
      lambda : collections.defaultdict(lambda: 1.))
    
    self._rng_key = rng_key
    self._variable_client = variable_client
    self._networks = networks
    self._n_parameter_updates = 0
    
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
    reached = (current.goals == goal.goals).all()
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
  
  def get_goal_dict(self):
    return self._hash2obs
  
  def get_value_dict(self) -> dict:
    """Convert to regular dict because courier cannot handle fancy dicts."""
    return self._default_dict_to_dict(self._value_matrix)
    
  def update(self, hash2obs: Dict, hash2count: Dict):
    """Update based on goals achieved by the different actors."""
    self._hash2obs.update(hash2obs)
    self._update_count_dict(hash2count)
    
  def _update_count_dict(self, hash2count: Dict):
    with self._count_dict_lock:
      for goal in hash2count:
        self._hash2counts[goal] += hash2count[goal]
  
  def _construct_oarg(self, obs, action, reward, goal_features):
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
      return {k: self._construct_oarg(*v, k) for k, v in list(hash2oar.items())}
    
    nodes = get_nodes()
    
    augmented_observations = []
    actions = []
    rewards = []
    goals = []
    src_dest_pairs = []
        
    for src in nodes:
      for dest in nodes:
        obs = nodes[src]
        goal = nodes[dest]
        oarg = self.obs_augment_fn(obs, goal, 'concat')[0]
        augmented_observations.append(oarg.observation)
        actions.append(oarg.action)
        rewards.append(oarg.reward)
        goals.append(oarg.goals)
        src_dest_pairs.append((src, dest))
        
    if augmented_observations:
      augmented_observations = np.asarray(
        augmented_observations)[np.newaxis, ...]
      actions = np.asarray(actions)[np.newaxis, ...]
      rewards = np.asarray(rewards)[np.newaxis, ...]
      goals = np.asarray(goals)[np.newaxis, ...]
      print(f'Created OARG with {len(augmented_observations)} observations, {len(nodes)} nodes')
      return src_dest_pairs, OARG(augmented_observations, actions, rewards, goals)
    
    return None, None
  
  @staticmethod
  def _default_dict_to_dict(dd):
    d = {}
    for key in dd:
      d[key] = {k: v for k, v in dd[key].items()}
    return d
    
  def _save_value_dict(self, iteration):   
    t0 = time.time()
    print('Saving the value matrix..')
    with open(f'value_matrix_iteration_{iteration}.pkl', 'wb+') as f:
      pickle.dump(self._default_dict_to_dict(self._value_matrix), f)
    print(f'Took {time.time() - t0}s to save value matrix.')

  def run(self):
    for iteration in itertools.count():
      t0 = time.time()
      src_dest_pairs, batch_oarg = self._construct_obs_matrix()
      if batch_oarg is not None:
        
        lstm_state = self.get_recurrent_state(batch_oarg.observation.shape[1])
        
        values, _ = self._networks.unroll(
          self._params,
          self._rng_key,
          batch_oarg,
          lstm_state
        )  # (1, B, |A|)
        values = values.max(axis=-1)[0]  # (1, B, |A|) -> (1, B) -> (B,)
        
        print(f'[iteration={iteration}] values={values.shape}, max={values.max()} dt={time.time() - t0}s')
        self._update_value_dict(src_dest_pairs, values)
        if len(src_dest_pairs) > 1000:
          pprint.pprint(self._value_matrix[(1, 1)])
