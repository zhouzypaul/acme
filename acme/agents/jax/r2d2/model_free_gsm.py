import os
import time
import pickle
import threading
import collections

from typing import Dict, Optional, Tuple, Union, List

from acme.wrappers.oar_goal import OARG
from acme.wrappers.observation_action_reward import OAR
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.agents.jax.rnd import networks as rnd_networks
from acme.agents.jax.cfn.networks import CFNNetworks
from acme.agents.jax.rnd.networks import compute_rnd_reward
from acme.agents.jax.cfn.networks import compute_cfn_reward
from acme.jax import variable_utils
from acme.jax import networks as networks_lib
from acme.core import Saveable
from acme.agents.jax.r2d2.model_free_goal_sampler import MFGoalSampler
# from acme.agents.jax.r2d2.goal_sampler import GoalSampler
from acme.utils.paths import get_save_directory
import numpy as np

class GoalSpaceManager(Saveable):
  """Worker that maintains the skill-graph."""

  def __init__(self, goal_space_size, use_tabular_bonuses=True):
    self._hash2proto = {}
    self._hash2counts = collections.defaultdict(int)
    self._hash2bonus = {}
    self._tabular_bonus = use_tabular_bonuses
    self._goal_space_size = goal_space_size
    self._count_dict_lock = threading.Lock()

    # Learning curve for each goal
    self._edge2successes = collections.defaultdict(list)
    self._edge2successes_lock = threading.Lock()

    base_dir = get_save_directory()
    self._gsm_loop_last_timestamp = time.time()
    self._base_plotting_dir = os.path.join(base_dir, 'plots')
    os.makedirs(self._base_plotting_dir, exist_ok=True)

    print('Created model-free GSM.')

  def get_goal_dict(self) -> Dict:
    keys = list(self._hash2proto.keys())
    return {k: self._hash2proto[k] for k in keys}

  def get_count_dict(self) -> Dict:
    keys = list(self._hash2counts.keys())
    return {k: self._hash2counts[k] for k in keys}

  def update(
    self,
    hash2proto: Dict,
    hash2count: Dict,
    edge2success: Dict
  ):
    for hash, proto in hash2proto.items():
      self._hash2proto[hash] = np.asarray(proto)

    with self._count_dict_lock:
      for hash, count in hash2count.items():
        self._hash2counts[hash] += count
        if self._tabular_bonus:
          self._hash2bonus[hash] = 1. / np.sqrt(self._hash2counts[hash] + 1)

    self._update_edge_success_dict(edge2success)

  def _update_edge_success_dict(self, edge2success: Dict):
    with self._edge2successes_lock:
      for key in edge2success:
        self._edge2successes[key].append(edge2success[key])

  def save(self):
    return (
      self._hash2counts,
      self._hash2proto,
      self._hash2bonus,
      self._edge2successes
    )

  def restore(self, state):
    assert len(state) == 4, len(state)
    self._hash2counts = state[0]
    self._hash2proto = state[1]
    self._hash2bonus = state[2]
    self._edge2successes = state[3]

  def step(self):
    if time.time() - self._gsm_loop_last_timestamp > 1 * 60:
      self.dump_plotting_vars()
      self._gsm_loop_last_timestamp = time.time()

  def _reached(self, current_hash, goal_hash) -> bool:  # TODO(ab/mm): don't replicate
    assert isinstance(current_hash, np.ndarray), type(current_hash)
    assert isinstance(goal_hash, np.ndarray), type(goal_hash)

    dims = np.where(goal_hash == 1)

    return (current_hash[dims]).all()

  def begin_episode(self, current_node: Tuple, task_goal_probability: float = 0.1) -> Tuple[Tuple, Dict]:
    goal_sampler = MFGoalSampler(
      self._hash2proto,
      self._hash2counts,
      self._hash2bonus,
      binary_reward_func=self._reached,
      goal_space_size=self._goal_space_size)
    expansion_node = goal_sampler.begin_episode(current_node)
    return expansion_node, {}

  def dump_plotting_vars(self):
    """Save plotting vars for the GSMPlotter to load and do its magic with."""
    try:
      with open(os.path.join(self._base_plotting_dir, 'plotting_vars.pkl'), 'wb') as f:
        pickle.dump(self.save(), f)
    except Exception as e:
      print(f'Failed to dump plotting vars: {e}')