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

  def __init__(self):
    self._hash2counts = collections.defaultdict(int)
    self._hash2bonus = {}
    self._tabular_bonus = True
    self._count_dict_lock = threading.Lock()
    print('Created model-free GSM.')

  def get_goal_set(self) -> Dict:
    return list(self._hash2counts.keys())

  def update(
    self,
    hash2count: Dict,
  ):
    with self._count_dict_lock:
      for hash, count in hash2count.items():
        self._hash2counts[hash] += count
        if self._tabular_bonus:
          self._hash2bonus[hash] = 1. / np.sqrt(self._hash2counts[hash] + 1)

  def save(self):
    return (self._hash2counts,)

  def restore(self, state):
    self._hash2counts = state[0]

  def _reached(self, current_hash, goal_hash) -> bool:  # TODO(ab/mm): don't replicate
    assert isinstance(current_hash, np.ndarray), type(current_hash)
    assert isinstance(goal_hash, np.ndarray), type(goal_hash)

    dims = np.where(goal_hash == 1)

    return (current_hash[dims]).all()

  def begin_episode(self, current_node: Tuple, task_goal_probability: float = 0.1) -> Tuple[Tuple, Dict]:
    goal_sampler = MFGoalSampler(
      self._hash2counts,
      self._hash2bonus,
      binary_reward_func=self._reached,
      goal_space_size=10)
    expansion_node = goal_sampler.begin_episode(current_node)
    return expansion_node, {}
