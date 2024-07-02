import os
import jax
import time
import random
import pickle
import threading
import collections
import jax.numpy as jnp

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
from acme.salient_event.factored import SalientEventClassifier
# from acme.agents.jax.r2d2.goal_sampler import GoalSampler
from acme.utils.paths import get_save_directory
import numpy as np

class GoalSpaceManager(Saveable):
  """Worker that maintains the skill-graph."""

  def __init__(
      self,
      goal_space_size,
      rng_key: networks_lib.PRNGKey,
      use_tabular_bonuses=False,
      networks: Optional[r2d2_networks.R2D2Networks] = None,
      variable_client: Optional[variable_utils.VariableClient] = None,
      exploration_networks: Optional[CFNNetworks] = None,
      exploration_variable_client: Optional[variable_utils.VariableClient] = None,
      use_exploration_vf_for_expansion: bool = False,
      use_intermediate_difficulty: bool = True,
      use_uvfa_reachability: bool = False,
    ):
    self._rng_key = rng_key
    self._hash2proto = {}
    self._hash2counts = collections.defaultdict(int)
    self._tabular_bonus = use_tabular_bonuses
    self._goal_space_size = goal_space_size
    self._count_dict_lock = threading.Lock()

    self._hash2bonus = {}
    self._bonus_counts = {}  # how many times we have updated the bonus for hash.
    self._bonus_counts_lock = threading.Lock()
    self._exploration_networks = exploration_networks
    self._exploration_variable_client = exploration_variable_client
    self._use_exploration_vf_for_expansion = use_exploration_vf_for_expansion
    self._use_intermediate_difficulty = use_intermediate_difficulty
    self._use_uvfa_reachability = use_uvfa_reachability

    self._networks = networks
    self._variable_client = variable_client

    self._hash2obs = {}
    self._hash2obs_lock = threading.Lock()

    self.classifier_id_lock = threading.Lock()
    self.classifiers = []

    # Learning curve for each goal
    self._edge2successes = collections.defaultdict(list)
    self._edge2successes_lock = threading.Lock()

    base_dir = get_save_directory()
    self._gsm_loop_last_timestamp = time.time()
    self._base_plotting_dir = os.path.join(base_dir, 'plots')
    os.makedirs(self._base_plotting_dir, exist_ok=True)

    self._classifier2positives = collections.defaultdict(list)

    print('Created model-free GSM.')
    print(f'[GSM] use_intermediate_difficulty: {use_intermediate_difficulty} ',
          f'use_uvfa_reachability: {use_uvfa_reachability}')

  def get_goal_dict(self) -> Dict:
    keys = list(self._hash2proto.keys())
    return {k: self._hash2proto[k] for k in keys}

  def get_count_dict(self) -> Dict:
    keys = list(self._hash2counts.keys())
    return {k: self._hash2counts[k] for k in keys}
  
  @property
  def _params(self):
    return self._variable_client.params if self._variable_client else []
  
  @property
  def _exploration_params(self):
    return self._exploration_variable_client.params if self._exploration_variable_client else []

  def update(
    self,
    hash2proto: Dict,
    hash2count: Dict,
    edge2success: Dict,
    hash2obs: Dict,
  ):
    for hash, proto in hash2proto.items():
      self._hash2proto[hash] = np.asarray(proto)

    with self._count_dict_lock:
      for hash, count in hash2count.items():
        self._hash2counts[hash] += count
        if self._tabular_bonus:
          self._hash2bonus[hash] = 1. / np.sqrt(self._hash2counts[hash] + 1)

    self._update_edge_success_dict(edge2success)

    with self._hash2obs_lock:
      self._update_obs_dict(hash2obs)

  def _update_edge_success_dict(self, edge2success: Dict):
    with self._edge2successes_lock:
      for key in edge2success:
        self._edge2successes[key].append(edge2success[key])

  def _update_obs_dict(self, hash2obs: Dict):
    for goal in hash2obs:
      oarg = self._construct_oarg(*hash2obs[goal], goal)
      if goal not in self._hash2obs:
        self._hash2obs[goal] = collections.deque(maxlen=10)
      self._hash2obs[goal].append(oarg)

  def _construct_oarg(self, obs, action, reward, goal_features) -> OARG:
    """Convert the obs, action, etc from the GSM into an OARG object.

    Args:
        obs (list): obs image in list format
        action (int): action taken when this oarg was seen
        reward (float): gc reward taken when this oarg was seen
        goal_features (tuple): goal hash in tuple format
    """
    return OARG(
      observation=np.asarray(obs, dtype=obs.dtype),
      action=action,
      reward=reward,
      goals=np.asarray(goal_features, dtype=np.int16)
    )

  def save(self):
    keys = list(self._hash2obs.keys())
    with self._hash2obs_lock:
      hash2obs = {k: list(self._hash2obs[k]) for k in keys}
    return (
      self._hash2counts,
      self._hash2proto,
      self._hash2bonus,
      self._edge2successes,
      self.classifiers,
      hash2obs,
      self._classifier2positives
    )

  def restore(self, state):
    assert len(state) == 7, len(state)
    self._hash2counts = state[0]
    self._hash2proto = state[1]
    self._hash2bonus = state[2]
    self._edge2successes = state[3]
    self.classifiers = state[4]
    self._hash2obs = {k: collections.deque(v, maxlen=10) for k, v in state[5].items()}
    self._classifier2positives = state[6]

  def step(self):
    if self._use_exploration_vf_for_expansion and self._hash2obs:
        self._compute_and_update_novelty_values()
        self.update_params(wait=False)
    
    if time.time() - self._gsm_loop_last_timestamp > 1 * 60:
      self.dump_plotting_vars()
      self._update_classifier_decisions()
      self._gsm_loop_last_timestamp = time.time()

  def update_params(self, wait: bool = False):
    if self._exploration_variable_client:
      self._exploration_variable_client.update(wait=wait)
    if self._variable_client:
      self._variable_client.update(wait=wait)

  def _reached(self, current_hash, goal_hash) -> bool:  # TODO(ab/mm): don't replicate
    assert isinstance(current_hash, np.ndarray), type(current_hash)
    assert isinstance(goal_hash, np.ndarray), type(goal_hash)

    dims = np.where(goal_hash == 1)

    return (current_hash[dims]).all()

  def begin_episode(self, current_node: Tuple, task_goal_probability: float = 0.1) -> Tuple[Tuple, Dict]:
    # print('[GSM] Beginning episode with current node:', current_node)
    goal_sampler = MFGoalSampler(
      self._hash2proto,
      self._hash2counts,
      self._hash2bonus,
      binary_reward_func=self._reached,
      goal_space_size=self._goal_space_size,
      uvfa_params=self._params,
      uvfa_rng_key=self._rng_key,
      uvfa_networks=self._networks,
      use_uvfa_reachability=self._use_uvfa_reachability,
    )
    expansion_node = goal_sampler.begin_episode(current_node)
    return expansion_node, {}
  
  def _nodes2oarg(self, nodes: Dict) -> OARG:
    keys = []
    observations = []
    actions = []
    rewards = []
    goals = []
    for key in nodes:
      oarg = nodes[key]
      keys.append(key)
      observations.append(oarg.observation)
      actions.append(oarg.action)
      rewards.append(oarg.reward)
      goals.append(oarg.goals)
    return keys, OARG(
      observation=jnp.asarray(observations),
      action=jnp.asarray(actions)[jnp.newaxis, ...],
      reward=jnp.asarray(rewards)[jnp.newaxis, ...],
      goals=jnp.asarray(goals)[jnp.newaxis, ...]
    )

  def _compute_and_update_novelty_values(self, n_nodes: int = 50):
    """Compute the CFN value function for the nodes in the GSM."""
    def get_recurrent_state(batch_size=None):
      return self._exploration_networks.direct_rl_networks.init_recurrent_state(
        self._rng_key, batch_size)
    
    t0 = time.time()
    n_nodes = min(n_nodes, len(self._hash2obs))
    keys = random.sample(self._hash2obs.keys(), k=n_nodes)
    key2lens = {key: len(self._hash2obs[key]) for key in keys}
    key2idx = {key: random.choice(range(length)) for key, length in key2lens.items()}
    nodes = {key: self._hash2obs[key][key2idx[key]] for key in keys}
    node_hashes, oarg = self._nodes2oarg(nodes)
    cfn_oar = oarg._replace(
        observation=oarg.observation[..., :3])
    cfn_oar = cfn_oar._replace(
      observation=cfn_oar.observation[None, ...])
    q_values, _ = self._exploration_networks.direct_rl_networks.unroll(
      self._exploration_params,
      self._rng_key,
      cfn_oar,
      get_recurrent_state(len(node_hashes))
    )
    values = q_values.max(axis=-1)[0]  # (1, B, |A|) -> (1, B) -> (B,)
    # clip the values to be between 0 and 1
    # values = values.clip(0., 1.)
    values = values.ravel().tolist()

    # Update the hash counts for the bonus dict
    with self._bonus_counts_lock:
      for key in keys:
        self._bonus_counts[key] = self._bonus_counts.get(key, 0) + 1

    self._update_bonuses(node_hashes, values)
    # print(f'[GSM-Profiling] Took {time.time() - t0}s to compute & update CFN values.')

  def _update_bonuses(self, src_hashes, bonuses,
                      use_incremental_update: bool = False):
    assert len(src_hashes) == len(bonuses)
    for key, value in zip(src_hashes, bonuses):
      if not use_incremental_update:
        if self._use_intermediate_difficulty:
          value = np.clip(value, 0., 1.)
          self._hash2bonus[key] = (-4. * ((value - 0.5) ** 2) + 1)
        else:
          self._hash2bonus[key] = value
      else:
        # Incremental mean update
        assert key in self._bonus_counts and self._bonus_counts[key] > 0, (
          key, self._bonus_counts, self._bonus_counts[key])
        curr = self._hash2bonus.get(key, 0)
        error = value - curr
        self._hash2bonus[key] = curr + (error / self._bonus_counts[key])

  def potentially_register_new_classifier(self, salient_patches: dict, most_novel_img: jax.Array) -> int:
    """Register a new classifier if it doesn't already exist. Return the ID of the classifier."""
    print(f'[GSM] Registering new classifier with {len(salient_patches)} salient patches.')
    
    if len(salient_patches) == 0:
      print(f'[GSM] SalientEventClassifier has no salient patches.')
      return -1
    
    # convert the jnp arrays to np arrays
    salient_patches = {k: np.asarray(v) for k, v in salient_patches.items()}
    
    classifier = SalientEventClassifier(
      salient_patches,
      prototype_image=np.asarray(most_novel_img, dtype=most_novel_img.dtype),
      base_plotting_dir=self._base_plotting_dir,
    )
    
    for existing_classifier in self.classifiers:
      if existing_classifier.equals(classifier):
        print(f'[GSM] Classifier already exists with ID {existing_classifier.classifier_id}.')
        return -1  # Classifier already exists.
        
    with self.classifier_id_lock:  
      print(f'[GSM] Adding new classifier with ID {len(self.classifiers)}.')
      classifier.assign_id(len(self.classifiers))
      self.classifiers.append(classifier)
      return classifier.classifier_id

  def dump_plotting_vars(self):
    """Save plotting vars for the GSMPlotter to load and do its magic with."""
    try:
      with open(os.path.join(self._base_plotting_dir, 'plotting_vars.pkl'), 'wb') as f:
        pickle.dump(self.save(), f)
    except Exception as e:
      print(f'Failed to dump plotting vars: {e}')

  def _update_classifier_decisions(self):
    """Save the decisions made by the classifier."""
    t0 = time.time()

    observations: List[OARG] = []
    
    for obs_deque in self._hash2obs.values():
      observations.extend(list(obs_deque))

    # If there are more than 1000 observations, sample 1000 of them.
    if len(observations) > 1000:
      observations = random.sample(observations, 1000)

    for classifier in self.classifiers:
      decisions = [classifier(obs.observation) for obs in observations]

      # update classifier decisions
      self._classifier2positives[classifier.classifier_id] = [
        obs for obs, decision in zip(observations, decisions) if decision
      ]

      print(f'[GSM] Classifier {classifier.classifier_id} found {sum(decisions)} positives.')
      
    print(f'[GSM] Took {time.time() - t0}s to update {len(self.classifiers)} classifier',
          f'decisions on {len(observations)} observations.')
