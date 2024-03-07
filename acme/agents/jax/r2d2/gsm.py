import os
import time
import ipdb
import pickle
import pprint
import random
import dm_env
import itertools
import threading
import collections
import numpy as np
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
from acme.agents.jax.r2d2.goal_sampler import GoalSampler
from acme.utils.paths import get_save_directory


class GoalSpaceManager(Saveable):
  """Worker that maintains the skill-graph."""

  def __init__(
      self,
      environment: dm_env.Environment,
      rng_key: networks_lib.PRNGKey,
      networks: r2d2_networks.R2D2Networks,
      variable_client: variable_utils.VariableClient,
      exploration_networks: Union[rnd_networks.RNDNetworks, CFNNetworks],
      exploration_variable_client: variable_utils.VariableClient,
      tensor_increments: int = 1000,
      exploration_algorithm_is_cfn: bool = True,
      prob_augmenting_bonus_constant : float = 0.1,
      connect_nodes_one_step_away: bool = False,
      off_policy_edge_threshold: float = 0.75,
      rmax_factor: float = 2.,
      use_pessimistic_graph_for_planning: bool = False,
      max_vi_iterations: int = 20,
      goal_space_size: int = 100,
      should_switch_goal: bool = False,
      use_exploration_vf_for_expansion: bool = False
    ):
    self._environment = environment
    self._exploration_algorithm_is_cfn = exploration_algorithm_is_cfn
    self._prob_augmenting_bonus_constant = prob_augmenting_bonus_constant
    self._connect_nodes_one_step_away = connect_nodes_one_step_away
    self._off_policy_edge_threshold = off_policy_edge_threshold
    self._rmax_factor = rmax_factor
    self._use_pessimistic_graph_for_planning = use_pessimistic_graph_for_planning
    self._max_vi_iterations = max_vi_iterations
    self._goal_space_size = goal_space_size
    self._should_switch_goal = should_switch_goal
    self._use_exploration_vf_for_expansion = use_exploration_vf_for_expansion

    if exploration_algorithm_is_cfn:
      assert isinstance(exploration_networks, CFNNetworks), type(exploration_networks)

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
    self._hash2bonus = {}
    
    self._tensor_increments = tensor_increments
    self._n_actions = tensor_increments
    self._n_states = self._n_actions
    
    self._transition_matrix = np.zeros(
      (self._n_states, self._n_actions), dtype=np.float32)
    
    self._hash2idx = {}
    self._idx2hash = {}
    self._idx_dict_lock = threading.Lock()

    self._edges = set()
    self._edges_set_lock = threading.Lock()

    self._off_policy_edges = set()
    
    self._rng_key = rng_key
    self._variable_client = variable_client
    self._exploration_variable_client = exploration_variable_client

    self._networks = networks
    self._exploration_networks = exploration_networks

    self._iteration_iterator = itertools.count()
    self._gsm_loop_last_timestamp = time.time()
    self._gsm_iteration_times = []

    self._already_plotted_goals = set()
    self.has_seen_task_goal = False

    self._hash2bellman = collections.defaultdict(lambda: collections.deque(maxlen=50))
    self._hash2vstar = collections.defaultdict(list)

    # Learning curve for each goal
    self._edge2successes = collections.defaultdict(list)
    self._edge2successes_lock = threading.Lock()

    base_dir = get_save_directory()
    self._base_plotting_dir = os.path.join(base_dir, 'plots')
    self._gsm_iteration_times_dir = os.path.join(self._base_plotting_dir, 'gsm_iteration_times')
    os.makedirs(self._base_plotting_dir, exist_ok=True)
    os.makedirs(self._gsm_iteration_times_dir, exist_ok=True)

    print(f'[GSM] Created GSM with R-Max factor {self._rmax_factor}',
          f'Off-policy edge threshold {self._off_policy_edge_threshold}',
          f'Prob augmenting bonus constant {self._prob_augmenting_bonus_constant}',
          f'Use pessimistic graph for planning {self._use_pessimistic_graph_for_planning}',
          f'Max VI iterations {max_vi_iterations}')

  def begin_episode(self, current_node: Tuple, task_goal_probability: float = 0.1) -> Tuple[Tuple, Dict]:
    """Create and solve the AMDP. Then return the abstract policy."""
    goal_sampler = GoalSampler(
      *self.get_variables(),
      task_goal_probability=task_goal_probability,
      task_goal=self.task_goal,
      exploration_goal=self.exploration_goal,
      exploration_goal_probability=0.,
      rmax_factor=self._rmax_factor,
      goal_space_size=self._goal_space_size,
      should_switch_goal=self._should_switch_goal,
      max_vi_iterations=self._max_vi_iterations
    )
    expansion_node = goal_sampler.begin_episode(current_node)

    if goal_sampler._amdp:
      abstract_policy = goal_sampler._amdp.get_policy()
      max_bellman_errors = goal_sampler._amdp.max_bellman_errors
      self._hash2bellman[expansion_node].extend(max_bellman_errors)
      self._hash2vstar[expansion_node] = goal_sampler._amdp.get_values()
      return expansion_node, abstract_policy
  
  def get_descendants(self, current_node: Tuple): 
    return GoalSampler(
      *self.get_variables(),
      task_goal_probability=0.1,
      task_goal=self.task_goal,
      exploration_goal=self.exploration_goal,
      exploration_goal_probability=0.,
      rmax_factor=self._rmax_factor,
      goal_space_size=self._goal_space_size,
    ).get_descendants(current_node)

  def get_variables(self, names=()):
    del names
    hash2idx = self._thread_safe_deepcopy(self._hash2idx)
    n_actions = self._transition_matrix.shape[1]
    n_nodes = min(n_actions, len(hash2idx))
    hash2idx = {k: v for k, v in hash2idx.items() if v < n_actions}
    transition_tensor = self.get_transition_tensor(n_nodes)
    return self.get_goal_dict(),\
          self.get_count_dict(),\
          self.get_bonus_dict(),\
          self.get_on_policy_count_dict(),\
          *self.get_extrinsic_reward_dicts(),\
          hash2idx,\
          transition_tensor,\
          self._thread_safe_deepcopy(self._idx2hash)
    
  # TODO(ab): lock transition matrix during the copy operation
  def get_transition_tensor(self, n_actions: int):
    n_states = n_actions
    actual_transition_matrix = self._transition_matrix[:n_states, :n_actions].copy()
    return actual_transition_matrix

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
  def exploration_hash(self) -> Tuple:
    obs = self._environment.observation_spec()
    exploration_goal_feats = -1 * np.ones(
      self._environment.task_goal_features.shape, dtype=obs.goals.dtype)
    return tuple(exploration_goal_feats)
    
  @property
  def _params(self):
    return self._variable_client.params if self._variable_client else []
  
  @property
  def _exploration_params(self):
    return self._exploration_variable_client.params if self._exploration_variable_client else []
  
  def update_params(self, wait: bool = False):
    """Update to a more recent copy of the learner params."""
    if self._variable_client:
      t0 = time.time()
      self._variable_client.update(wait)
      print(f'[GSM] Took {time.time() - t0}s to update GSM learner params.',
            f'VarUpdatePeriod={self._variable_client._update_period.total_seconds()}s',
            f'TimeSinceLastVarUpdate={time.time() - self._variable_client._last_call}s')
    if self._exploration_variable_client:
      self._exploration_variable_client.update(wait)

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
  
  def is_special_context(self, hash: Tuple) -> bool:
    return hash == tuple(self.exploration_goal.goals) or hash == tuple(self.task_goal.goals)
  
  def get_recurrent_state(self, batch_size=None):
    return self._networks.init_recurrent_state(self._rng_key, batch_size)
  
  def get_count_dict(self) -> Dict:
    return self._thread_safe_deepcopy(self._hash2counts)
  
  def get_bonus_dict(self) -> Dict:
    return self._thread_safe_deepcopy(self._hash2bonus)
  
  def get_on_policy_count_dict(self):
    with self._on_policy_count_dict_lock:
      return self._default_dict_to_dict(self._on_policy_counts)
  
  def get_goal_dict(self):
    return self._thread_safe_deepcopy(self._hash2obs)
  
  def get_on_policy_edges(self):
    return self._edges
  
  def get_extrinsic_reward_dicts(self) -> Tuple[Dict, Dict]:
    return self._thread_safe_deepcopy(self._hash2reward),\
      self._thread_safe_deepcopy(self._hash2discount)
    
  def update(
    self,
    hash2obs: Dict,
    hash2count: Dict,
    edge2count: Dict,
    hash2discount: Dict,
    expansion_node_new_node_hash_pairs: List[Tuple[Tuple, Tuple]],
    edge2success: Dict,
  ):
    """Update based on goals achieved by the different actors."""
    self._update_obs_dict(hash2obs)
    self._update_count_dict(hash2count)
    self._update_on_policy_count_dict(edge2count)
    self._update_idx_dict(hash2obs)
    self._hash2discount.update(hash2discount)
    self._update_edges_set(expansion_node_new_node_hash_pairs)
    self._update_edge_success_dict(edge2success)
    
  def _update_count_dict(self, hash2count: Dict):
    with self._count_dict_lock:
      for goal in hash2count:
        self._hash2counts[goal] += hash2count[goal]

  def _update_obs_dict(self, hash2obs: Dict):
    for goal in hash2obs:
      oarg = self._construct_oarg(*hash2obs[goal], goal)
      self._hash2obs[goal] = oarg
      self._hash2reward[goal] = oarg.reward

      if oarg.reward > 0:
        self.has_seen_task_goal = True
        
  def _update_on_policy_count_dict(self, hash2count: Dict):
    with self._on_policy_count_dict_lock:
      for key in hash2count:
        src, dest = key
        self._on_policy_counts[src][dest] += hash2count[key]

  def _update_edge_success_dict(self, edge2success: Dict):
    with self._edge2successes_lock:
      for key in edge2success:
        if self.exploration_hash not in key:
          self._edge2successes[key].append(edge2success[key])

  def _update_idx_dict(self, hash2obs: Dict):
    with self._idx_dict_lock:
      for hash in hash2obs:
        if hash not in self._hash2idx:
          idx = len(self._hash2idx)
          self._hash2idx[hash] = idx
          self._idx2hash[idx] = hash

  def get_has_seen_task_goal(self):
    return self.has_seen_task_goal

  def _update_edges_set(
    self,
    expansion_node_new_node_pairs: List[Tuple[Tuple, Tuple]]
  ):
    with self._edges_set_lock:

      # Add the new nodes to their corresponding expansion nodes.
      for expansion_node, new_node_hash in expansion_node_new_node_pairs:
        self._edges.add((expansion_node, new_node_hash))

        # Connect the new node to all the nodes that the expansion node is connected to.
        if self._connect_nodes_one_step_away:
          for connected_node in self._get_one_step_connected_nodes(expansion_node):
            self._edges.add((connected_node, new_node_hash))

      print(f'[GSM] Number of edges: {len(self._edges)}')

  def _get_one_step_connected_nodes(self, node: Tuple) -> List[Tuple]:
    connected_nodes = []
    for edge in self._edges:
      if edge[1] == node:
        connected_nodes.append(edge[0])
    return connected_nodes
  
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

  # TODO(ab): lock the transition matrix while updating it.
  def _update_transition_tensor(self, src_dest_pairs, values):
    """Update the transition tensor with the values from the UVFA network."""
    
    if len(self._hash2idx) >= self._n_actions:
      self._resize_transition_tensor()

    for (src, dest), value in zip(src_dest_pairs, values):
      if not self.is_special_context(dest) and \
        src in self._hash2idx and dest in self._hash2idx:
        src_idx = self._hash2idx[src]
        dest_idx = self._hash2idx[dest]
        
        prob = np.clip(value, 0., 1.)
        
        # NOTE: We are adding the bonus to the unclipped value.
        bonus = 1 / np.sqrt(self._on_policy_counts[src][dest] + 1)
        weighted_bonus = self._prob_augmenting_bonus_constant * bonus
        
        optimistic_prob = np.clip(value + weighted_bonus, 0., 1.)
        pessimistic_prob = np.clip(value - weighted_bonus, 0., 1.) if \
          self._use_pessimistic_graph_for_planning else prob
        
        if (src, dest) not in self._off_policy_edges and pessimistic_prob > self._off_policy_edge_threshold:
          print(f'[GSM] Adding off-policy edge {src} -> {dest} (prob={pessimistic_prob:.3f})')
          self._off_policy_edges.add((src, dest))
        
        if (src, dest) in self._off_policy_edges and pessimistic_prob <= self._off_policy_edge_threshold:
          print(f'[GSM] Removing off-policy edge {src} -> {dest} (prob={pessimistic_prob:.3f})')
          self._off_policy_edges.remove((src, dest))

        if (src, dest) in self._edges or (src, dest) in self._off_policy_edges:
          self._transition_matrix[src_idx, dest_idx] = optimistic_prob
        else:
          self._transition_matrix[src_idx, dest_idx] = 0.      

  def _resize_transition_tensor(self):
    """Dynamically resize the transition tensor."""
    t0 = time.time()
    n_actions = self._n_actions * 2
    n_states = n_actions
    old_transition_matrix = self._transition_matrix.copy()
    self._transition_matrix = np.zeros(
      (n_states, n_actions),
      dtype=self._transition_matrix.dtype)
    self._transition_matrix[
      :old_transition_matrix.shape[0],
      :old_transition_matrix.shape[1]
    ] = old_transition_matrix
    self._n_states, self._n_actions = self._transition_matrix.shape
    print(f'[GSM] Resized transition tensor to {self._transition_matrix.shape} in {time.time() - t0}s')

  def _update_bonuses(self, src_hashes, bonuses):
    assert len(src_hashes) == len(bonuses)
    for key, value in zip(src_hashes, bonuses):
      self._hash2bonus[key] = value

  def _edges2oarg(self, edges: List[Tuple]) -> OARG:
    """Convert the edges to an OARG object."""
    goals = []
    actions = []
    rewards = []
    augmented_observations = []
    for src, dest in edges:
      if src in self._hash2obs and dest in self._hash2obs:
        src_obs = self._hash2obs[src]
        dest_obs = self._hash2obs[dest]
        oarg = self.obs_augment_fn(src_obs, dest_obs, 'concat')[0]
        augmented_observations.append(oarg.observation)
        actions.append(oarg.action)
        rewards.append(oarg.reward)
        goals.append(oarg.goals)
      
    if augmented_observations:
      augmented_observations = jnp.asarray(
        augmented_observations)[jnp.newaxis, ...]
      augmented_observations = jnp.asarray(augmented_observations)
      actions = jnp.asarray(actions)[jnp.newaxis, ...]
      rewards = jnp.asarray(rewards)[jnp.newaxis, ...]
      goals = jnp.asarray(goals)[jnp.newaxis, ...]

      return OARG(augmented_observations, actions, rewards, goals)
    
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
  
  def _oarg2probabilities(self, batch_oarg):
    """Use the UVFA network to compute the edge probabilities."""
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
    
    return values
  
  def _update_on_policy_edge_probabilities(self, n_edges: int = 1000):
    """Update the transition tensor with the values from the UVFA network."""
    t0 = time.time()
    n_edges = min(n_edges, len(self._edges))
    edges = random.sample(self._edges, k=n_edges)
    batch_oarg = self._edges2oarg(edges)
    if batch_oarg:
      print(f'[GSM] Updating {len(edges)} on-policy edges.')
      values = self._oarg2probabilities(batch_oarg)
      self._update_transition_tensor(edges, values)
    print(f'[GSM-Profiling] Took {time.time() - t0}s to update on-policy edges.')

  def _update_off_policy_edge_probabilities(self, n_nodes: int = 50):
    """Update the transition tensor with the values from the UVFA network."""
    t0 = time.time()
    n_nodes = min(n_nodes, len(self._hash2obs))
    keys = random.sample(self._hash2obs.keys(), k=n_nodes)
    nodes = {key: self._hash2obs[key] for key in keys}
    edges = [(g1, g2) for g1 in nodes for g2 in nodes if g1 != g2]
    batch_oarg = self._edges2oarg(edges)
    if batch_oarg:
      print(f'[GSM] Updating {len(edges)} off-policy edges.')
      values = self._oarg2probabilities(batch_oarg)
      self._update_transition_tensor(edges, values)
    print(f'[GSM-Profiling] Took {time.time() - t0}s to update off-policy edges.')

  def _compute_and_update_novelty_values(self, n_nodes: int = 50):
    """Compute the CFN value function for the nodes in the GSM."""
    def get_recurrent_state(batch_size=None):
      return self._exploration_networks.direct_rl_networks.init_recurrent_state(
        self._rng_key, batch_size)
    
    t0 = time.time()
    n_nodes = min(n_nodes, len(self._hash2obs))
    keys = random.sample(self._hash2obs.keys(), k=n_nodes)
    nodes = {key: self._hash2obs[key] for key in keys}
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
    self._update_bonuses(node_hashes, values)
    print(f'[GSM-Profiling] Took {time.time() - t0}s to compute & update CFN values.')

  def _compute_and_update_novelty_bonuses(self, n_nodes: int = 50):
    """Compute the novelty bonuses for the nodes in the GSM."""
    t0 = time.time()
    n_nodes = min(n_nodes, len(self._hash2obs))
    keys = random.sample(self._hash2obs.keys(), k=n_nodes)
    nodes = {key: self._hash2obs[key] for key in keys}
    node_hashes, oarg = self._nodes2oarg(nodes)

    if self._exploration_algorithm_is_cfn:
      print(f'[GSM] Computing {len(node_hashes)} novelty bonuses.')
      cfn_oar = oarg._replace(
        observation=oarg.observation[..., :3])
      bonuses = compute_cfn_reward(
        self._exploration_params.params,
        self._exploration_params.target_params,
        cfn_oar,
        self._exploration_networks,
        self._exploration_params.random_prior_mean,
        jnp.sqrt(self._exploration_params.random_prior_var + 1e-4),
      )
    else:
      exploration_transitions = OAR(
        observation=oarg.observation[..., :3],
        action=oarg.action[None, ...],
        reward=oarg.reward[None, ...])
      bonuses = compute_rnd_reward(
        self._exploration_params.params,
        self._exploration_params.target_params,
        exploration_transitions,
        self._exploration_networks,
        self._exploration_params.observation_mean,
        self._exploration_params.observation_var,
      )
    bonuses = bonuses.ravel().tolist()
    self._update_bonuses(node_hashes, bonuses)
    print(f'[GSM-Profiling] Took {time.time() - t0}s to compute & bonuses.')
  
  @staticmethod
  def _default_dict_to_dict(dd):
    d = {}
    keys = list(dd.keys())
    for key in keys:
      d[key] = dict(dd[key])
    return d
  
  @staticmethod
  def _dict_to_default_dict(nested_dict, inner_default):
    dd = collections.defaultdict(
      lambda: collections.defaultdict(inner_default))
    for key in nested_dict:
      dd[key] = collections.defaultdict(
        inner_default, nested_dict[key])
    return dd
  
  @staticmethod
  def _thread_safe_deepcopy(d: Dict):
    # Technically size of d can change during list(d.keys()),
    # but low prob and hasn't happened yet.
    keys = list(d.keys())
    return {k: d[k] for k in keys}

  def step(self):
    iteration: int = next(self._iteration_iterator)

    if len(self._hash2obs) > 2:
      self._update_on_policy_edge_probabilities()
      self._update_off_policy_edge_probabilities()
      if self._use_exploration_vf_for_expansion:
        self._compute_and_update_novelty_values()
      else:
        self._compute_and_update_novelty_bonuses()
      self.dump_plotting_vars()
      self.update_params(wait=False)

      dt = time.time() - self._gsm_loop_last_timestamp
      print(f'Iteration {iteration} Goal Space Size {len(self._hash2obs)} dt={dt}')
      self._gsm_iteration_times.append(dt)
      self._gsm_loop_last_timestamp = time.time()

  def run(self):
    for iteration in self._iteration_iterator:
      self.step()
      print(f'[GSM-RunLoop] Iteration {iteration} Goal Space Size {len(self._hash2obs)}')

  def dump_plotting_vars(self):
    """Save plotting vars for the GSMPlotter to load and do its magic with."""
    try:
      with open(os.path.join(self._base_plotting_dir, 'plotting_vars.pkl'), 'wb') as f:
        pickle.dump(self.save(), f)
    except Exception as e:
      print(f'Failed to dump plotting vars: {e}')

  def save(self) -> Tuple[Dict]:
    t0 = time.time()
    print('[GSM] Checkpointing..')
    to_return = self.get_variables()
    hash2bell = self._thread_safe_deepcopy(self._hash2bellman)
    hash2bell = {k: list(v) for k, v in hash2bell.items()}
    reward_mean = self._exploration_params.reward_mean if not self._use_exploration_vf_for_expansion else 0.
    reward_var = self._exploration_params.reward_var if not self._use_exploration_vf_for_expansion else 0.
    to_return = (*to_return,
                 self._edges,
                 self._off_policy_edges,
                 reward_mean,
                 reward_var,
                 hash2bell,
                 self._thread_safe_deepcopy(self._hash2vstar),
                 self._gsm_iteration_times,
                 self._edge2successes,
                 self.has_seen_task_goal)
    assert len(to_return) == 18, len(to_return)
    print(f'[GSM] Checkpointing took {time.time() - t0}s.')
    return to_return

  def restore(self, state: Tuple[Dict]):
    t0 = time.time()
    print('About to start restoring GSM from checkpoint.')
    assert len(state) == 18, len(state)
    self._hash2obs = state[0]
    self._hash2counts = collections.defaultdict(int, state[1])
    self._hash2bonus = state[2]
    self._on_policy_counts = self._dict_to_default_dict(state[3], int)
    self._hash2reward = state[4]
    self._hash2discount = state[5]
    self._hash2idx = state[6]
    self._transition_matrix = self.restore_transition_tensor(state[7])
    self._idx2hash = state[8]
    self._edges = state[9]
    self._off_policy_edges = state[10]
    self._hash2bellman = collections.defaultdict(
      lambda: collections.deque(maxlen=50),
      {k: collections.deque(v, maxlen=50) for k, v in state[13].items()})
    self._hash2vstar = state[14]
    self._gsm_iteration_times = state[15]
    self._edge2successes = state[16]
    self.has_seen_task_goal = state[17]
    assert isinstance(self._edges, set), type(state[9])
    assert isinstance(self._off_policy_edges, set), type(state[10])
    assert isinstance(self.has_seen_task_goal, bool), type(state[17])
    print(f'[GSM] Restored transition tensor {self._transition_matrix.shape}')
    print(f'[GSM] Took {time.time() - t0}s to restore from checkpoint.')

    # _edges = self.get_all_edges_with(src=(8, 11, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0))
    # print(f'Edges from (8, 11, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0): {_edges}')

  def restore_transition_tensor(self, transition_matrix):
    k = self._tensor_increments
    self._n_actions = ((len(self._hash2idx) // k) + 1) * k
    self._n_states = self._n_actions
    transition_tensor = np.zeros(
      (self._n_states, self._n_actions), dtype=np.float32)
    n_real_nodes = len(transition_matrix)
    transition_tensor[:n_real_nodes, :n_real_nodes] = transition_matrix
    return transition_tensor
  
  def get_all_edges_with(self, src=None, dest=None):
    """Get all edges that have src or dest as a node."""
    edges = set()
    for edge in self._edges:
      if src is not None and edge[0] == src:
        prob = self._transition_matrix[self._hash2idx[src], self._hash2idx[edge[1]]]
        edges.add((edge, prob))
      if dest is not None and edge[1] == dest:
        prob = self._transition_matrix[self._hash2idx[edge[0]], self._hash2idx[dest]]
        edges.add((edge, prob))
    return edges
