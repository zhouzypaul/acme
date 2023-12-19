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
import matplotlib.pyplot as plt

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
    ):
    self._environment = environment
    self._exploration_algorithm_is_cfn = exploration_algorithm_is_cfn
    self._prob_augmenting_bonus_constant = prob_augmenting_bonus_constant
    self._connect_nodes_one_step_away = connect_nodes_one_step_away
    self._off_policy_edge_threshold = off_policy_edge_threshold
    self._rmax_factor = rmax_factor
    self._use_pessimistic_graph_for_planning = use_pessimistic_graph_for_planning
    self._max_vi_iterations = max_vi_iterations

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

    self._hash2bellman = collections.defaultdict(lambda: collections.deque(maxlen=50))
    self._hash2vstar = collections.defaultdict(list)

    # Learning curve for each goal
    self._hash2successes = collections.defaultdict(list)
    self._hash2successes_lock = threading.Lock()

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

  def begin_episode(self, current_node: Tuple) -> Tuple[Tuple, Dict]:
    """Create and solve the AMDP. Then return the abstract policy."""
    goal_sampler = GoalSampler(
      *self.get_variables(),
      task_goal_probability=0.1,
      task_goal=self.task_goal,
      exploration_goal=self.exploration_goal,
      exploration_goal_probability=0.,
      rmax_factor=self._rmax_factor,
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
      rmax_factor=self._rmax_factor
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
    hash2success: Dict,
  ):
    """Update based on goals achieved by the different actors."""
    self._update_obs_dict(hash2obs)
    self._update_count_dict(hash2count)
    self._update_on_policy_count_dict(edge2count)
    self._update_idx_dict(hash2obs)
    self._hash2discount.update(hash2discount)
    self._update_edges_set(expansion_node_new_node_hash_pairs)
    self._update_on_policy_success_dict(hash2success)
    
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

  def _update_on_policy_success_dict(self, hash2success: Dict):
    with self._hash2successes_lock:
      for key in hash2success:
        if key != self.exploration_hash:
          self._hash2successes[key].append(hash2success[key])

  def _update_idx_dict(self, hash2obs: Dict):
    with self._idx_dict_lock:
      for hash in hash2obs:
        if hash not in self._hash2idx:
          idx = len(self._hash2idx)
          self._hash2idx[hash] = idx
          self._idx2hash[idx] = hash

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
    
    if len(self._hash2idx) >= self._n_actions:
      self._resize_transition_tensor()

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
        
  def _construct_obs_matrix(self):
    def get_nodes():  # TODO(ab): is this just a deepcopy of hash2goal?
      return self.get_goal_dict()

    def get_exploration_node():
      exploration_oarg = self.exploration_goal
      return {tuple(exploration_oarg.goals): exploration_oarg}
    
    def get_exploitation_node():
      task_goal = self.task_goal
      return {tuple(task_goal.goals): task_goal}
    
    def nodes2oarg(nodes: Dict) -> OARG:
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
    
    # TODO(ab): why get all nodes if we are going to subsample 50?
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
      for dest in dest_nodes:
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

      src_keys, src_oarg = nodes2oarg(nodes)

      return src_dest_pairs, OARG(augmented_observations, actions, rewards, goals), src_keys, src_oarg
    
    return None, None, None, None
  
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

  def visualize_value_function(
    self,
    episode: int,
    task_value_vector: Optional[np.ndarray] = None,
    exploration_value_vector: Optional[np.ndarray] = None,
    n_goal_subplots: int = 4
  ):
    assert n_goal_subplots % 2 == 0, 'Ask for even number of subplots.'
    assert task_value_vector is None or len(task_value_vector.shape) == 1
    assert exploration_value_vector is None or len(exploration_value_vector.shape) == 1

    plot_task_value_fn = task_value_vector is not None
    plot_explore_value_fn = exploration_value_vector is not None

    def _get_value(src: Tuple, dest: Tuple):
      row = self._hash2idx[src]
      col = self._hash2idx[dest]
      return self._transition_matrix[row, col]

    starts = list(self._hash2idx.keys())
    
    if len(starts) < n_goal_subplots:
      return
    
    selected_starts = random.sample(starts, k=n_goal_subplots)

    plt.figure(figsize=(14, 14))
    n_subplots = n_goal_subplots + \
      int(plot_task_value_fn) + int(plot_explore_value_fn)

    for i, start in enumerate(selected_starts):
      xs = []; ys = []; values = []
      for goal_hash in starts:
        xs.append(goal_hash[0])
        ys.append(goal_hash[1])
        value = _get_value(start, goal_hash)
        values.append(value)
      plt.subplot(n_subplots // 2, n_subplots // 2, i + 1)
      plt.scatter(xs, ys, c=values)
      plt.colorbar()
      plt.title(f'Start State: {start}')

    if plot_task_value_fn:
      xs = []; ys = []; values = []
      for start in starts:
        xs.append(start[0])
        ys.append(start[1])
        values.append(task_value_vector[self._hash2idx[start]])
      plt.subplot(n_subplots // 2, n_subplots // 2, n_goal_subplots + 1)
      plt.scatter(xs, ys, c=values)
      plt.colorbar()
      plt.title('Task Reward Function')

    if plot_explore_value_fn:
      xs = []; ys = []; values = []
      for start in starts:
        xs.append(start[0])
        ys.append(start[1])
        values.append(exploration_value_vector[self._hash2idx[start]])
      plt.subplot(n_subplots // 2, n_subplots // 2, n_goal_subplots + 2)
      plt.scatter(xs, ys, c=values)
      plt.colorbar()
      plt.title('Exploration Context')

    plt.savefig(os.path.join(self._gcvf_plotting_dir, f'uvfa_{episode}.png'))
    plt.close()

  def _make_spatial_bonus_plot(self, episode):
    hashes = list(self._hash2bonus.keys())
    xs, ys, bonuses = [], [], []
    for hash in hashes:
      xs.append(hash[0])
      ys.append(hash[1])
      bonuses.append(self._hash2bonus[hash])
    plt.scatter(xs, ys, c=bonuses, s=40, marker='s')
    plt.colorbar()
    plt.savefig(os.path.join(self._spatial_plotting_dir, f'spatial_bonus_{episode}.png'))
    plt.close()

  def _plot_discovered_goals(self, episode):
    for goal_hash in self._thread_safe_deepcopy(self._hash2obs):
      if goal_hash not in self._already_plotted_goals:
        obs = self._hash2obs[goal_hash]
        # TODO(ab): Maybe compute the bonus if it is not already in hash2bonus.
        score = self._hash2bonus[goal_hash] if goal_hash in self._hash2bonus else 0.
        filename = f'goal_{goal_hash}_episode_{episode}.png'
        title = f'Score: {score:.3f} ' + \
                f'Mean: {self._exploration_params.reward_mean:.3f} ' + \
                f'Var: {self._exploration_params.reward_var:.3f}'
        plt.imshow(obs.observation)
        plt.title(title)
        plt.savefig(os.path.join(self._discovered_goals_dir, filename))
        plt.close()
        
        self._already_plotted_goals.add(goal_hash)

  def _plot_hash2bonus(self, episode):
    hashes = list(self._hash2bonus.keys())
    xs, ys, bonuses = [], [], []
    for hash in hashes:
      xs.append(hash[0])
      ys.append(hash[1])
      bonuses.append(self._hash2bonus[hash])
    plt.scatter(xs, ys, c=bonuses, s=40, marker='s')
    plt.colorbar()
    plt.savefig(os.path.join(self._node_expansion_prob_dir, f'expansion_probs_{episode}.png'))
    plt.close()

  def _plot_bellman_errors(self, episode):
    """Randomly sample 4 nodes and plot their bellman errors as a function of iteration."""
    nodes = list(self._hash2bellman.keys())
    
    if nodes:
      selected_nodes = random.sample(nodes, k=min(len(nodes), 4))
      plt.figure(figsize=(14, 14))
      for i, node in enumerate(selected_nodes):
        plt.subplot(2, 2, i + 1)
        plt.plot(self._hash2bellman[node], marker='o', linestyle='-')
        plt.title(f'Goal: {node}')
      plt.suptitle(f'MBE vs # VI Iterations at GSM Iteration {episode}')
      plt.savefig(os.path.join(self._bellman_errors_plotting_dir, f'bellman_errors_{episode}.png'))
      plt.close()

  def _plot_spatial_vstar(self, episode):
    """Spatially plot the AMDP V* for 4 randomly sampled goal nodes."""
    def plot_vf(hash2val: Dict, name: str):
      xs = []; ys = []; values = []
      for key, val in hash2val.items():
        xs.append(key[0])
        ys.append(key[1])
        values.append(val)

      if values:
        plt.scatter(xs, ys, c=values, s=100, marker='s')
        plt.colorbar()
        plt.title(name)

    nodes = list(self._hash2vstar.keys())
    
    if nodes:
      selected_nodes = random.sample(nodes, k=min(len(nodes), 4))
      plt.figure(figsize=(14, 14))
      for i, node in enumerate(selected_nodes):
        plt.subplot(2, 2, i + 1)
        plot_vf(self._hash2vstar[node], name=f'Goal: {node}')
      plt.suptitle(f'AMDP V-Star at GSM Iteration {episode}')
      plt.savefig(os.path.join(self._amdp_vstar_plotting_dir, f'vstar_{episode}.png'))
      plt.close()

  def _plot_skill_graph(self, episode, include_off_policy_edges=True):
    """Spatially plot the nodes and edges of the skill-graph."""

    def split_edges(edges, hash_bit):
      """Split edges based on whether the hash bit is on/off for src and dest."""
      no_no = []
      no_yes = []
      yes_no = []
      yes_yes = []
      for edge in edges:
        src_hash = edge[0]
        dest_hash = edge[1]
        if src_hash[hash_bit] == 0 and dest_hash[hash_bit] == 0:
          no_no.append(edge)
        elif src_hash[hash_bit] == 0 and dest_hash[hash_bit] == 1:
          no_yes.append(edge)
        elif src_hash[hash_bit] == 1 and dest_hash[hash_bit] == 0:
          yes_no.append(edge)
        elif src_hash[hash_bit] == 1 and dest_hash[hash_bit] == 1:
          yes_yes.append(edge)
      return no_no, no_yes, yes_no, yes_yes

    def plot_edges(e, color):
      for edge in e:
        x1 = edge[0][0]
        y1 = edge[0][1]
        x2 = edge[1][0]
        y2 = edge[1][1]
        plt.scatter([x1, x2], [y1, y2], color=color)
        plt.plot([x1, x2], [y1, y2], color=color, alpha=0.3)

    def split_then_plot(edges, hash_bit, color):
      no_no, no_yes, yes_no, yes_yes = split_edges(edges, hash_bit=2)
      
      plt.subplot(2, 2, 1)
      plot_edges(no_no, color=color)
      plt.title('No Key -> No Key')
      plt.subplot(2, 2, 2)
      plot_edges(no_yes, color=color)
      plt.title('No Key -> Yes Key')
      plt.subplot(2, 2, 3)
      plot_edges(yes_no, color=color)
      plt.title('Yes Key -> No Key')
      plt.subplot(2, 2, 4)
      plot_edges(yes_yes, color=color)
      plt.title('Yes Key -> Yes Key')
    
    edges = list(self._edges)
    off_policy_edges = list(self._off_policy_edges)
    plt.figure(figsize=(14, 14))
    split_then_plot(edges, hash_bit=2, color='black')
    split_then_plot(off_policy_edges, hash_bit=2, color='red')
    
    plt.savefig(os.path.join(self._skill_graph_plotting_dir, f'skill_graph_{episode}.png'))
    plt.close()

  def step(self):
    t0 = time.time()
    iteration: int = next(self._iteration_iterator)

    src_dest_pairs, batch_oarg, cfn_nodes, cfn_oarg = self._construct_obs_matrix()
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
      self._update_transition_tensor(src_dest_pairs, values)

      if self._exploration_algorithm_is_cfn:
        cfn_oar = cfn_oarg._replace(
          observation=cfn_oarg.observation[..., :3])
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
          observation=cfn_oarg.observation[..., :3],
          action=cfn_oarg.action[None, ...],
          reward=cfn_oarg.reward[None, ...])
        bonuses = compute_rnd_reward(
          self._exploration_params.params,
          self._exploration_params.target_params,
          exploration_transitions,
          self._exploration_networks,
          self._exploration_params.observation_mean,
          self._exploration_params.observation_var,
        )
      bonuses = bonuses.ravel().tolist()
      self._update_bonuses(cfn_nodes, bonuses)

      if len(src_dest_pairs) > 10:
        # TODO(ab): get from env and pass around
        # start_state_features = (1, 5, 0, 0)  
        # start_state_features = (8, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # FourRooms
        # start_state_features = (2, 10, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0)  # DoorKey
        # start_state_features = (3, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0)  # S3R1
        start_state_features = (7, 7, 0, 1, 1, 1, 1, 2, 1, 0, 0, 0)  # S5R3
        if start_state_features in self._hash2idx:
          row_idx = self._hash2idx[start_state_features]
          pprint.pprint(self._transition_matrix[row_idx, :len(self._hash2idx)])

      dt = time.time() - self._gsm_loop_last_timestamp
      print(f'Iteration {iteration} Goal Space Size {len(self._hash2obs)} dt={dt}')
      self._gsm_iteration_times.append(dt)

      self.dump_plotting_vars()

      self.update_params(wait=False)
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
    to_return = (*to_return,
                 self._edges,
                 self._off_policy_edges,
                 self._exploration_params.reward_mean,
                 self._exploration_params.reward_var,
                 hash2bell,
                 self._thread_safe_deepcopy(self._hash2vstar),
                 self._gsm_iteration_times,
                 self._hash2successes)
    assert len(to_return) == 17, len(to_return)
    print(f'[GSM] Checkpointing took {time.time() - t0}s.')
    return to_return

  def restore(self, state: Tuple[Dict]):
    t0 = time.time()
    print('About to start restoring GSM from checkpoint.')
    assert len(state) == 17, len(state)
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
    self._hash2successes = state[16]
    assert isinstance(self._edges, set), type(state[9])
    assert isinstance(self._off_policy_edges, set), type(state[10])
    print(f'[GSM] Restored transition tensor {self._transition_matrix.shape}')
    print(f'[GSM] Took {time.time() - t0}s to restore from checkpoint.')

  def restore_transition_tensor(self, transition_matrix):
    k = self._tensor_increments
    self._n_actions = ((len(self._hash2idx) // k) + 1) * k
    self._n_states = self._n_actions
    transition_tensor = np.zeros(
      (self._n_states, self._n_actions), dtype=np.float32)
    n_real_nodes = len(transition_matrix)
    transition_tensor[:n_real_nodes, :n_real_nodes] = transition_matrix
    return transition_tensor
