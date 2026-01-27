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
import acme.salient_event.classifier as classifier_lib
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
      reachability_novelty_combination_method: str = 'multiplication',
      reachability_novelty_addition_alpha: float = 0.5,
      descendant_threshold: float = 0.1,
      subsampled_classifiers_dir: Optional[str] = None,
      use_pixel_baseline: bool = False,
    ):
    self._use_pixel_baseline = use_pixel_baseline
    self._rng_key = rng_key
    self._hash2proto = {}
    self._hash2counts = collections.defaultdict(int)
    self._hash2avg_reward = collections.defaultdict(float)
    self._selection_history = collections.deque(maxlen=20000)
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
    self._descendant_threshold = descendant_threshold

    self._networks = networks
    self._variable_client = variable_client

    self._hash2obs = {}
    self._hash2obs_lock = threading.Lock()
    self._hash2infos = collections.defaultdict(set)
    self._hash2infos_lock = threading.Lock()

    self.classifier_id_lock = threading.Lock()
    self.classifiers = []

    self.reachability_novelty_addition_alpha = reachability_novelty_addition_alpha
    self.reachability_novelty_combination_method = reachability_novelty_combination_method

    # Learning curve for each goal
    self._edge2successes = collections.defaultdict(list)
    self._edge2successes_lock = threading.Lock()
    
    # Counter for unique states per classifier
    self._hash2unique_state_count = collections.defaultdict(int)

    base_dir = get_save_directory()
    self._gsm_loop_last_timestamp = time.time()
    self._base_plotting_dir = os.path.join(base_dir, 'plots')
    os.makedirs(self._base_plotting_dir, exist_ok=True)

    self._classifier2inferredinfo = {}

    print('Created model-free GSM.')
    print(f'[GSM] use_intermediate_difficulty: {use_intermediate_difficulty} ',
          f'use_uvfa_reachability: {use_uvfa_reachability}')

    if subsampled_classifiers_dir:
      self._initialize_goal_space_from_classifiers(subsampled_classifiers_dir)

  def _initialize_goal_space_from_classifiers(self, subsampled_classifiers_dir: str):
    """Load classifiers from the given directory (or directories), sort them, reassign classifier IDs contiguously,
    and save a mapping from new IDs to old IDs in the first directory.
    
    Args:
        subsampled_classifiers_dir: A single directory path or a comma-separated list of paths.
    """
    # Handle multiple directories
    directories = [d.strip() for d in subsampled_classifiers_dir.split(',')]
    print(f"Initializing goal-space from subsampled classifiers in: {directories}")
    
    combined_files = [] # List of (filepath, directory_index)
    
    for i, directory in enumerate(directories):
        if not os.path.exists(directory):
            print(f"Warning: Classifier directory not found: {directory}")
            continue
            
        files = sorted([f for f in os.listdir(directory) if ('classifier' in f and f.endswith('.pkl'))])
        for f in files:
            combined_files.append(os.path.join(directory, f))
            
    print(f"Total classifiers found across {len(directories)} directories: {len(combined_files)}")

    new_classifiers = []
    new_to_old_mapping = {}
    total_classifiers = len(combined_files)
    new_id = 0

    # First pass: load classifiers and reassign new contiguous IDs.
    for filepath in combined_files:
        with open(filepath, 'rb') as f:
            clf = pickle.load(f)
        old_id = clf.get("classifier_id", None)
        
        # Reassign classifier_id to a new contiguous value.
        clf = classifier_lib.assign_id(clf, new_id)
        if self._use_pixel_baseline:
            clf['use_pixel_baseline'] = True
        new_classifiers.append(clf)
        
        # Store origin path and old ID to resolve ambiguity
        new_to_old_mapping[new_id] = {'old_id': old_id, 'source_path': filepath}
        
        key = clf["classifier_id"]
        # ram state
        # make this a one hot using the key value so dic should have
        # one hot vectors for each classifier
        protovector = np.zeros((total_classifiers,), dtype=bool)
        protovector[key] = True
        self._hash2proto[key] = protovector
        # set break and double check hash2proto
        self._hash2counts[key] = 0
        # _hash2infos will be updated later.
        self._hash2infos[(key,)] = set()
        new_id += 1
        
    # Save the combined mapping to the PRIMARY (first) directory
    if directories and os.path.exists(directories[0]):
        mapping_path = os.path.join(directories[0], "id_mapping_combined.pkl")
        with open(mapping_path, 'wb') as f:
            pickle.dump(new_to_old_mapping, f)
        print(f"Saved combined ID mapping to {mapping_path}")

    # Second pass: for each classifier, compute its goal vector.
    # The goal vector is a boolean array of length total_classifiers.
    # For every classifier in the set that returns True on the current classifier's prototype_image,
    # the corresponding bit is set to True.
    for clf in new_classifiers:
        key = clf["classifier_id"]
        prototype_image = clf['prototype_image']
        goal_vector = np.zeros((total_classifiers,), dtype=bool)
        # Loop over all classifiers and update the goal vector.
        for other in new_classifiers:
            # Use the classify function to determine if the 'other' classifier fires on prototype_image.
            if classifier_lib.classify(other, prototype_image):
                goal_vector[other["classifier_id"]] = True
        # Wrap the prototype_image and the computed goal vector in an OARG.
        default_oarg = self._construct_oarg(
            obs=prototype_image,
            action=0,
            reward=0.0,
            goal_features=goal_vector
        )
        self._hash2obs[key] = collections.deque([default_oarg], maxlen=10)

    self.classifiers = new_classifiers
    print(f"Initialized goal-space with {len(self.classifiers)} classifiers.")

  def get_goal_dict(self) -> Dict:
    keys = list(self._hash2proto.keys())
    return {k: self._hash2proto[k] for k in keys}

  def get_count_dict(self) -> Dict:
    keys = list(self._hash2counts.keys())
    return {k: self._hash2counts[k] for k in keys}

  def get_inferred_info_dict(self) -> Dict:
    return self._classifier2inferredinfo

  def get_unique_state_count_dict(self) -> Dict:
    return dict(self._hash2unique_state_count)
  
  @property
  def _params(self):
    return self._variable_client.params if self._variable_client else []
  
  @property
  def _exploration_params(self):
    return self._exploration_variable_client.params if self._exploration_variable_client else []

  def get_salient_event_classifiers(self) -> List[Dict]:
    return self.classifiers

  def update(
    self,
    hash2proto: Dict,
    hash2count: Dict,
    edge2success: Dict,
    hash2obs: Dict,
    hash2infos: Dict,
    discovered_goals: set = None,  # All discovered goals (including HER)
  ):
    for hash, proto in hash2proto.items():
      self._hash2proto[hash] = np.asarray(proto)

    with self._count_dict_lock:
      for hash, count in hash2count.items():
        self._hash2counts[hash] += count
        
        # Update average reward if observations are available
        if hash in hash2obs:
            # hash2obs[hash] is a SINGLE tuple (obs, action, reward, goal)
            # reward is at index 2
            reward = hash2obs[hash][2]
            sum_rewards = reward
            n_new = 1
            
            # Incremental average update
            # Note: _hash2counts was just incremented by 'count' which should roughly match n_new
            # But strictly: NewAvg = (OldAvg * OldCount + SumNew) / NewCount
            # We treat _hash2counts as the TotalCount. (OldCount = Total - n_new)
            total_count = self._hash2counts[hash]
            old_count = max(0, total_count - n_new)
            old_avg = self._hash2avg_reward[hash]
            
            self._hash2avg_reward[hash] = (old_avg * old_count + sum_rewards) / max(1, total_count)

        if self._tabular_bonus:
          # Use average reward to weight the novelty bonus: (AvgReward + 1) / sqrt(Count)
          # Adding 1 ensures we still explore 0-reward goals based on pure novelty
          self._hash2bonus[hash] = (self._hash2avg_reward[hash] + 1.0) / np.sqrt(self._hash2counts[hash] + 1)

    self._update_edge_success_dict(edge2success)

    with self._hash2obs_lock:
      self._update_obs_dict(hash2obs)

    with self._hash2infos_lock:
      self._update_info_dict(hash2infos)
    
    # Store discovered goals for filtering
    if discovered_goals is not None:
      self._discovered_goals = discovered_goals

  def _update_edge_success_dict(self, edge2success: Dict):
    with self._edge2successes_lock:
      for key in edge2success:
        self._edge2successes[key].append(edge2success[key])

  def _update_info_dict(self, hash2infos: Dict):
    for clf_id_tuple, info_set in hash2infos.items():
      for info in info_set:
        if info not in self._hash2infos[clf_id_tuple]:
          self._hash2infos[clf_id_tuple].add(info)
          self._hash2unique_state_count[clf_id_tuple] += 1

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
      self._classifier2inferredinfo,
      self._hash2infos,
      self._selection_history,
      self._hash2unique_state_count,
    )

  def restore(self, state):
    assert len(state) >= 8, len(state)
    self._hash2counts = state[0]
    self._hash2proto = state[1]
    self._hash2bonus = state[2]
    self._edge2successes = state[3]
    self.classifiers = state[4]
    self._hash2obs = {k: collections.deque(v, maxlen=10) for k, v in state[5].items()}
    self._classifier2inferredinfo = state[6]
    self._hash2infos = state[7]
    self._selection_history = state[8] if len(state) > 8 else collections.deque(maxlen=20000)
    
    if len(state) > 9:
        self._hash2unique_state_count = state[9]
    else:
        # Backward compatibility: populate count from existing sets
        self._hash2unique_state_count = collections.defaultdict(int)
        for k, v in self._hash2infos.items():
            self._hash2unique_state_count[k] = len(v)

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
      reachability_method=self.reachability_novelty_combination_method,
      reachability_novelty_combination_alpha=self.reachability_novelty_addition_alpha,
      descendant_threshold=self._descendant_threshold,
      discovered_goals=getattr(self, '_discovered_goals', set()),  # Pass discovered goals for filtering
    )
    expansion_node = goal_sampler.begin_episode(current_node)
    
    # Track the difficulty (bonus) of the selected node
    if expansion_node in self._hash2bonus:
        self._selection_history.append((time.time(), self._hash2bonus[expansion_node]))
        
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
      obs = oarg.observation
      if obs.ndim == 2:  # (84, 84)
          obs = obs[..., None]
      elif obs.ndim == 3 and obs.shape[2] != 1: # (84, 84, ?) -> (84, 84, 1)
          # Assuming we want the first channel if there are multiple? Or maybe it's (1, 84, 84)?
          # Let's assume (84, 84, C) and we want grayscale so we take mean or first logic from visualizer?
          # Actually, likely it is (1, 84, 84).
          if obs.shape[0] == 1:
              obs = obs.squeeze(0)[..., None]
          elif obs.shape[-1] > 1:
               # Just take first channel to match typical behavior
               obs = obs[..., 0:1]
      
      observations.append(obs)
      actions.append(oarg.action)
      rewards.append(oarg.reward)
      goals.append(oarg.goals)
    
    # Handle inhomogeneous goal shapes (if goal space size changed)
    max_len = max([len(g) for g in goals]) if goals else 0
    padded_goals = []
    for g in goals:
        if len(g) < max_len:
            # Pad with False (zeros)
            padding = np.zeros((max_len - len(g),), dtype=g.dtype)
            padded_goals.append(np.concatenate([g, padding]))
        else:
            padded_goals.append(g)

    return keys, OARG(
      observation=jnp.asarray(observations),
      action=jnp.asarray(actions)[jnp.newaxis, ...],
      reward=jnp.asarray(rewards)[jnp.newaxis, ...],
      goals=jnp.asarray(padded_goals)[jnp.newaxis, ...]
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
    # oarg.observation has shape (B, 84, 84, 1)
    # oarg.action/reward have shape (1, B, ...)
    
    # We need observation to match (1, B, ...) for unroll
    observation = oarg.observation[None, ...] # (1, B, 84, 84, 1)
    
    cfn_oar = oarg._replace(
      observation=jnp.asarray(observation).astype('uint8')
    )
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

      # If using exploration VF, multiply by count-based bonus to get both signals
      if self._use_exploration_vf_for_expansion:
         count_bonus = 1. / np.sqrt(self._hash2counts[key] + 1)
         self._hash2bonus[key] *= count_bonus

  def potentially_register_new_classifier(
    self,
    salient_patches: dict,
    most_novel_img: jax.Array,
    most_novel_info: Tuple,
  ) -> int:
    """Register a new classifier if it doesn't already exist. Return the ID of the classifier."""
    print(f'[GSM] Registering new classifier with {len(salient_patches)} salient patches.')
    
    if len(salient_patches) == 0:
      print(f'[GSM] SalientEventClassifier has no salient patches.')
      return -1

    
    if self._hash2proto:
      key = list(self._hash2proto.keys())[0]
      proto = self._hash2proto[key]

      if len(self.classifiers) >= proto.shape[0]:
        print(f'[GSM] Already have {len(self.classifiers)} classifiers, not registering new one.')
        return -1

    # convert the jnp arrays to np arrays
    salient_patches = {k: np.asarray(v) for k, v in salient_patches.items()}
    
    classifier: Dict = classifier_lib.create_classifier(
      salient_patches,
      prototype_image=np.asarray(most_novel_img, dtype=most_novel_img.dtype),
      prototype_info_vector=most_novel_info,
      base_plotting_dir=self._base_plotting_dir,
    )
    
    for existing_classifier in self.classifiers:
      if classifier_lib.equals(existing_classifier, classifier):
        print(f'[GSM] Classifier already exists with ID {existing_classifier["classifier_id"]}.')
        return -1  # Classifier already exists.
        
    with self.classifier_id_lock:  
      print(f'[GSM] Adding new classifier with ID {len(self.classifiers)}.')
      classifier = classifier_lib.assign_id(classifier, len(self.classifiers))
      self.classifiers.append(classifier)

    return classifier['classifier_id']

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
    
    with self._hash2infos_lock:
      for classifier_id, info_set in self._hash2infos.items():
        # Compute the logical And of all the info vectors
        if len(info_set) == 0:
          continue
        inferred_info = np.ones_like(list(info_set)[0])
        for info in info_set:
          inferred_info = np.logical_and(inferred_info, info)
        idx = classifier_id[0]
        self._classifier2inferredinfo[idx] = inferred_info

    print(f'[GSM] Updated classifier decisions in {time.time() - t0}s.')
