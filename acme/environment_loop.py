# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple agent-environment training loop."""

import os
import operator
import time
import math
import itertools
from typing import List, Optional, Sequence, Tuple, Dict, Callable
import matplotlib.pyplot as plt

from acme import core
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.utils import signals
from acme.wrappers.oar_goal import OARG
from acme.agents.jax.r2d2 import GoalSpaceManager
from acme.agents.jax.r2d2.subgoal_sampler import SubgoalSampler
from acme.utils.utils import GoalBasedTransition
from acme.utils.utils import termination, truncation, continuation
from acme.utils.utils import scores2probabilities, remove_duplicates_keep_last
from acme import specs as env_specs
from acme.utils.paths import get_save_directory
from acme.agents.jax.cfn.cfn import CFN
from acme.agents.jax.cfn.networks import compute_cfn_reward, CFNNetworks
from acme import dsc
from acme.agents.jax.r2d2.networks import R2D2Networks

import dm_env
from dm_env import specs
import numpy as np
import tree
import random
import copy
import collections
import ipdb
import jax.numpy as jnp

NodeHash = Tuple


class EnvironmentLoop(core.Worker):
  """A simple RL environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. Agent is updated if `should_update=True`. This can be used as:

    loop = EnvironmentLoop(environment, actor)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.

  A list of 'Observer' instances can be specified to generate additional metrics
  to be logged by the logger. They have access to the 'Environment' instance,
  the current timestep datastruct and the current action.
  """

  def __init__(
      self,
      environment: dm_env.Environment,
      actor: core.Actor,
      exploration_actor: Optional[core.Actor] = None,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      should_update: bool = True,
      label: str = 'environment_loop',
      observers: Sequence[observers_lib.EnvLoopObserver] = (),
      goal_space_manager: GoalSpaceManager = None,
      task_goal_probability: float = 0.1,
      always_learn_about_task_goal: bool = True,
      always_learn_about_exploration_goal: bool = False,
      actor_id: int = 0,
      use_random_policy_for_exploration: bool = True,
      pure_exploration_probability: float = 1.,
      cfn: Optional[CFN] = None,
      exploration_networks: Optional[CFNNetworks] = None,
      exploitation_networks: Optional[R2D2Networks] = None,
      n_sigmas_threshold_for_goal_creation: int = 0,
      novelty_threshold_for_goal_creation: float = -1.,
      is_evaluator: bool = False,
      planner_backup_strategy: str = 'graph_search',
      max_option_duration: int = 400
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._actor = actor
    self._actor_id = actor_id
    self._exploration_actor = exploration_actor
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        label, steps_key=self._counter.get_steps_key())
    self._should_update = should_update
    self._observers = observers
    self._goal_space_manager = goal_space_manager
    self._task_goal_probability = task_goal_probability
    self._always_learn_about_task_goal = always_learn_about_task_goal
    self._always_learn_about_exploration_goal = always_learn_about_exploration_goal
    self._use_random_policy_for_exploration = use_random_policy_for_exploration
    self._pure_exploration_probability = pure_exploration_probability
    self._cfn = cfn
    self._exploration_networks = exploration_networks
    self._exploitation_networks = exploitation_networks
    self._n_sigmas_threshold_for_goal_creation = n_sigmas_threshold_for_goal_creation
    self._novelty_threshold_for_goal_creation = novelty_threshold_for_goal_creation
    self._planner_backup_strategy = planner_backup_strategy
    self._max_option_duration = max_option_duration

    self.goal_dict = {}
    self.count_dict = {}
    self._n_courier_errors = 0
    self._has_seen_task_goal = False
    self._is_evaluator = is_evaluator
    self._edges = set()
    self._reward_dict = {}
    self._discount_dict = {}

    self._goal_achievement_rates = collections.defaultdict(float)
    self._goal_pursual_counts = collections.defaultdict(int)
    self._node2successes = collections.defaultdict(list)
    
    self._planner_failure_history = []

    self._env_spec = env_specs.make_environment_spec(self._environment)

    # For debugging and visualizations
    self._start_ts = None
    self._episode_iter = itertools.count()
    self._binary2info = lambda vec: self._environment.binary2info(vec, sparse_info=True)

    base_dir = get_save_directory()
    self._exploration_traj_dir = os.path.join(base_dir, 'plots', 'exploration_trajectories')
    self._target_node_plot_dir = os.path.join(base_dir, 'plots', 'target_node_plots')
    self._n_planner_failures_dir = os.path.join(base_dir, 'plots', 'n_planner_failures')
    
    os.makedirs(self._exploration_traj_dir, exist_ok=True)
    os.makedirs(self._target_node_plot_dir, exist_ok=True)
    os.makedirs(self._n_planner_failures_dir, exist_ok=True)
    
    print(f'Going to save exploration trajectories to {self._exploration_traj_dir}')
    print(f'Going to save target node plots to {self._target_node_plot_dir}')
    
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
  def task_goal_hash(self) -> Tuple:
    return tuple(self._environment.task_goal_features)

  def _reached(self, current_hash, goal_hash) -> bool:
    assert isinstance(current_hash, np.ndarray), type(current_hash)
    assert isinstance(goal_hash, np.ndarray), type(goal_hash)

    dims = np.where(goal_hash == 1)

    if np.any(current_hash == -1):
      import ipdb; ipdb.set_trace()  # Shouldn't happen

    return (current_hash[dims]).all()
    
  def goal_reward_func(self, current: OARG, goal: np.ndarray) -> Tuple[bool, float]:
    """Is the goal achieved in the current state."""
    
    reached = self._reached(current.goals, goal)
    return reached, float(reached)

  def augment_ts_with_goal(
    self,
    timestep: dm_env.TimeStep,
    goal: np.ndarray,  # 1-hot vector
    method: str
  ) -> dm_env.TimeStep:
    """Concatenate the goal to the current observation."""
    new_oarg, reached, reward = self.augment_obs_with_goal(
      timestep.observation, goal, method)
    if reached:
      return dm_env.TimeStep(
        step_type=dm_env.StepType.LAST,
        reward=np.array(reward, dtype=np.float32),
        observation=new_oarg,
        discount=np.array(0, dtype=np.float32)
      )

    # NOTE(ab): overwrites extrinsic reward (maintains done though)
    return timestep._replace(observation=new_oarg,
                             reward=np.array(reward, dtype=np.float32))
  
  # TODO(ab): proto-goals - we need to target 1-hot vectors.
  def augment_obs_with_goal(
    self,
    obs: OARG,
    goal: np.ndarray,  # 1-hot vector
    method: str
  ) -> Tuple[OARG, bool, float]:
    def binary2img(proto_goal: np.ndarray) -> np.ndarray:
      assert proto_goal.dtype == bool, proto_goal.dtype
      goal_img_shape = (*obs.observation.shape[:2], 1)
      flat_obs_shape = np.prod(goal_img_shape)  # 84 * 84
      goal_image = np.zeros(flat_obs_shape, dtype=obs.observation.dtype)
      goal_image[np.where(proto_goal)] = 1.
      return goal_image.reshape(goal_img_shape)
    
    new_obs = self.augment(
      obs.observation,
      binary2img(goal),
      method=method)
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
      return np.concatenate((obs[:, :, :3], goal), axis=-1)
    raise NotImplementedError(method)

  # TODO(ab): this should be based on which options are available at s_t?
  # TODO(ab): proto-goals - this will now correspond to a *list* of abstract nodes.
  def _get_current_node(self, timestep: dm_env.TimeStep) -> Tuple:
    return tuple([int(g) for g in timestep.observation.goals])
  
  def episodic_rollout(
    self,
    timestep: dm_env.TimeStep,
    episode_logs: Dict
  ) -> Tuple[List[GoalBasedTransition], List[Tuple[OARG, OARG]], Tuple]:
    """Single episode rollout of the DSG algorithm."""
    
    def reached_expansion_node(ts: dm_env.TimeStep, node: Tuple) -> bool:
      if node is not None:
        # TODO(ab): proto-goals - ensure node is 1-hot
        obs = ts.observation
        return self._reached(obs.goals, np.asarray(node, dtype=obs.goals.dtype))
      return False
    
    needs_reset = False
    expansion_node = None

    # Map pairs of tuples (each containing the hot indices) to whether the edge was successful.
    overall_attempted_edges: List[Tuple[NodeHash, NodeHash, bool]] = []
    
    while not needs_reset:
      t0 = time.time()
      expansion_node = tuple(self.task_goal.goals)
      current_node = self._get_current_node(timestep)

      # TODO(ab/mm): Add task goal feature
      if self._goal_space_manager:
        ret, _ = self._goal_space_manager.begin_episode(
          current_node, task_goal_probability=1.0 if self._is_evaluator else 0.1)
        if ret is not None:
          expansion_node = ret

      print(f'[EnvironmentLoop] Expansion Node: {self._binary2info(expansion_node)}')
      print(f'[EnvironmentLoop] begin_episode() took {time.time() - t0}s.')

      t0 = time.time()
      timestep, needs_reset, episode_logs = self.gc_rollout(
        timestep._replace(step_type=dm_env.StepType.FIRST),
        np.asarray(expansion_node), episode_logs, use_random_actions=False
      )

      # Log the attempted edge and whether it was successful.
      # Each gc-rollout corresponds to many edges, so we need to log them all.
      current_one_hot_idx, _ = self.convert2onehots(timestep.observation.goals)
      expansion_node_one_hot_idx, _ = self.convert2onehots(np.asarray(expansion_node))
      expansion_node_one_hot_idx = expansion_node_one_hot_idx[0]  # Expansion node will surely be 1-hot.
      attempted_edges = [(current_hash, expansion_node_one_hot_idx, bool(timestep.reward > 0.)) for current_hash in current_one_hot_idx]

      delta = timestep.reward - self._goal_achievement_rates[expansion_node]
      self._goal_pursual_counts[expansion_node] += 1
      self._goal_achievement_rates[expansion_node] += (delta / self._goal_pursual_counts[expansion_node])
      print(f'Success rate for {self._binary2info(expansion_node)} is {self._goal_achievement_rates[expansion_node]} ({self._goal_pursual_counts[expansion_node]})')
      print(f'[EnvironmentLoop] GC Rollout took {time.time() - t0}s.')

      reached_target = reached_expansion_node(timestep, expansion_node)
      self._node2successes[expansion_node].append(reached_target)

      overall_attempted_edges.extend(attempted_edges)

      if not needs_reset and reached_target and \
        random.random() < self._pure_exploration_probability:
        print(f'[EnvironmentLoop] Reached {self._binary2info(expansion_node)}; starting pure exploration rollout.')
        timestep, needs_reset, episode_logs = self.exploration_rollout(
          timestep, episode_logs, trajectory_key='exploration_trajectory')
        print(f"[EnvironmentLoop] Length of exploration trajectory = {len(episode_logs['exploration_trajectory'])}")
    
    print(f"Episode traj len: {len(episode_logs['episode_trajectory'])}. Num attempted edges: {len(overall_attempted_edges)}")

    return episode_logs, overall_attempted_edges, expansion_node

  def run_episode(self, is_warmup_episode: bool) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Args:
      is_warmup_episode: do pure exploration when this is true.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    episode_start_time = time.time()
    
    episode_logs = {}
    episode_logs['select_action_durations'] = []
    episode_logs['env_step_durations'] = []
    episode_logs['episode_steps'] = 0
    episode_logs['episode_trajectory'] = []
    episode_logs['exploration_trajectory'] = []
    episode_logs['episode_return'] = tree.map_structure(
      _generate_zeros_from_spec, self._environment.reward_spec())
    
    new_hash2goals = {}
    attempted_edges = []
    env_reset_start = time.time()
    timestep = self._environment.reset()
    env_reset_duration = time.time() - env_reset_start
    start_state = copy.deepcopy(timestep.observation)
    expansion_node = tuple(timestep.observation.goals)
    self._start_ts = timestep

    # TODO(ab/mm): make sure UVFA can handle this dtype
    if self._goal_space_manager:
      t0 = time.time()
      print(f'[EnvironmentLoop] Getting goal set from GSM.')
      self.count_dict = self._goal_space_manager.get_count_dict()
      self.goal_dict = self._goal_space_manager.get_goal_dict()
      print(f'[EnvironmentLoop] Took {time.time() - t0}s to get goal set from GSM.')
    else:
      print(f'[EnvironmentLoop] self._goal_space_manager is None.')

    if not is_warmup_episode:
      t0 = time.time()
      episode_logs, attempted_edges, expansion_node = self.episodic_rollout(timestep, episode_logs)
      print(f'[EnvironmentLoop] Took {time.time() - t0}s to get episodic_rollout.')
    else:
      t0 = time.time()
      _, _, episode_logs = self.exploration_rollout(timestep, episode_logs, 'episode_trajectory')
      print(f'[EnvironmentLoop] Took {time.time() - t0}s to get exploration_rollout.')

    # Extract new goals to add to the goal-space.
    explore_traj_key = 'exploration_trajectory' if not is_warmup_episode else 'episode_trajectory'

    if (episode_logs[explore_traj_key] or is_warmup_episode) and not self._is_evaluator:
      new_hash2goals = self.extract_new_goals(episode_logs[explore_traj_key])
    
    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_logs['episode_steps'])

    counts = self.get_logging_counts_dict(counts)

    ############################################################
    # Stream the episodic trajectory to the goal space manager.
    if self._goal_space_manager is not None and not self._is_evaluator:
      t0 = time.time()
      filtered_trajectory = episode_logs['episode_trajectory'] + episode_logs['exploration_trajectory']

      # TODO(ab/mm): HACK - when we discover goals incrementally, we need to uncomment this.
      # self.filter_achieved_goals(
      #   new_hash2goals,
      #   episode_logs['episode_trajectory'] + episode_logs['exploration_trajectory']
      # )
      t1 = time.time()

      if new_hash2goals:
        print(f'[EnvLoop] Found {len(new_hash2goals)} goals but ignoring them rn.')
        
      # Two interesting outputs: proto2counts, hash2proto
      extracted_results = self.extract_achieved_goals(filtered_trajectory, attempted_edges)
      
      # HER.
      if not is_warmup_episode and not self._is_evaluator:
        self.goal_space_hindsight_replay(
          start_state, episode_logs['episode_trajectory'], extracted_results['hash2proto'])
      
      self._goal_space_manager.update(
        hash2proto=extracted_results['hash2proto'],
        hash2count=extracted_results['proto2count'],
        edge2success=extracted_results['hash_pair_to_success'],
        hash2obs=extracted_results['proto2obs']
      )

      print(f'Took {t1 - t0}s to filter achieved goals')
      print(f'Took {time.time() - t1}s to stream achieved goals to the GSM')
    
    # Collect the results and combine with counts.
    steps_per_second = episode_logs['episode_steps'] / (time.time() - episode_start_time)
    result = {
      'episode_length': episode_logs['episode_steps'],
      'episode_return': episode_logs['episode_return'],
      'steps_per_second': steps_per_second,
      'env_reset_duration_sec': env_reset_duration,
      'select_action_duration_sec': np.mean(episode_logs['select_action_durations']),
      'env_step_duration_sec': np.mean(episode_logs['env_step_durations']),
      'start_state': self._binary2info(start_state.goals),
      'expansion_node': self._binary2info(expansion_node)
    }
    result.update(counts)
    for observer in self._observers:
      result.update(observer.get_metrics())
    return result
  
  
  def exploration_rollout(self, ts: dm_env.TimeStep, episode_logs: Dict, trajectory_key: str):
    """Rollout the exploration policy and return dict mapping new goal hashes to their OARGs.

    Args:
      ts (dm_env.TimeStep): current timestep.
      episode_logs (Dict): logs keeping track of the episode so far.
      trajectory_key (str): which key to store the exploration traj in episode_logs.

    Returns:
      timestep (dm_env.TimeStep): ts at the end of the rollout.
      needs_reset (bool): currently, we run the exploration policy till episode end.
      episode_logs (Dict): updated log of episode so far.
    """

    if self._exploration_actor:
      print(f'About to roll out {self._exploration_actor}')
      obs: OARG = ts.observation
      new_obs = OARG(
        observation=obs.observation[:, :, :3],
        action=obs.action, reward=obs.reward, goals=obs.goals)
      ts = ts._replace(
        observation=new_obs,
        step_type=dm_env.StepType.FIRST)
      timestep, needs_reset, episode_logs = self.vanilla_policy_rollout(
        timestep=ts,
        episode_logs=episode_logs,
        trajectory_key=trajectory_key)
    else:
      print(f'Could have been rolling out the RND policy... {self._exploration_actor}')
      timestep, needs_reset, episode_logs = self.gc_rollout(
        timestep=ts._replace(step_type=dm_env.StepType.FIRST),
        goal=self.exploration_goal,
        episode_logs=episode_logs,
        use_random_actions=self._use_random_policy_for_exploration
      )

    if self._cfn is not None and episode_logs[trajectory_key]:
      print(f'[EnvironmentLoop] Updating CFN with {len(episode_logs[trajectory_key])} transitions.')
      self.update_cfn_ground_truth_counts(episode_logs[trajectory_key])

    print(f'[EnvironmentLoop] Ended exploration in {self._binary2info(timestep.observation.goals)}')
    assert needs_reset, 'Currently, we run the exploration policy till episode end.'
    
    return timestep, needs_reset, episode_logs

  def convert_trajectory_to_node_trajectory(
      self,
      trajectory: List[GoalBasedTransition],
      source_dict: Dict,
      maintain_index_distances: bool = False
  ) -> List[Tuple]:
    """Convert a list of transitions to a list of nodes that are in the source_dict."""
    def goals2key(goals: Tuple) -> Tuple:
      return tuple([int(g) for g in goals])

    def add(node_trajectory: List[Tuple], key: Tuple):
      item = key if key in source_dict else None
      if item is not None or maintain_index_distances:
        node_trajectory.append(item)
      return node_trajectory
    
    hash_trajectory: List[Tuple] = []
    
    hash_trajectory = add(
      hash_trajectory,
      goals2key(trajectory[0].ts.observation.goals)
    )
    
    for transition in trajectory:
      visited: OARG = transition.next_ts.observation
      visited_hash: Tuple = goals2key(visited.goals)
      hash_trajectory = add(hash_trajectory, visited_hash)

    return hash_trajectory

  def new_extract_new_goals(self, trajectory: List[GoalBasedTransition]) -> Dict:
    """Get all the proto-goal bits that are on in the most-novel obs."""
    oargs: List[OARG] = [trans.next_ts.observation for trans in trajectory]
    novelties: List[float] = [trans.intrinsic_reward for trans in trajectory]
    if novelties != [] and self.should_create_new_goals(novelties):
      most_salient_idx = np.argmax(novelties)
      most_salient_obs: OARG = oargs[most_salient_idx]
      most_salient_obs_bits = np.where(most_salient_obs.goals)
      return {b: most_salient_obs for b in most_salient_obs_bits if b not in self.goal_dict}
  
  def extract_new_goals(self, trajectory: List[GoalBasedTransition]) -> Dict:
    """Given the exploration trajectory, create a new node / salient event clf out of the most novel transition."""
    return {}  # TODO(ab/mm): implement this.

  def _select_action(self, timestep: dm_env.TimeStep, random_action: bool):
    """Generate an action from the agent's policy."""
    if random_action:
      n_actions = self._env_spec.actions.num_values
      return np.random.randint(0, n_actions, dtype=self._env_spec.actions.dtype)
    # import ipdb; ipdb.set_trace()
    
    return self._actor.select_action(timestep.observation)
  
  def vanilla_policy_rollout(
      self,
      timestep: dm_env.TimeStep,
      episode_logs: dict,
      trajectory_key: str = 'exploration_trajectory'
  ):
    """Rollout vanilla policy for exploration.

    Args:
        timestep (dm_env.TimeStep): starting timestep
        episode_logs (dict): for logging
        trajectory_key (str): what key to use in episode_logs

    Returns:
        timestep (dm_env.TimeStep): final ts
        done (bool): env needs reset
        logs (dict): updated episode logs
    """
    assert trajectory_key in ('exploration_trajectory', 'episode_trajectory')
    trajectory: List[GoalBasedTransition] = []
  
    # Make the first observation.
    self._exploration_actor.observe_first(timestep)
    for observer in self._observers:
      # Initialize the observer with the current state of the env after reset
      # and the initial timestep.
      observer.observe_first(self._environment, timestep)

    # Run an episode.
    while not timestep.last():
      # Book-keeping.
      episode_logs['episode_steps'] += 1

      # Generate an action from the agent's policy.
      select_action_start = time.time()
      action = self._exploration_actor.select_action(timestep.observation)
      episode_logs['select_action_durations'].append(time.time() - select_action_start)  

      # Step the environment with the agent's selected action.
      env_step_start = time.time()
      next_timestep = self._environment.step(action)
      episode_logs['env_step_durations'].append(time.time() - env_step_start)

      trajectory.append(
        GoalBasedTransition(
          ts=timestep,
          action=action,
          reward=next_timestep.reward,
          discount=next_timestep.discount,
          next_ts=next_timestep,
          pursued_goal=self.exploration_goal,
          # intrinsic reward corresponding to timestep.observation
          # TODO(ab): what should we initialize prev_intrinsic_reward to? Does it matter?
          intrinsic_reward=self._exploration_actor._state.prev_intrinsic_reward
        ))

      # Have the agent and observers observe the timestep.
      self._exploration_actor.observe(action, next_timestep=next_timestep)
      for observer in self._observers:
        # One environment step was completed. Observe the current state of the
        # environment, the current timestep and the action.
        observer.observe(self._environment, next_timestep, action)

      # Give the actor the opportunity to update itself.
      if self._should_update:
        self._exploration_actor.update()

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_logs['episode_return'] = tree.map_structure(
        operator.iadd,
        episode_logs['episode_return'],
        next_timestep.reward
      )
      timestep = next_timestep

    episode_logs[trajectory_key].extend(trajectory)

    return timestep, timestep.last(), episode_logs

  def gc_rollout(
    self,
    timestep: dm_env.TimeStep,
    goal: np.ndarray,
    episode_logs: dict,
    use_random_actions: bool = False
  ) -> Tuple[dm_env.TimeStep, bool, dict]:
    """Rollout goal-conditioned policy.

    Args:
        timestep (dm_env.TimeStep): starting timestep
        goal (np.ndarray): target binary goal vector
        episode_logs (dict): for logging
        use_random_actions (bool): if true, we also don't do any learning.

    Returns:
        timestep (dm_env.TimeStep): final ts
        done (bool): env needs reset
        logs (dict): updated episode logs
    """
    assert timestep.first(), timestep

    reached = False
    needs_reset = False
    duration = 0
    should_interrupt_option = False
    trajectory: List[GoalBasedTransition] = []

    def should_interrupt(current_ts: dm_env.TimeStep, duration: int, goal: np.ndarray) -> bool:
      """Interrupt the option if you are in a node and duration warrants timeout."""
      achieved_goal_hashes = self.convert2onehots(current_ts.observation.goals)[0]
      inside_graph = set(achieved_goal_hashes).intersection(set(self.goal_dict.keys()))
      return duration >= self._max_option_duration and \
        tuple(goal) != tuple(self.task_goal.goals) and \
        tuple(goal) != tuple(self.exploration_goal.goals) and \
        len(inside_graph) > 0
    
    timestep = self.augment_ts_with_goal(
      timestep,
      goal,
      'concat' if timestep.observation.observation.shape[-1] == 3 else 'relabel'
    )
    
    # Make the first observation. This also resets the hidden state.
    if not use_random_actions:
      self._actor.observe_first(timestep)
      for observer in self._observers:
        # Initialize the observer with the current state of the env after reset
        # and the initial timestep.
        observer.observe_first(self._environment, timestep)

    print(f'[EnvironmentLoop] Starting gc-rollout towards g={self._binary2info(goal)}.')
    while not needs_reset and not reached and not should_interrupt_option:
      # Book-keeping.
      episode_logs['episode_steps'] += 1
      duration += 1

      # Generate an action from the agent's policy.
      select_action_start = time.time()
      action = self._select_action(timestep, random_action=use_random_actions)
      episode_logs['select_action_durations'].append(time.time() - select_action_start)

      # Step the environment with the agent's selected action.
      env_step_start = time.time()
      next_timestep = self._environment.step(action)
      episode_logs['env_step_durations'].append(time.time() - env_step_start)
      
      needs_reset = next_timestep.last()  # timeout or terminal state

      # Save terminal states for transmitting to GSM later.
      extrinsic_reward = next_timestep.reward.copy()
      extrinsic_discount = next_timestep.discount.copy()
      
      # Augment the ts with the current goal being pursued
      next_timestep = self.augment_ts_with_goal(next_timestep, goal, 'concat')
      
      trajectory.append(
        GoalBasedTransition(
          ts=timestep,
          action=action,
          reward=extrinsic_reward,
          discount=extrinsic_discount,
          next_ts=next_timestep,
          pursued_goal=goal
      ))

      should_interrupt_option = should_interrupt(next_timestep, duration, goal)

      if should_interrupt_option and next_timestep.reward < 1:
        next_timestep = truncation(next_timestep)

      # Have the agent and observers observe the timestep.
      if not use_random_actions:
        self._actor.observe(action, next_timestep=next_timestep)
        for observer in self._observers:
          # One environment step was completed. Observe the current state of the
          # environment, the current timestep and the action.
          observer.observe(self._environment, next_timestep, action)

      # Give the actor the opportunity to update itself.
      if self._should_update:
        self._actor.update()

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_logs['episode_return'] = tree.map_structure(
        operator.iadd,
        episode_logs['episode_return'],
        extrinsic_reward
      )
      timestep = next_timestep
      
      reached = timestep.last() and timestep.reward > 0
      if duration % 100 == 0 or needs_reset or reached or should_interrupt_option:
        print(f'\tT={duration} needs_reset={needs_reset} reached={reached} should_interrupt={should_interrupt_option}')

      # Update the episode count dict
      hashes, _ = self.convert2onehots(timestep.observation.goals)

      for key in hashes:
        if key in self.count_dict:
          self.count_dict[key] += 1
        else:
          self.count_dict[key] = 1

    episode_logs['episode_trajectory'].extend(trajectory)

    print(f'Goal={self._binary2info(goal)} Achieved={self._binary2info(timestep.observation.goals)} R={timestep.reward} T={duration}')

    return timestep, needs_reset, episode_logs

  def filter_achieved_goals(
    self,
    discovered_goals: Dict,
    trajectory: List[GoalBasedTransition],
  ) -> List[GoalBasedTransition]:
    """Filter the trajectory based on goals in the old + new goal space."""

    goal_space = {**self.goal_dict, **discovered_goals}
    filtered_transitions: List[GoalBasedTransition] = []
    for transition in trajectory:
      if tuple(transition.next_ts.observation.goals) in goal_space:
        filtered_transitions.append(transition)
    return filtered_transitions
  
  def stream_achieved_goals_to_gsm(
    self,
    trajectory: List[GoalBasedTransition],
    attempted_edges: List[Tuple[OARG, OARG, bool]],
    expansion_node_new_node_pairs: List[Tuple[Tuple, Tuple]]
  ):
    """Send the goals achieved during the episode to the GSM."""
    def goals2key(goals: Tuple) -> Tuple:
      return tuple([int(g) for g in goals])
    
    def obs2key(obs: OARG) -> Tuple:
      return tuple([int(g) for g in obs.goals])
  
    def get_key(ts: dm_env.TimeStep) -> Tuple:
      """Extract a hashable from a transition."""
      return tuple([int(g) for g in ts.observation.goals])
    
    def extract_counts() -> Dict:
      """Return a dictionary mapping goal to visit count in episode."""
      achieved_goals = [get_key(trans.next_ts) for trans in trajectory]
      return dict(collections.Counter(achieved_goals))
    
    def attempted2counts() -> Dict:
      attempted_hashes = [(obs2key(x), obs2key(y)) for x, y, _ in attempted_edges]
      return dict(collections.Counter(attempted_hashes))
    
    def edge2success() -> Dict:
      return {(obs2key(src), obs2key(dest)): bool(success) for src, dest, success in attempted_edges}
    
    def extract_discounts() -> Dict:
      """Terminal states in the MDP have a discount of 0."""
      return {get_key(trans.next_ts): trans.discount.item() for trans in trajectory}
    
    def extract_goal_to_obs() -> Dict:
      """Return a dictionary mapping goal hash to the OAR when goal was achieved."""
      return {
        get_key(transition.next_ts): 
        # We convert to jnp b/c courier cannot handle np arrays
        # TODO(ab): This assumes that the 1st 3 channels are the obs
        (jnp.asarray(transition.next_ts.observation.observation[:, :, :3]),
         int(transition.next_ts.observation.action),
         float(transition.reward.item())  # TODO(ab): is this the correct reward?
        ) 
        for transition in trajectory
      }
    
    hash2count = extract_counts()
    hash2obs = extract_goal_to_obs()
    edge2count = attempted2counts()
    hash2discount = extract_discounts()
    edge2successes = edge2success()

    # Filter out pairs in which both nodes are the same.
    expansion_node_new_node_pairs = [
      (goals2key(expansion_node), goals2key(new_node))
      for expansion_node, new_node in expansion_node_new_node_pairs
      if expansion_node != new_node
    ]

    # futures allows us to update() asynchronously
    print(f'[EnvironmentLoop] expansion_node_new_node_pairs: {expansion_node_new_node_pairs}')
    t0 = time.time()
    self._goal_space_manager.update(
      hash2obs,
      hash2count,
      edge2count,
      hash2discount,
      expansion_node_new_node_pairs,
      edge2successes
    )
    print(f'[EnvironmentLoop] Took {time.time() - t0}s to update the GSM.')
  
  def replay_trajectory_with_new_goal(
    self,
    trajectory: List[GoalBasedTransition],
    hindsight_goal: np.ndarray,
    update_hidden_state: bool = False
  ):
    """Replay the same trajectory with a goal achieved in hindsight.

    Args:
        trajectory (List): trajectory from pursuing one goal.
        hindsight_goal (np.ndarray): hindsight goal (1-hot) for learning.
        update_hidden_state (bool): whether to forward pass through pi(s,g).
          Need to update h_t if NOT using a GoalBasedQNetowork architecture.
    """
      
    ts0 = trajectory[0].ts
    augmented_ts0 = self.augment_ts_with_goal(ts0, hindsight_goal, 'relabel')
    
    # terminate early if the 1st transition achieves the goal
    if augmented_ts0.reward == 1:
      return 
    
    self._actor.observe_first(augmented_ts0._replace(step_type=dm_env.StepType.FIRST))
    
    for i, transition in enumerate(trajectory):
      augmented_ts = self.augment_ts_with_goal(transition.ts, hindsight_goal, 'relabel')
      if update_hidden_state:
        hindsight_action = self._actor.select_action(augmented_ts.observation)  # updates h_t
      augmented_next_ts = self.augment_ts_with_goal(transition.next_ts, hindsight_goal, 'relabel')
      reached = self.goal_reward_func(transition.next_ts.observation, hindsight_goal)[0]
      
      if reached:
        self._actor.observe(transition.action, termination(augmented_next_ts))
        break
      elif i == len(trajectory) - 1:
        self._actor.observe(transition.action, truncation(augmented_next_ts))
      else:
        self._actor.observe(transition.action, continuation(augmented_next_ts))

  def replay_trajectory_for_exploration_reward(self, trajectory: List[GoalBasedTransition]):

    prev_reward = 0.
    ts0 = trajectory[0].ts
    augmented_ts0 = self.exploration_goal_augment(ts0, prev_reward)

    self._actor.observe_first(augmented_ts0._replace(step_type=dm_env.StepType.FIRST))

    for i, transition in enumerate(trajectory):
      augmented_ts: dm_env.TimeStep = self.exploration_goal_augment(
        transition.ts, prev_reward)
      self._actor.select_action(augmented_ts.observation)  # updates h_t
      augmented_next_ts: dm_env.TimeStep = self.exploration_goal_augment(
        transition.next_ts, transition.intrinsic_reward)
      prev_reward = transition.intrinsic_reward
      if i == len(trajectory) - 1:
        self._actor.observe(transition.action, truncation(augmented_next_ts))
      else:
        self._actor.observe(transition.action, continuation(augmented_next_ts))

  @staticmethod
  def convert2onehots(achieved_protos: np.ndarray) -> Tuple[List[Tuple], jnp.ndarray]:
    """Given an observation's proto/goals, return the list of 1-hot vectors."""
    hot_idx = np.where(achieved_protos == 1)
    if hot_idx:
      hot_idx = hot_idx[0]
      n = len(hot_idx)
      one_hots = np.zeros(
        shape=(n, achieved_protos.shape[0]),
        dtype=achieved_protos.dtype)
      one_hots[range(n), hot_idx] = 1
      hashes = [(int(i),) for i in hot_idx.tolist()]
      return hashes, jnp.asarray(one_hots)
    return [], []

  def extract_achieved_goals(
      self,
      trajectory: List[GoalBasedTransition],
      attempted_edges: List[Tuple[NodeHash, NodeHash, bool]],
  ):
    """Extract goals that were achieved in the input trajectory."""
    
    def attempted2counts():  # TODO(ab/mm): adapt for proto-goal edges.
      attempted_hashes = [(x, y) for x, y, _ in attempted_edges]
      return dict(collections.Counter(attempted_hashes))
    
    def edge2success():
      # Mapping a pair of one-hot tuples to success bool.
      # E.g, ((1,), (3,)) -> True means that the edge from node 1 to node 3 was successful.
      return {(x, y): success for x, y, success in attempted_edges}
    
    proto2obs = {}
    hash2proto = {}
    proto2count = collections.defaultdict(int)
    proto2reward = collections.defaultdict(float)
    proto2discount = collections.defaultdict(float)

    for transition in trajectory:
      observation = transition.next_ts.observation
      prev_observation = transition.ts.observation
      proto_hashes, proto_one_hots = self.convert2onehots(observation.goals)
      prev_proto_hashes, _ = self.convert2onehots(prev_observation.goals)
      prev_proto_hashes = set(prev_proto_hashes)
      assert len(proto_hashes) == len(proto_one_hots)
      for key, one_hot in zip(proto_hashes, proto_one_hots):
        proto2obs[key] = (
          jnp.asarray(observation.observation[..., :3]),
          int(observation.action),
          float(observation.reward.item())
        )
        hash2proto[key] = one_hot
        
        # Increment count when transition causes the proto-goal to be achieved
        achieved = int(key not in prev_proto_hashes)
        proto2count[key] += achieved

        # Maintain max extrinsic reward corresponding to the proto-goal
        extrinsic_reward = transition.next_ts.reward or 0.
        proto2reward[key] = max(proto2reward[key], extrinsic_reward if achieved else 0.)
        proto2discount[key] = min(proto2discount[key], transition.next_ts.discount)

    hash_pair_to_success = edge2success()
    # hash_pair_to_attempted_counts = attempted2counts()

    return dict(
      proto2obs=proto2obs,
      hash2proto=hash2proto,
      proto2count=proto2count,
      proto2reward=proto2reward,
      proto2discount=proto2discount,
      hash_pair_to_success=hash_pair_to_success,
      #hash_pair_to_attempted_counts=hash_pair_to_attempted_counts,
    )
  
  def goal_space_hindsight_replay(self, start_state: OARG, trajectory: List[GoalBasedTransition], hash2proto: Dict):
    
    def goal_space_novelty_selection(num_goals_to_replay: int = 5) -> List[OARG]:
      counts = [self.count_dict[g] for g in goal_hashes]
      scores = np.asarray([1. / (1 + count) for count in counts])
        
      probs = scores2probabilities(scores)
      selected_indices = np.random.choice(
        range(len(goal_hashes)),
        p=probs,
        size=min(num_goals_to_replay, len(goal_hashes)),
        replace=False
      )
      return [triggered_goals[goal_hashes[i]] for i in selected_indices]
    
    start_time = time.time()
    
    # Map from tuple hash to 1-hot vector that we can condition the UVFA on.
    triggered_goals = {g: np.asarray(hash2proto[g]) for g in hash2proto if g in self.goal_dict}

    # Filter out the goals from triggered_goals that are reached in the start_state
    triggered_goals = {k: v for k, v in triggered_goals.items() if not self._reached(start_state.goals, v)}

    goal_hashes = list(triggered_goals.keys())

    if triggered_goals:
      hindsight_goals = goal_space_novelty_selection()
      for hindsight_goal in hindsight_goals:
        print(f'[HER] replaying wrt to {self._binary2info(hindsight_goal)}')
        assert not self._reached(start_state.goals, hindsight_goal), self._binary2info(hindsight_goal)
        self.replay_trajectory_with_new_goal(trajectory, hindsight_goal)

      # TODO(ab/mm): implement task goal feature.
      if self._always_learn_about_task_goal and \
          self._has_seen_task_goal and self._task_goal_probability > 0:
        print(f'[HER] replaying wrt to task goal {np.where(self.task_goal.goals)}')
        self.replay_trajectory_with_new_goal(trajectory, self.task_goal)

    print(f'HER took {time.time() - start_time}s')

  def get_logging_counts_dict(self, counts: Dict) -> Dict:
    """Return which counts to log, we are omitting CFN counts to prevent race conditions."""
    keys = [
      'actor_steps',
      'actor_episodes',
      'learner_steps',
      'evaluator_episodes',
      'evaluator_steps',
    ]
    return {key: counts.get(key, 0) for key in keys}

  # TODO(ab): fix this function and move to the GSM.
  def visualize_goal_space(
      self, ts0: dm_env.TimeStep, node2success: Dict, current_episode: int):
    node2rate = {}
    node2attempts = {}
    for node in node2success:
      success_curve = node2success[node]
      node2rate[node] = sum(success_curve) / len(success_curve)
      node2attempts[node] = len(success_curve)
    x_locations = []
    y_locations = []
    num_attempts = []
    success_rates = []
    for node in node2rate:
      if node != self.exploration_hash and node != self.task_goal_hash:
        x_locations.append(node[0])
        y_locations.append(node[1])
        success_rates.append(node2rate[node])
        num_attempts.append(node2attempts[node])

    # Visualize all graph nodes, not just descendants.
    start_node = tuple([int(g) for g in ts0.observation.goals])
    hash2oarg = self.goal_dict
    
    descendants = self._goal_space_manager.get_descendants(start_node)
    
    graph_x = [goal_hash[0] for goal_hash in hash2oarg]
    graph_y = [goal_hash[1] for goal_hash in hash2oarg]
    descendants_x = [goal_hash[0] for goal_hash in descendants]
    descendants_y = [goal_hash[1] for goal_hash in descendants]

    filename = f'actor_{self._actor_id}_expansion_nodes_episode_{current_episode}.png'

    plt.figure(figsize=(16, 16))
    plt.subplot(221)
    plt.scatter(x_locations, y_locations, c=success_rates)
    plt.colorbar()
    plt.title('Success Rate')

    plt.subplot(222)
    plt.scatter(x_locations, y_locations, c=num_attempts)
    plt.colorbar()
    plt.title('Num Attempts')

    plt.subplot(223)
    plt.scatter(graph_x, graph_y, label='Graph nodes', c='black')
    plt.scatter(descendants_x, descendants_y, label='Descendants', c='red')
    plt.title('Global and Local Graph')
    plt.legend()

    plt.suptitle(f'Episode {current_episode}')
    plt.savefig(os.path.join(self._target_node_plot_dir, filename))
    plt.close()

  def update_cfn_ground_truth_counts(self, trajectory: List[GoalBasedTransition]):

    def get_key(ts: dm_env.TimeStep) -> Tuple:
      return tuple([int(g) for g in ts.observation.goals])

    def extract_counts() -> dict:
      """Return a dictionary mapping goal to visit count in episode."""
      hashes = [get_key(trans.next_ts) for trans in trajectory]
      return dict(collections.Counter(hashes))

    def extract_goal_to_obs() -> dict:
      """Return a dictionary mapping goal hash to the OAR when goal was achieved."""
      return {
        get_key(transition.next_ts): 
        # We convert to jnp b/c courier cannot handle np arrays
        # TODO(ab/sl): This assumes that the 1st 3 channels are the obs
        (jnp.asarray(transition.next_ts.observation.observation[:, :, :3]),
         int(transition.next_ts.observation.action),
         float(transition.reward.item()))  # This should be r_e + beta * r_int
        for transition in trajectory
      }
    
    hash2counts = extract_counts()
    hash2obs = extract_goal_to_obs()

    self._cfn.futures.update_ground_truth_counts(hash2obs, hash2counts)

  def run(
      self,
      num_episodes: Optional[int] = None,
      num_steps: Optional[int] = None,
      n_warmup_episodes: int = 5
  ) -> int:
    """Perform the run loop.

    Run the environment loop either for `num_episodes` episodes or for at
    least `num_steps` steps (the last episode is always run until completion,
    so the total number of steps may be slightly more than `num_steps`).
    At least one of these two arguments has to be None.

    Upon termination of an episode a new episode will be started. If the number
    of episodes and the number of steps are not given then this will interact
    with the environment infinitely.

    Args:
      num_episodes: number of episodes to run the loop for.
      num_steps: minimal number of steps to run the loop for.
      n_warmup_episodes: number of pure exploration episodes in the start.

    Returns:
      Actual number of steps the loop executed.

    Raises:
      ValueError: If both 'num_episodes' and 'num_steps' are not None.
    """

    if not (num_episodes is None or num_steps is None):
      raise ValueError('Either "num_episodes" or "num_steps" should be None.')

    def should_terminate(episode_count: int, step_count: int) -> bool:
      return ((num_episodes is not None and episode_count >= num_episodes) or
              (num_steps is not None and step_count >= num_steps))

    episode_count: int = 0
    step_count: int = 0
    with signals.runtime_terminator():
      while not should_terminate(episode_count, step_count):
        episode_start = time.time()
        is_warmup_episode =  episode_count < n_warmup_episodes and self._goal_space_manager is not None
        result = self.run_episode(is_warmup_episode)
        result = {**result, **{'episode_duration': time.time() - episode_start}}
        episode_count += 1
        step_count += int(result['episode_length'])
        # Log the given episode results.
        self._logger.write(result)

        # TODO(ab/mm): implement this.
        # if self._actor_id == 1 and episode_count % 10 == 0:
        #   self.visualize_goal_space(self._start_ts, self._node2successes, episode_count)

    return step_count


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)
