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
from acme.jax.variable_utils import VariableClient
from acme.goal_sampler import GoalSampler

import dm_env
from dm_env import specs
import numpy as np
import tree
import random
import copy
import collections
import ipdb
import jax.numpy as jnp


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
      max_option_duration: int = 400,
      recompute_expansion_node_on_interruption: bool = False,
      decentralized_planner_kwargs: Dict = {},
      use_gsm_var_client: bool = False
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
    self._gsm_variable_client = VariableClient(
      goal_space_manager,
      key='gsm_variables',
      update_period=1000
    ) if goal_space_manager and (use_gsm_var_client or decentralized_planner_kwargs) else None
    self._use_gsm_var_client = use_gsm_var_client
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
    self._recompute_expansion_node_on_interruption = recompute_expansion_node_on_interruption

    self.goal_dict = {}
    self.count_dict = {}
    self._n_courier_errors = 0
    self._has_seen_task_goal = False
    self._is_evaluator = is_evaluator
    self._edges = set()
    self._reward_dict = {}
    self._discount_dict = {}
    self._bonus_dict = {}
    self._on_policy_edge_count = {}
    self._hash2idx = {}
    self._transition_matrix = np.zeros((1, 1))
    self._idx2hash = {}

    self._goal_achievement_rates = collections.defaultdict(float)
    self._goal_pursual_counts = collections.defaultdict(int)
    self._node2successes = collections.defaultdict(list)
    
    self._planner_failure_history = []

    self._env_spec = env_specs.make_environment_spec(self._environment)

    # GoalSampler params
    self._decentralized_planner_kwargs = decentralized_planner_kwargs

    # For debugging and visualizations
    self._start_ts = None
    self._episode_iter = itertools.count()

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

    # Exploration mode
    if (goal_hash == -1).all():
      return False

    dims = np.where(goal_hash >= 0)

    if np.any(current_hash == -1):
      ipdb.set_trace()  # Shouldn't happen

    return (current_hash[dims] == goal_hash[dims]).all()
    
  def goal_reward_func(self, current: OARG, goal: OARG) -> Tuple[bool, float]:
    """Is the goal achieved in the current state."""

    # Exploration mode
    if (goal.goals == -1).all():
      return False, self._get_intrinsic_reward(current)

    reached = self._reached(current.goals, goal.goals)
    return reached, float(reached)
  
  def _get_intrinsic_reward(self, state: OARG) -> float:
    """Novelty-based intrinsic reward associated with `state`."""

    r_int = 1.
    key = tuple(state.goals)

    if key in self.count_dict:
      count = self.count_dict[key]
      r_int = 1. / math.sqrt(count) if count > 0 else 1.

    return r_int

  def augment_ts_with_goal(
    self, timestep: dm_env.TimeStep, goal: OARG, method: str
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
    
  def augment_obs_with_goal(
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

  # TODO(ab): this should be based on which options are available at s_t?
  def _get_current_node(self, timestep: dm_env.TimeStep) -> Tuple:
    return tuple([int(g) for g in timestep.observation.goals])

  def _update_dicts_from_gsm(self):
    def _var_client_version():
      gsm_vars = self._gsm_variable_client.params
      self.goal_dict = gsm_vars[0]
      self.count_dict = gsm_vars[1]
      self._edges = gsm_vars[2]
      self._reward_dict = gsm_vars[3]
      self._discount_dict = gsm_vars[4]
      self._has_seen_task_goal = gsm_vars[5]
      if self._has_seen_task_goal:
        print(f'[EnvironmentLoop] Has seen task goal.')

    def _decentralized_planning_version():
      gsm_vars = self._gsm_variable_client.params
      self.goal_dict = gsm_vars[0]
      self.count_dict = gsm_vars[1]
      self._bonus_dict = gsm_vars[2]
      self._on_policy_edge_count = gsm_vars[3]
      self._reward_dict = gsm_vars[4]
      self._discount_dict = gsm_vars[5]
      self._hash2idx = gsm_vars[6]
      self._transition_matrix = gsm_vars[7]
      self._idx2hash = gsm_vars[8]
      self._edges = gsm_vars[9]
      self._has_seen_task_goal = self._goal_space_manager.get_has_seen_task_goal()
      if self._has_seen_task_goal:
        print(f'[EnvironmentLoop] Has seen task goal.')
    
    def _courier_version():
      self.count_dict = self._goal_space_manager.get_count_dict()
      self.goal_dict = self._goal_space_manager.get_goal_dict()
      self._has_seen_task_goal = self._goal_space_manager.get_has_seen_task_goal()
      self._edges = self._goal_space_manager.get_on_policy_edges()
      self._reward_dict, self._discount_dict = self._goal_space_manager.get_extrinsic_reward_dicts()
      if self._has_seen_task_goal:
        print(f'[EnvironmentLoop] Has seen task goal.')

    t0 = time.time()
    try:
      if self._decentralized_planner_kwargs:
        _decentralized_planning_version()
      elif self._use_gsm_var_client:
        _var_client_version()
      else:
        _courier_version()
    except Exception as e:  # If error, keep the old stale copy of the dicts
      self._n_courier_errors += 1
      print(f'[GoalSampler] Warning: Courier error # {self._n_courier_errors}. Exception: {e}')
    print(f'[EnvironmentLoop] Updating dicts from GSM took {time.time() - t0}s.')
  
  def episodic_rollout(
    self,
    timestep: dm_env.TimeStep,
    episode_logs: Dict
  ) -> Tuple[List[GoalBasedTransition], List[Tuple[OARG, OARG]], Tuple]:
    """Single episode rollout of the DSG algorithm."""
    
    def reached_expansion_node(ts: dm_env.TimeStep, node: Tuple) -> bool:
      if node is not None:
        obs = ts.observation
        return self._reached(obs.goals, np.asarray(node, dtype=obs.goals.dtype))
      return False
    
    needs_reset = False
    expansion_node = None
    overall_attempted_edges: List[Tuple[OARG, OARG, bool]] = []
    
    if self._goal_space_manager:
      self._update_dicts_from_gsm()
    
    while not needs_reset:
      t0 = time.time()
      abstract_policy = {}
      expansion_node = tuple(self.task_goal.goals)
      current_node = self._get_current_node(timestep)
      if self._goal_space_manager and self._decentralized_planner_kwargs:
        goal_sampler = GoalSampler(
          self.goal_dict,
          self.count_dict,
          self._bonus_dict,
          self._on_policy_edge_count,
          self._reward_dict,
          self._discount_dict,
          self._hash2idx,
          self._transition_matrix,
          self._idx2hash,
          self._edges,
          task_goal_probability=0.1,
          task_goal=self.task_goal,
          exploration_goal=self.exploration_goal,
          exploration_goal_probability=0.,
          **self._decentralized_planner_kwargs)
        expansion_node = goal_sampler.begin_episode(current_node)
        if goal_sampler._amdp:
          abstract_policy = goal_sampler._amdp.get_policy()
      elif self._goal_space_manager:
        planning_results = self._goal_space_manager.begin_episode(current_node)
        if planning_results:
          expansion_node, abstract_policy = planning_results
      print(f'[EnvironmentLoop] Expansion Node: {expansion_node}')
      print(f'[EnvironmentLoop] begin_episode() took {time.time() - t0}s.')

      subgoal_sampler = SubgoalSampler(
        abstract_policy,
        self.goal_dict,
        task_goal_probability=self._task_goal_probability if self._has_seen_task_goal else 0.,
        task_goal=self.task_goal,
        exploration_goal_probability=0.,
        exploration_goal=self.exploration_goal,
        edges=self._edges,
        expansion_node=expansion_node,
        reward_dict=self._reward_dict,
        discount_dict=self._discount_dict,
        sampling_method='amdp' if self._goal_space_manager else 'task',
        default_behavior=self._planner_backup_strategy if self._goal_space_manager else 'task')

      subgoal_seq = subgoal_sampler.get_subgoal_sequence(current_node, expansion_node)
      print(f'[EnvironmentLoop] Subgoal seq to {expansion_node}: {subgoal_seq}')

      # Planner failure: is the subgoal_seq all start-states?
      planner_failure = subgoal_seq and all([s == current_node for s in subgoal_seq])
      self._planner_failure_history.append(planner_failure)

      t0 = time.time()
      timestep, needs_reset, attempted_edges, episode_logs = self.in_graph_rollout(
        timestep, expansion_node, subgoal_sampler, episode_logs, reached_expansion_node
      )
      print(f'[EnvironmentLoop] In-Graph Rollout took {time.time() - t0}s.')

      reached_target = reached_expansion_node(timestep, expansion_node)
      self._node2successes[expansion_node].append(reached_target)

      overall_attempted_edges.extend(attempted_edges)

      if not needs_reset and reached_target and \
        random.random() < self._pure_exploration_probability:
        
        overall_attempted_edges.append((timestep.observation, self.exploration_goal, True))
        print(f'[EnvironmentLoop] Reached {expansion_node}; starting pure exploration rollout.')
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
    env_reset_start = time.time()
    timestep = self._environment.reset()
    env_reset_duration = time.time() - env_reset_start
    start_state = copy.deepcopy(timestep.observation)
    expansion_node = tuple(timestep.observation.goals)
    self._start_ts = timestep
    
    if not is_warmup_episode:
      episode_logs, attempted_edges, expansion_node = self.episodic_rollout(timestep, episode_logs)
    else:
      t0 = time.time()
      if self._goal_space_manager:
        self._update_dicts_from_gsm()
      attempted_edges = [(timestep.observation, self.exploration_goal, True)]
      _, _, episode_logs = self.exploration_rollout(timestep, episode_logs, 'episode_trajectory')

    # Extract new goals to add to the goal-space.
    explore_traj_key = 'exploration_trajectory' if not is_warmup_episode else 'episode_trajectory'

    if (episode_logs[explore_traj_key] or is_warmup_episode) and not self._is_evaluator:
      new_hash2goals = self.extract_new_goals(episode_logs[explore_traj_key])
      
    # HER.
    if not is_warmup_episode and not self._is_evaluator:
      self.hingsight_experience_replay(start_state, episode_logs['episode_trajectory'])
    
    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_logs['episode_steps'])

    counts = self.get_logging_counts_dict(counts)
    
    # Stream the episodic trajectory to the goal space manager.
    if self._goal_space_manager is not None and not self._is_evaluator:
      t0 = time.time()
      filtered_trajectory = self.filter_achieved_goals(
        new_hash2goals,
        episode_logs['episode_trajectory'] + episode_logs['exploration_trajectory']
      )
      t1 = time.time()

      # TODO(ab): Connect them all together (n^2).
      # expansion_node_new_node_pairs = [(expansion_node, new_node) for new_node in new_hash2goals]
      expansion_node_new_node_pairs = []

      if new_hash2goals:

        # Convert exploration trajectory to a list of new nodes. E.g, [n1, n2, n1, n1]
        new_node_trajectory = self.convert_trajectory_to_node_trajectory(
          episode_logs[explore_traj_key],
          new_hash2goals,
          maintain_index_distances=False
        )

        # Convert exploration trajectory to a list of known nodes
        old_node_trajectory = self.convert_trajectory_to_node_trajectory(
          episode_logs[explore_traj_key],
          self.goal_dict,
          maintain_index_distances=False
        )

        source_node = old_node_trajectory[-1] if old_node_trajectory else expansion_node

        assert None not in new_node_trajectory, new_node_trajectory

        deduplicated_new_node_trajectory = remove_duplicates_keep_last(new_node_trajectory)

        for i in range(len(deduplicated_new_node_trajectory) - 1):
          expansion_node_new_node_pairs.append((
            deduplicated_new_node_trajectory[i],
            deduplicated_new_node_trajectory[i + 1]
          ))

        # Connect a node from the existing graph to the first new node. 
        if deduplicated_new_node_trajectory and \
          source_node != deduplicated_new_node_trajectory[0]:
          expansion_node_new_node_pairs.append((
            source_node,
            deduplicated_new_node_trajectory[0]
          ))

        print(f'Edges to be added to the GSM: {expansion_node_new_node_pairs}')
      
      self.stream_achieved_goals_to_gsm(
        filtered_trajectory,
        attempted_edges,
        expansion_node_new_node_pairs
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
      'start_state': start_state.goals,
      'expansion_node': expansion_node
    }
    result.update(counts)
    for observer in self._observers:
      result.update(observer.get_metrics())
    return result

  def in_graph_rollout(
    self,
    timestep: dm_env.TimeStep,
    target_node: Tuple,
    subgoal_sampler: SubgoalSampler,
    episode_logs: Dict,
    termination_func: Callable[[dm_env.TimeStep, Tuple], bool],
  ) -> Tuple[dm_env.TimeStep, bool, List[Tuple[OARG, OARG]], Dict]:
    """Pick in-graph subgoals to go from the current timestep to target_node."""
    
    # Triple of (src, dest, success)
    attempted_edges: List[Tuple[OARG, OARG, bool]] = []
    needs_reset = False

    while not needs_reset and not termination_func(timestep, target_node):
      goal = subgoal_sampler(self._get_current_node(timestep))

      obs0 = copy.deepcopy(timestep.observation)

      timestep, needs_reset, episode_logs, interrupted = self.gc_rollout(
        timestep._replace(step_type=dm_env.StepType.FIRST),
        goal, episode_logs, use_random_actions=False
      )
      
      # assert timestep.last(), timestep

      # Log the attempted edge and whether it was successful.
      attempted_edges.append((obs0, goal, timestep.reward > 0.))

      key = tuple(goal.goals)
      delta = timestep.reward - self._goal_achievement_rates[key]
      self._goal_pursual_counts[key] += 1
      self._goal_achievement_rates[key] += (delta / self._goal_pursual_counts[key])
      print(f'Success rate for {key} is {self._goal_achievement_rates[key]} ({self._goal_pursual_counts[key]})')

      if interrupted and self._recompute_expansion_node_on_interruption:
        break

    return timestep, needs_reset, attempted_edges, episode_logs
  
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
      if self._actor_id == 1:
        self.visualize_exploration_trajectory(episode_logs[trajectory_key], next(self._episode_iter))
    else:
      print(f'Could have been rolling out the RND policy... {self._exploration_actor}')
      timestep, needs_reset, episode_logs, _ = self.gc_rollout(
        timestep=ts._replace(step_type=dm_env.StepType.FIRST),
        goal=self.exploration_goal,
        episode_logs=episode_logs,
        use_random_actions=self._use_random_policy_for_exploration
      )

    if self._cfn is not None and episode_logs[trajectory_key]:
      print(f'[EnvironmentLoop] Updating CFN with {len(episode_logs[trajectory_key])} transitions.')
      self.update_cfn_ground_truth_counts(episode_logs[trajectory_key])

    print(f'[EnvironmentLoop] Ended exploration in {timestep.observation.goals}')
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
  
  def extract_new_goals(self, trajectory: List[GoalBasedTransition]) -> Dict:
    
    def extract(
        visited_states: Dict,
        unseen_goal_hashes: List[Tuple],
        novelties: List[float],
        method: str = 'novelty_sampling'
    ) -> Dict:
      """Extract new goals from the interesting trajectory."""
      if method == 'all':
        return {g: visited_states[g] for g in unseen_goal_hashes}
      if method == 'max_novelty':
        most_novel_goal_hash: Tuple = unseen_goal_hashes[np.argmax(novelties)]
        most_novel_state: OARG = visited_states[most_novel_goal_hash]
        print(f'Most novel goal {most_novel_goal_hash} to be added to goal space.')

        return {most_novel_goal_hash: most_novel_state}
      if method == 'novelty_sampling':

        subgoals = {}
        most_novel_goal_hash: Tuple = unseen_goal_hashes[np.argmax(novelties)]
        most_novel_state: OARG = visited_states[most_novel_goal_hash]
        print(f'Most novel goal {most_novel_goal_hash} to be added to goal space.')

        # Extract traj segment between expansion node and most novel goal (excluding on both ends)
        most_novel_goal_idx = unseen_goal_hashes.index(most_novel_goal_hash)
        traj_segment = unseen_goal_hashes[:most_novel_goal_idx]

        if traj_segment:
          print(f'Saw {len(unseen_goal_hashes)} unseen goals, {len(traj_segment)} before the most novel one.')
          print(f'Traj segment: {traj_segment}')

          subgoals = dsc.sample_random_subgoals(
            traj_segment,
            visited_states,
            hash2int,
            num_subgoals=3,
            method='weighted'
          )

          print(f'Extracted {len(subgoals)} subgoals: {subgoals.keys()}')

        return {most_novel_goal_hash: most_novel_state, **subgoals}

      if method == 'dsc':
        value_table = dsc.construct_value_table(
          self._exploitation_networks,
          self._actor._params,
          self._actor._random_key,
          visited_states,
          lambda obs, goal: self.augment_obs_with_goal(obs, goal, 'concat')[0]
        )

      raise NotImplementedError(method)
    
    def should_create_new_goals(novelty_scores: List[float]) -> bool:
      """Should we create new goals from the trajectory?"""
      if self._novelty_threshold_for_goal_creation > 0:
        thresh = self._novelty_threshold_for_goal_creation
      else:
        reward_mean = self._exploration_actor._rnd_state.reward_mean
        reward_std = jnp.sqrt(self._exploration_actor._rnd_state.reward_var + 1e-4)
        thresh = reward_mean + (self._n_sigmas_threshold_for_goal_creation * reward_std)
      max_novelty = max(novelty_scores)
      should_create = max_novelty > thresh
      print(f'Novelty threshold: {thresh}. Should create? {max_novelty, should_create}')
      return should_create
    
    def filter_uncontrollable_goals(new_goals: List[Tuple]) -> List[Tuple]:
      """Filter out goals that are jumping/falling/dead/uncontrollable."""
      idx_to_filter = [3, 4, 5, 6]  # jumping, dead, falling, uncontrollable
      filtered_goals = []
      for g in new_goals:
        goal = np.asarray(g)
        if not np.any(goal[idx_to_filter]) and goal[0] > 5 and goal[0] <= 148:
          filtered_goals.append(g)
      return filtered_goals

    goal_space = self.goal_dict

    visited_states = {
      tuple(trans.next_ts.observation.goals): trans.next_ts.observation 
      for trans in trajectory
    }

    # Associate each hash with its most recently assigned intrinsic reward.
    hash2int = {
      tuple(trans.next_ts.observation.goals): trans.intrinsic_reward
      for trans in trajectory
    }
    
    # NOTE: Using hash for equality here.
    visited_hashes: List[Tuple] = list(visited_states.keys())
    new_goal_hashes: List[Tuple] = [g for g in visited_hashes if g not in goal_space]

    new_goal_hashes = filter_uncontrollable_goals(new_goal_hashes)

    print(f'Extracting new goals from {len(new_goal_hashes)} unseen goal hashes.')

    if new_goal_hashes:
      # novelty_scores: List[float] = [self._get_intrinsic_reward(visited_states[g]) \
      #                                for g in new_goal_hashes]
      novelty_scores: List[float] = [hash2int[g] for g in new_goal_hashes]
      
      if should_create_new_goals(novelty_scores):

        discovered_goals = extract(
          visited_states,
          new_goal_hashes,
          novelty_scores,
          method='novelty_sampling'
        )
        
        start_state_hash = tuple(self._start_ts.observation.goals)

        if start_state_hash not in goal_space:
          print(f'Adding start state {start_state_hash} to goal space.')
          return {
            **discovered_goals,
            start_state_hash: self._start_ts.observation
          }

        return discovered_goals

    return {}

  def _select_action(self, timestep: dm_env.TimeStep, random_action: bool):
    """Generate an action from the agent's policy."""
    if random_action:
      n_actions = self._env_spec.actions.num_values
      return np.random.randint(0, n_actions, dtype=self._env_spec.actions.dtype)
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

      # if self._environment.get_current_info({})['room_number'] == 0:
      #   ipdb.set_trace()

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
    self, timestep: dm_env.TimeStep, goal: OARG, episode_logs: dict,
    use_random_actions: bool = False
    ) -> Tuple[dm_env.TimeStep, bool, dict, bool]:
    """Rollout goal-conditioned policy.

    Args:
        timestep (dm_env.TimeStep): starting timestep
        goal (OARG): target obs
        episode_logs (dict): for logging
        use_random_actions (bool): if true, we also don't do any learning.

    Returns:
        timestep (dm_env.TimeStep): final ts
        done (bool): env needs reset
        logs (dict): updated episode logs
        interrupted (bool): whether the option was interrupted (timeout)
    """
    reached = False
    needs_reset = False
    duration = 0
    should_interrupt_option = False
    trajectory: List[GoalBasedTransition] = []

    def should_interrupt(current_ts: dm_env.TimeStep, duration: int, goal: OARG) -> bool:
      """Interrupt the option if you are in a node and duration warrants timeout."""
      return duration >= self._max_option_duration and \
        tuple(goal.goals) != tuple(self.task_goal.goals) and \
        tuple(goal.goals) != tuple(self.exploration_goal.goals) and \
        tuple(current_ts.observation.goals) in self.goal_dict
    
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

        if self._gsm_variable_client:
          self._gsm_variable_client.update()

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

      # Update the episode count dict
      key = tuple(timestep.observation.goals)
      
      if self.count_dict and key in self.count_dict:
        self.count_dict[key] += 1
      else:
        self.count_dict[key] = 1
      
    episode_logs['episode_trajectory'].extend(trajectory)
    
    print(f'Goal={goal.goals} Achieved={timestep.observation.goals} R={timestep.reward} T={duration}')

    return timestep, needs_reset, episode_logs, should_interrupt_option

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

    print(f'[EnvironmentLoop] expansion_node_new_node_pairs: {expansion_node_new_node_pairs}')

    t0 = time.time()

    # futures allows us to update() asynchronously
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
    hindsight_goal: OARG,
    update_hidden_state: bool = False
  ):
    """Replay the same trajectory with a goal achieved in hindsight.

    Args:
        trajectory (List): trajectory from pursuing one goal.
        hindsight_goal (OARG): hindsight goal for learning.
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
      
  def hingsight_experience_replay(
    self, start_state: OARG, trajectory: List[GoalBasedTransition]):
    """Learn about goal(s) achieved when following a diff goal."""

    def goal_space_novelty_selection(
      traj: List[GoalBasedTransition],
      num_goals_to_replay: int = 5,
      use_tabular_counts: bool = True
    ) -> List[OARG]:
      achieved_goals = {
        tuple(trans.next_ts.observation.goals): trans.next_ts.observation
        for trans in traj if tuple(trans.next_ts.observation.goals) in self.goal_dict
      }
      goal_hashes = list(achieved_goals.keys())

      # TODO(ab): When the number of goal_hashes is <= num_goals_to_replay and sampling_by_replacement is False,
      # we can skip computing scores/probs and return all the indices.
      # Another way to implement this would be to interpret the exploration bonus as a 
      # replay probability, i.e, with prob = bonus, replay achieved goal.
      # We would still need to cap the number of goals to replay and should sort by bonus.
      if len(goal_hashes) <= num_goals_to_replay:
        return [achieved_goals[g] for g in goal_hashes]

      if (not use_tabular_counts) and (self._exploration_actor is not None):
        achieved_oarg = OARG(
          observation=jnp.asarray([achieved_goals[g].observation[..., :3] for g in goal_hashes]),
          action=jnp.asarray([achieved_goals[g].action for g in goal_hashes]),
          reward=jnp.asarray([achieved_goals[g].reward for g in goal_hashes]),
          goals=jnp.asarray([achieved_goals[g].goals for g in goal_hashes])
        )
        # TODO(ab): this is specific to CFN right now
        scores = compute_cfn_reward(
          self._exploration_actor._rnd_state.params,
          self._exploration_actor._rnd_state.target_params,
          achieved_oarg,
          self._exploration_networks,
          self._exploration_actor._rnd_state.random_prior_mean,
          jnp.sqrt(self._exploration_actor._rnd_state.random_prior_var + 1e-12))
        scores = np.asarray(scores)
      else:
        counts = [self.count_dict[g] for g in goal_hashes]
        scores = np.asarray([1. / (1 + count) for count in counts])
        
      probs = scores2probabilities(scores)
      selected_indices = np.random.choice(
        range(len(goal_hashes)),
        p=probs,
        size=min(num_goals_to_replay, len(goal_hashes)),
        replace=False
      )
      return [achieved_goals[goal_hashes[i]] for i in selected_indices]

    def future_selection(traj: List[dm_env.TimeStep]) -> List[OARG]:
      start_idx = len(traj) // 2
      goal_idx = random.randint(start_idx, len(traj) - 1)
      goal_ts = traj[goal_idx]
      assert isinstance(goal_ts, dm_env.TimeStep)
      return [goal_ts.observation]
    
    def get_achieved_goals() -> List[dm_env.TimeStep]:
      """Filter out goals that are satisfied at s_t."""
      feasible_goals: List[dm_env.TimeStep] = []
      for transition in trajectory:
        if not self.goal_reward_func(
          start_state, transition.next_ts.observation)[0]:
          feasible_goals.append(transition.next_ts)
      return feasible_goals

    start_time = time.time()
    
    filtered_trajectory = get_achieved_goals()
    if filtered_trajectory:
      hindsight_goals = goal_space_novelty_selection(trajectory)
      for hindsight_goal in hindsight_goals:
        print(f'[HER] replaying wrt to {hindsight_goal.goals}')
        self.replay_trajectory_with_new_goal(trajectory, hindsight_goal)
      
      task_goal = self.task_goal

      if self._always_learn_about_task_goal and \
        self._has_seen_task_goal and self._task_goal_probability > 0:
        # not self.goal_reward_func(hindsight_goal, task_goal)[0] and \
        print(f'[HER] replaying wrt to task goal {task_goal.goals}')
        self.replay_trajectory_with_new_goal(trajectory, task_goal)

      if self._always_learn_about_exploration_goal:
        t0 = time.time()
        self.replay_trajectory_for_exploration_reward(trajectory)
        print(f'[HER] Took {time.time() - t0}s to replay for exploration rewards.')

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

  def visualize_exploration_trajectory(self, trajectory: List[GoalBasedTransition], episode: int):

    pos2bonuses = collections.defaultdict(list)
    executed_actions: List[int] = []

    for transition in trajectory:
      oarg = transition.ts.observation
      intrinsic_reward = transition.intrinsic_reward
      goal_hash = tuple(oarg.goals)
      x_position = goal_hash[0]
      y_position = goal_hash[1]
      pos2bonuses[(x_position, y_position)].append(intrinsic_reward)
      executed_actions.append(transition.action.item())

    x_positions = []
    y_positions = []
    exploration_bonuses = []
    for pos, bonuses in pos2bonuses.items():
      x_positions.append(pos[0])
      y_positions.append(pos[1])
      exploration_bonuses.append(np.mean(bonuses))

    executed_action_distribution = collections.Counter(executed_actions)
    plt.title(f'As: {executed_action_distribution}')
    plt.scatter(x_positions, y_positions, c=exploration_bonuses)
    plt.colorbar()
    plt.savefig(os.path.join(self._exploration_traj_dir,
                             f'exploration_traj_episode_{episode}.png'))
    plt.close()

  def visualize_n_planner_failures(self, episode):
    """Visualize the number of times the n-planner failed to find a plan."""
    # Use a circle-dash marker
    plt.plot(np.cumsum(self._planner_failure_history), marker='o', linestyle='-')
    plt.xlabel('Episode')
    plt.title(f'N-Planner Failures @ Episode {episode}')
    plt.savefig(os.path.join(self._n_planner_failures_dir,
                             f'n_planner_failures.png'))
    plt.close()

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
      if node and node != self.exploration_hash and node != self.task_goal_hash:
        x_locations.append(node[0])
        y_locations.append(node[1])
        success_rates.append(node2rate[node])
        num_attempts.append(node2attempts[node])

    # Visualize all graph nodes, not just descendants.
    start_node = tuple([int(g) for g in ts0.observation.goals])
    hash2oarg = self.goal_dict
    
    if self._decentralized_planner_kwargs:
      descendants = GoalSampler(
            *self._gsm_variable_client.params,
            task_goal_probability=self._task_goal_probability,
            task_goal=self.task_goal,
            exploration_goal=self.exploration_goal,
            exploration_goal_probability=0.,
            **self._decentralized_planner_kwargs).get_descendants(start_node)
    else:
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
      n_warmup_episodes: int = 20
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

        if self._actor_id == 1 and episode_count % 10 == 0:
          self.visualize_goal_space(self._start_ts, self._node2successes, episode_count)
          self.visualize_n_planner_failures(episode_count)

    return step_count


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)
