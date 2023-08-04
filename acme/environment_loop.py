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

import operator
import time
import math
from typing import List, Optional, Sequence, Tuple, Dict

from acme import core
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.utils import signals
from acme.wrappers.oar_goal import OARG
from acme.agents.jax.r2d2.gsm import GoalSpaceManager
from acme.goal_sampler import GoalSampler
from acme.utils.utils import GoalBasedTransition
from acme.utils.utils import termination, truncation, continuation

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
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      should_update: bool = True,
      label: str = 'environment_loop',
      observers: Sequence[observers_lib.EnvLoopObserver] = (),
      goal_space_manager: GoalSpaceManager = None,
      task_goal_probability: float = 0.1,
      always_learn_about_task_goal: bool = False,
      always_learn_about_exploration_goal: bool = False
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._actor = actor
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        label, steps_key=self._counter.get_steps_key())
    self._should_update = should_update
    self._observers = observers
    self._goal_space_manager = goal_space_manager
    self._task_goal_probability = task_goal_probability
    self._always_learn_about_task_goal = always_learn_about_task_goal
    self._always_learn_about_exploration_goal = always_learn_about_exploration_goal

    self._goal_sampler = GoalSampler(
      goal_space_manager,
      task_goal_probability,
      self.task_goal,
      self.exploration_goal,
      method='amdp' if self._goal_space_manager else 'task'
    )

    self.count_dict = {}

    self._goal_achievement_rates = collections.defaultdict(float)
    self._goal_pursual_counts = collections.defaultdict(int)
    
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
    
  def goal_reward_func(self, current: OARG, goal: OARG) -> Tuple[bool, float]:
    """Is the goal achieved in the current state."""

    # Exploration mode
    if (goal.goals == -1).all():
      state_key = tuple(current.goals)
      r_int = 1.
      if state_key in self.count_dict:
        count = self.count_dict[state_key]
        r_int = 1. / math.sqrt(count) if count > 0 else 1.
      return False, r_int
    
    dims = np.where(goal.goals >= 0)

    if np.any(current.goals == -1):
      ipdb.set_trace()

    reached = (current.goals[dims] == goal.goals[dims]).all()
    return reached, float(reached)

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

  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

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
    episode_logs['episode_return'] = tree.map_structure(
      _generate_zeros_from_spec, self._environment.reward_spec())
    
    needs_reset = False
    env_reset_start = time.time()
    timestep = self._environment.reset()
    env_reset_duration = time.time() - env_reset_start
    start_state = copy.deepcopy(timestep.observation)
    
    self._goal_sampler.begin_episode(timestep)
    self.count_dict = copy.deepcopy(self._goal_sampler.count_dict)
    
    # State-goal pairs.
    attempted_edges: List[Tuple[OARG, OARG]] = []
    
    while not needs_reset:
      goal = self._goal_sampler(timestep)

      attempted_edges.append((timestep.observation, goal))

      timestep, needs_reset, episode_logs = self.gc_rollout(
        timestep._replace(step_type=dm_env.StepType.FIRST), goal, episode_logs
      )
      
      assert timestep.last(), timestep

      key = tuple(goal.goals)
      delta = timestep.reward - self._goal_achievement_rates[key]
      self._goal_pursual_counts[key] += 1
      self._goal_achievement_rates[key] += (delta / self._goal_pursual_counts[key])
      print(f'Success rate for {key} is {self._goal_achievement_rates[key]} ({self._goal_pursual_counts[key]})')
      
    # HER
    t0 = time.time()
    self.hingsight_experience_replay(start_state, episode_logs['episode_trajectory'])
    print(f'Took {time.time() - t0}s to do HER')
    
    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_logs['episode_steps'])
    
    # Stream the episodic trajectory to the goal space manager.
    if self._goal_space_manager is not None:
      t0 = time.time()
      self.stream_achieved_goals_to_gsm(episode_logs['episode_trajectory'], attempted_edges)
      print(f'Took {time.time() - t0}s to stream achieved goals to the GSM')
      
      # Update the Q-function parameters of the GSM
      t0 = time.time()
      self._goal_space_manager.update_params()
      print(f'Took {time.time() - t0}s to update GSM Learner Params')
    
    # Collect the results and combine with counts.
    steps_per_second = episode_logs['episode_steps'] / (time.time() - episode_start_time)
    result = {
      'episode_length': episode_logs['episode_steps'],
      'episode_return': episode_logs['episode_return'],
      'steps_per_second': steps_per_second,
      'env_reset_duration_sec': env_reset_duration,
      'select_action_duration_sec': np.mean(episode_logs['select_action_durations']),
      'env_step_duration_sec': np.mean(episode_logs['env_step_durations']),
      'start_state': start_state.goals
    }
    result.update(counts)
    for observer in self._observers:
      result.update(observer.get_metrics())
    return result
  
  def gc_rollout(
    self, timestep: dm_env.TimeStep, goal: OARG, episode_logs: dict
    ) -> Tuple[dm_env.TimeStep, bool, dict]:
    """Rollout goal-conditioned policy.

    Args:
        timestep (dm_env.TimeStep): starting timestep
        goal (OARG): target obs
        episode_logs (dict): for logging

    Returns:
        timestep (dm_env.TimeStep): final ts
        done (bool): env needs reset
        logs (dict): updated episode logs
    """
    reached = False
    needs_reset = False
    trajectory: List[GoalBasedTransition] = []
    
    timestep = self.augment_ts_with_goal(
      timestep,
      goal,
      'concat' if timestep.observation.observation.shape[-1] == 3 else 'relabel'
    )
    
    # Make the first observation. This also resets the hidden state.
    self._actor.observe_first(timestep)
    for observer in self._observers:
      # Initialize the observer with the current state of the env after reset
      # and the initial timestep.
      observer.observe_first(self._environment, timestep)
    
    while not needs_reset and not reached:
      # Book-keeping.
      episode_logs['episode_steps'] += 1

      # Generate an action from the agent's policy.
      select_action_start = time.time()
      action = self._actor.select_action(timestep.observation)
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

      # Have the agent and observers observe the timestep.
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
        next_timestep.reward
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
    
    print(f'Goal={goal.goals} Achieved={timestep.observation.goals} R={timestep.reward}')

    return timestep, needs_reset, episode_logs
  
  def stream_achieved_goals_to_gsm(
    self,
    trajectory: List[GoalBasedTransition],
    attempted_edges: List[Tuple[OARG, OARG]]
  ):
    """Send the goals achieved during the episode to the GSM."""
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
      attempted_hashes = [(obs2key(x), obs2key(y)) for x, y in attempted_edges]
      return dict(collections.Counter(attempted_hashes))
    
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

    # futures allows us to update() asynchronously
    self._goal_space_manager.futures.update(
      hash2obs, hash2count, edge2count, hash2discount
    )
  
  def replay_trajectory_with_new_goal(
    self, trajectory: List[GoalBasedTransition], hindsight_goal: OARG):
    """Replay the same trajectory with a goal achieved in hindsight.

    Args:
        trajectory (List): trajectory from pursuing one goal.
        hindsight_goal (OARG): hindsight goal for learning.
    """
      
    ts0 = trajectory[0].ts
    augmented_ts0 = self.augment_ts_with_goal(ts0, hindsight_goal, 'relabel')
    
    # terminate early if the 1st transition achieves the goal
    if augmented_ts0.reward == 1:
      return 
    
    self._actor.observe_first(augmented_ts0._replace(step_type=dm_env.StepType.FIRST))
    
    for i, transition in enumerate(trajectory):
      augmented_ts = self.augment_ts_with_goal(transition.ts, hindsight_goal, 'relabel')
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
    def pick_hindsight_goal(traj: List[dm_env.TimeStep]) -> OARG:
      start_idx = len(traj) // 2
      goal_idx = random.randint(start_idx, len(traj) - 1)
      goal_ts = traj[goal_idx]
      assert isinstance(goal_ts, dm_env.TimeStep)
      return goal_ts.observation
    
    def get_achieved_goals() -> List[dm_env.TimeStep]:
      """Filter out goals that are satisfied at s_t."""
      feasible_goals: List[dm_env.TimeStep] = []
      for transition in trajectory:
        if not self.goal_reward_func(
          start_state, transition.next_ts.observation)[0]:
          feasible_goals.append(transition.next_ts)
      return feasible_goals
    
    filtered_trajectory = get_achieved_goals()
    if filtered_trajectory:
      hindsight_goal = pick_hindsight_goal(filtered_trajectory)
      print(f'[HER] replaying wrt to {hindsight_goal.goals}')
      self.replay_trajectory_with_new_goal(trajectory, hindsight_goal)
      
      task_goal = self.task_goal

      if self._always_learn_about_task_goal and \
        not self.goal_reward_func(hindsight_goal, task_goal)[0]:
        self.replay_trajectory_with_new_goal(trajectory, task_goal)

      if self._always_learn_about_exploration_goal:
        t0 = time.time()
        self.replay_trajectory_for_exploration_reward(trajectory)
        print(f'[HER] Took {time.time() - t0}s to replay for exploration rewards.')

  def run(
      self,
      num_episodes: Optional[int] = None,
      num_steps: Optional[int] = None,
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
        result = self.run_episode()
        result = {**result, **{'episode_duration': time.time() - episode_start}}
        episode_count += 1
        step_count += int(result['episode_length'])
        # Log the given episode results.
        self._logger.write(result)

    return step_count


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)
