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
from typing import List, Optional, Sequence, Tuple

from acme import core
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.utils import signals
from acme.wrappers.oar_goal import OARG

import dm_env
from dm_env import specs
import numpy as np
import tree
import random
import copy


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
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._actor = actor
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        label, steps_key=self._counter.get_steps_key())
    self._should_update = should_update
    self._observers = observers
    
  def goal_reward_func(self, current: OARG, goal: OARG) -> Tuple[bool, float]:
    """Is the goal achieved in the current state."""
    reached = (current.goals == goal.goals).all()
    return reached, float(reached)
    
  def select_goal(self, timestep, method='task') -> OARG:
    """Select a goal to pursue in the upcoming episode."""
    if method == 'task':
      task_goal_img = np.zeros_like(timestep.observation.observation)
      task_goal_features = np.array(
        [6, 6],  # TODO(ab): pass in the task goal features
        dtype=timestep.observation.goals.dtype)
      return OARG(
        task_goal_img,
        action=timestep.observation.action,  # doesnt matter
        reward=timestep.observation.reward,  # doesnt matter
        goals=task_goal_features)

  def augment_ts_with_goal(
    self, timestep: dm_env.TimeStep, goal: OARG, method: str):
    """Concatenate the goal to the current observation."""
    new_obs = self.augment(
      timestep.observation.observation,
      goal.observation, method=method)
    reached, reward = self.goal_reward_func(timestep.observation, goal)
    new_oarg = OARG(
      observation=new_obs,  # pursued goal
      action=timestep.observation.action,
      reward=np.array(reward, dtype=np.float32),
      goals=timestep.observation.goals  # achieved goals
    )
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
    select_action_durations: List[float] = []
    env_step_durations: List[float] = []
    episode_steps: int = 0
    episode_trajectory : List = []

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        self._environment.reward_spec())
    # TODO(ab): only do this if we timed out, not if we achieved the goal
    env_reset_start = time.time()
    timestep = self._environment.reset()
    env_reset_duration = time.time() - env_reset_start
    
    start_state = copy.deepcopy(timestep.observation)
    
    goal = self.select_goal(timestep)
    timestep = self.augment_ts_with_goal(timestep, goal, 'concat')
    
    # Make the first observation.
    self._actor.observe_first(timestep)
    for observer in self._observers:
      # Initialize the observer with the current state of the env after reset
      # and the initial timestep.
      observer.observe_first(self._environment, timestep)

    # Run an episode; terminate on goal achievement or timeout
    while not timestep.last():
      # Book-keeping.
      episode_steps += 1

      # Generate an action from the agent's policy.
      select_action_start = time.time()
      action = self._actor.select_action(timestep.observation)
      select_action_durations.append(time.time() - select_action_start)

      # Step the environment with the agent's selected action.
      env_step_start = time.time()
      next_timestep = self._environment.step(action)
      env_step_durations.append(time.time() - env_step_start)
      
      # Augment the ts with the current goal being pursued
      next_timestep = self.augment_ts_with_goal(next_timestep, goal, 'concat')
      
      episode_trajectory.append((
        timestep, action, next_timestep, goal
      ))

      # Have the agent and observers observe the timestep.
      # import ipdb; ipdb.set_trace()
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
      episode_return = tree.map_structure(operator.iadd,
                                          episode_return,
                                          next_timestep.reward)
      timestep = next_timestep
      
    # HER
    self.hingsight_experience_replay(start_state, episode_trajectory)

    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - episode_start_time)
    result = {
        'episode_length': episode_steps,
        'episode_return': episode_return,
        'steps_per_second': steps_per_second,
        'env_reset_duration_sec': env_reset_duration,
        'select_action_duration_sec': np.mean(select_action_durations),
        'env_step_duration_sec': np.mean(env_step_durations),
    }
    result.update(counts)
    for observer in self._observers:
      result.update(observer.get_metrics())
    return result
  
  def replay_trajectory_with_new_goal(
    self, trajectory: List, hindsight_goal: OARG):
    """Replay the same trajectory with a goal achieved in hindsight.

    Args:
        trajectory (List): trajectory from pursuing one goal.
        hindsight_goal (OARG): hindsight goal for learning.
    """
    ts0 = trajectory[0][0]
    augmented_ts0 = self.augment_ts_with_goal(ts0, hindsight_goal, 'relabel')
    self._actor.observe_first(augmented_ts0)
    
    for ts, action, next_ts, pursued_goal in trajectory:
      augmented_ts = self.augment_ts_with_goal(ts, hindsight_goal, 'relabel')
      hindsight_action = self._actor.select_action(augmented_ts.observation)
      augmented_next_ts = self.augment_ts_with_goal(next_ts, hindsight_goal, 'relabel')
      self._actor.observe(action, augmented_next_ts)
      if augmented_next_ts.reward == 1:  # truncate early
        assert augmented_next_ts.last(), 'Making sure that we are overwriting R'
        break
      
  def hingsight_experience_replay(
    self, start_state: OARG, trajectory: List):
    """Learn about goal(s) achieved when following a diff goal."""
    def pick_hindsight_goal(traj: List[dm_env.TimeStep]) -> OARG:
      start_idx = len(traj) // 2
      goal_idx = random.randint(start_idx, len(traj) - 1)
      goal_ts = traj[goal_idx]
      assert isinstance(goal_ts, dm_env.TimeStep)
      return goal_ts.observation
    
    def get_achieved_goals() -> List[dm_env.TimeStep]:
      """Filter out goals that are satisfied at s_t."""
      feasible_goals = []
      for _, _, ts, _ in trajectory:
        if not self.goal_reward_func(start_state, ts.observation)[0]:
          feasible_goals.append(ts)
      return feasible_goals
    
    filtered_trajectory = get_achieved_goals()
    if filtered_trajectory:
      hindsight_goal = pick_hindsight_goal(filtered_trajectory)
      print(f'replaying wrt to {hindsight_goal.goals}')
      self.replay_trajectory_with_new_goal(trajectory, hindsight_goal)

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
