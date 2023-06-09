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

"""A wrapper that puts the previous action and reward into the observation."""

from typing import NamedTuple

from acme import types
from acme import specs
from acme.wrappers import base

import dm_env
import tree
import numpy as np


class OARG(NamedTuple):
  """Container for (Observation, Action, Reward) tuples."""
  observation: types.Nest
  action: types.Nest
  reward: types.Nest
  goals: types.Nest


class ObservationActionRewardGoalWrapper(base.EnvironmentWrapper):
  """A wrapper that puts the previous action and reward into the observation."""
  
  def __init__(self,
               environment: dm_env.Environment,
               info2goals,
               n_goal_dims: int):
    super().__init__(environment)
    self._info2goals = info2goals  # func to map dict -> np.ndarray of goals
    self._n_goal_dims = n_goal_dims

  def reset(self) -> dm_env.TimeStep:
    # Initialize with zeros of the appropriate shape/dtype.
    action = tree.map_structure(
        lambda x: x.generate_value(), self._environment.action_spec())
    reward = tree.map_structure(
        lambda x: x.generate_value(), self._environment.reward_spec())
    timestep = self._environment.reset()
    goals = self._info2goals(self._environment.get_info())
    new_timestep = self._augment_observation(action, reward, timestep, goals)
    return new_timestep

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    goals = self._info2goals(self._environment.get_info())
    new_timestep = self._augment_observation(action, timestep.reward, timestep, goals)
    return new_timestep

  def _augment_observation(self, action: types.NestedArray,
                           reward: types.NestedArray,
                           timestep: dm_env.TimeStep,
                           goals: np.ndarray) -> dm_env.TimeStep:
    oar = OARG(observation=timestep.observation,
               action=action,
               reward=reward,
               goals=goals)
    return timestep._replace(observation=oar)
  
  def goal_spec(self):
    return specs.BoundedArray(
      shape=(self._n_goal_dims,),
      dtype=np.int16,
      minimum=0,  # TODO(ab): not sure this will work
      maximum=1024, 
    )

  def observation_spec(self):
    return OARG(observation=self._environment.observation_spec(),
                action=self.action_spec(),
                reward=self.reward_spec(),
                goals=self.goal_spec())

