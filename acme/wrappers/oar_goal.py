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
import hashlib

from collections import OrderedDict

from acme.salient_event.classifier import classify


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
               n_goal_dims: int,
               use_learned_goal_classifiers: bool = False,
               cache_maxsize: int = 100_000):
    super().__init__(environment)
    self._info2goals = info2goals  # func to map dict -> np.ndarray of goals
    self._n_goal_dims = n_goal_dims
    self.classifiers = []
    self._use_learned_goal_classifiers = use_learned_goal_classifiers

    self.cache = OrderedDict()
    self.cache_maxsize = cache_maxsize

  def get_info_vector(self):
    return self._info2goals(self._environment.get_info())

  def get_learned_goal_classifier_vector(self, ts: dm_env.TimeStep):
    goals = np.zeros((self._n_goal_dims), dtype=bool)
    obs_hash = hashlib.sha256(ts.observation.tobytes()).hexdigest()
    for classifier in self.classifiers:
      if (classifier['classifier_id'], obs_hash) in self.cache:
        decision = self.cache[(classifier['classifier_id'], obs_hash)]
      else:
        decision = classify(classifier, ts.observation)
        self.cache[(classifier['classifier_id'], obs_hash)] = decision
        if len(self.cache) > self.cache_maxsize:
          self.cache.popitem(last=False)
      if decision:
        goals[classifier["classifier_id"]] = True
    return goals

  def reset(self) -> dm_env.TimeStep:
    # Initialize with zeros of the appropriate shape/dtype.
    action = tree.map_structure(
        lambda x: x.generate_value(), self._environment.action_spec())
    reward = tree.map_structure(
        lambda x: x.generate_value(), self._environment.reward_spec())
    timestep = self._environment.reset()
    
    if self._use_learned_goal_classifiers:
      goals = self.get_learned_goal_classifier_vector(timestep)
      print(f'Reset with {len(self.classifiers)} classifiers')
    else:
      goals = self.get_info_vector()
    
    new_timestep = self._augment_observation(action, reward, timestep, goals)
    return new_timestep

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    if self._use_learned_goal_classifiers:
      goals = self.get_learned_goal_classifier_vector(timestep)
    else:
      goals = self.get_info_vector()
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
      dtype=bool,
      minimum=np.zeros((self._n_goal_dims), dtype=bool),  # TODO(ab): not sure this will work
      maximum=np.ones((self._n_goal_dims), dtype=bool), 
    )

  def observation_spec(self):
    return OARG(observation=self._environment.observation_spec(),
                action=self.action_spec(),
                reward=self.reward_spec(),
                goals=self.goal_spec())

