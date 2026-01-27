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

from typing import NamedTuple, Optional

from acme import types
from acme import specs
from acme.wrappers import base

import dm_env
import tree
import numpy as np
import hashlib
import os
import glob
import matplotlib.pyplot as plt

from collections import OrderedDict, defaultdict

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
               cache_maxsize: int = 100_000,
               classifier_trigger_dir: Optional[str] = None,
               max_triggers_per_classifier: int = 50):
    super().__init__(environment)
    self._info2goals = info2goals  # func to map dict -> np.ndarray of goals
    self._n_goal_dims = n_goal_dims
    self._classifiers = []
    self._use_learned_goal_classifiers = use_learned_goal_classifiers

    self.cache = OrderedDict()
    self.cache_maxsize = cache_maxsize
    self.classifier_firing_tracker = None  # Track classifier firings
    
    # Classifier trigger visualization
    self.classifier_trigger_dir = classifier_trigger_dir
    self.max_triggers_per_classifier = max_triggers_per_classifier
    self.trigger_count = defaultdict(int)  # Total triggers per classifier
    self.step_count = 0
    self.current_goal_idx = None  # Track current goal for visualization
    
    if classifier_trigger_dir:
      os.makedirs(classifier_trigger_dir, exist_ok=True)

  @property
  def classifiers(self):
    return self._classifiers

  @classifiers.setter
  def classifiers(self, classifiers_list):
    """Sanitize classifiers when they are set."""
    sanitized_list = []
    for clf in classifiers_list:
        # Shallow copy to avoid mutating original if shared
        new_clf = clf.copy()
        if 'salient_patches' in new_clf:
            new_patches = {}
            for k, patch in new_clf['salient_patches'].items():
                # Ensure patch is uint8
                if patch.dtype != np.uint8:
                    if patch.dtype.kind == 'f' and patch.max() <= 1.05:
                        # Normalized float -> uint8
                        new_patches[k] = (patch * 255).astype(np.uint8)
                    else:
                        # Just cast
                        new_patches[k] = patch.astype(np.uint8)
                else:
                    new_patches[k] = patch
            new_clf['salient_patches'] = new_patches
        sanitized_list.append(new_clf)
    
    self._classifiers = sanitized_list

  def set_current_goal(self, goal_idx: Optional[int]):
    """Set the current goal being pursued for visualization purposes."""
    self.current_goal_idx = goal_idx

  def get_info_vector(self):
    return self._info2goals(self._environment.get_info())

  def _hash_observation(self, obs: np.ndarray):
    is_image_type = len(obs.shape) == 3
    if is_image_type:
      return hashlib.sha256(obs.tobytes()).hexdigest()
    return np.floor(obs).astype(int).tostring()

  def _save_classifier_trigger(self, classifier_id: int, triggered_obs: np.ndarray, 
                               pursued_goal_idx: Optional[int] = None):
    """Save side-by-side plot: prototype state | triggered state.
    
    Shows bounding boxes on prototype and displays which goal is being pursued.
    Maintains only the last max_triggers_per_classifier files per classifier.
    """
    try:
      import cv2
      from acme.salient_event import patch_utils
      
      classifier = self.classifiers[classifier_id]
      prototype_img = classifier['prototype_image'].copy()  # Saved in classifier
      triggered_img = triggered_obs[:, :, :3].copy()  # Current environment state (RGB channels)
      
      # Draw bounding boxes on prototype
      if 'salient_patches' in classifier:
        bboxes = {i: bbox for i, bbox in enumerate(classifier['salient_patches'])}
        prototype_img = patch_utils.draw_bounding_boxes(prototype_img, bboxes)
      
      # Create side-by-side plot
      fig, axs = plt.subplots(1, 2, figsize=(10, 5))
      
      # Left: Prototype with bounding boxes
      axs[0].imshow(prototype_img)
      axs[0].set_title(f'Classifier {classifier_id} Prototype')
      axs[0].axis('off')
      
      # Right: Triggered state
      axs[1].imshow(triggered_img)
      title = f'Triggered State (step {self.step_count})'
      if pursued_goal_idx is not None:
        title += f'\nPursuing Goal: {pursued_goal_idx}'
      axs[1].set_title(title)
      axs[1].axis('off')
      
      plt.tight_layout()
      
      # Save to classifier-specific directory
      classifier_dir = os.path.join(self.classifier_trigger_dir, f'classifier_{classifier_id}')
      os.makedirs(classifier_dir, exist_ok=True)
      
      # Maintain only last N triggers per classifier
      trigger_num = self.trigger_count[classifier_id]
      filename = f'trigger_{trigger_num:05d}_step_{self.step_count:010d}.png'
      filepath = os.path.join(classifier_dir, filename)
      
      plt.savefig(filepath, dpi=100, bbox_inches='tight')
      plt.close(fig)
      
      # Clean up old triggers if we exceed max
      existing_files = sorted(glob.glob(os.path.join(classifier_dir, 'trigger_*.png')))
      if len(existing_files) > self.max_triggers_per_classifier:
        for old_file in existing_files[:-self.max_triggers_per_classifier]:
          os.remove(old_file)
      
    except Exception as e:
      print(f'[OARWrapper] Error saving classifier trigger: {e}')

  def get_learned_goal_classifier_vector(self, ts: dm_env.TimeStep, step_count: int = None):
    """Get goal vector from learned classifiers and optionally save trigger visualizations.
    
    Args:
      ts: Current timestep
      step_count: Optional step count for visualization filenames
    """
    if step_count is not None:
      self.step_count = step_count
    
    goals = np.zeros((self._n_goal_dims), dtype=bool)
    obs_hash = self._hash_observation(ts.observation)
    
    for classifier in self.classifiers:
      if (classifier['classifier_id'], obs_hash) in self.cache:
        decision = self.cache[(classifier['classifier_id'], obs_hash)]
      else:
        decision = classify(classifier, ts.observation)
        self.cache[(classifier['classifier_id'], obs_hash)] = decision
        if len(self.cache) > self.cache_maxsize:
          self.cache.popitem(last=False)
      
      if decision:
        classifier_id = classifier["classifier_id"]
        goals[classifier_id] = True
        
        # Log classifier firing
        if self.classifier_firing_tracker is not None:
          self.classifier_firing_tracker.log_firing(classifier_id)
        
        
        # Save visualization if enabled (uses self.current_goal_idx set by environment loop)
        # if self.classifier_trigger_dir:
          # self.trigger_count[classifier_id] += 1
          # self._save_classifier_trigger(classifier_id, ts.observation, self.current_goal_idx)
    
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

