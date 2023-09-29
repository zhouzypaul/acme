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

"""RND config."""
import dataclasses


@dataclasses.dataclass
class CFNConfig:
  """Configuration options for RND."""

  # Learning rate for the predictor.
  cfn_learning_rate: float = 1e-4

  # How many gradient updates to perform per step.
  num_sgd_steps_per_step: int = 1

  intrinsic_reward_coefficient: float = 0.001
  extrinsic_reward_coefficient: float = 1.0
  
  # When using stale_rewards, we compute intrinsic rewards during acting
  # and don't modify them during learning.
  use_stale_rewards: bool = False

  variable_update_period: int = 400

  min_replay_size: int = 50_000 // 4
  max_replay_size: int = 1_000_000
  samples_per_insert: int = 8
  samples_per_insert_tolerance_rate: float = 0.1
  
  cfn_replay_table_name: str = 'cfn_replay_table'
  prefetch_size: int = 2
  
  batch_size: int = 1024
  use_reward_normalization: bool = False
  cfn_output_dimensions: int = 20
  is_sequence_based: bool = False
  
  # Priority options
  importance_sampling_exponent: float = 0.6
  priority_exponent: float = 1.0
  max_priority_weight: float = 0.9

  predictor_learning_rate: float = 1e-3