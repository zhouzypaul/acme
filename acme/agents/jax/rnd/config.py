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
class RNDConfig:
  """Configuration options for RND."""

  # Learning rate for the predictor.
  predictor_learning_rate: float = 1e-4

  # If True, the direct rl algorithm is using the SequenceAdder data format.
  is_sequence_based: bool = False

  # How many gradient updates to perform per step.
  num_sgd_steps_per_step: int = 1

  intrinsic_reward_coefficient: float = 10.
  extrinsic_reward_coefficient: float = 1.0
  
  # When using stale_rewards, we compute intrinsic rewards during acting
  # and don't modify them during learning.
  use_stale_rewards: bool = True
