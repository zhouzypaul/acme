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

"""PPO config."""
import dataclasses

from acme.adders import reverb as adders_reverb
import rlax


@dataclasses.dataclass
class R2D2Config:
  """Configuration options for R2D2 agent."""
  discount: float = 0.997
  target_update_period: int = 2500
  evaluation_epsilon: float = 0.
  num_epsilons: int = 256
  variable_update_period: int = 400

  # Learner options
  burn_in_length: int = 40
  trace_length: int = 80
  sequence_period: int = 40
  learning_rate: float = 1e-3
  bootstrap_n: int = 5
  clip_rewards: bool = False
  tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR

  # Replay options
  samples_per_insert_tolerance_rate: float = 0.1
  samples_per_insert: float = 4.0
  min_replay_size: int = 50_000
  max_replay_size: int = 100_000
  batch_size: int = 64
  prefetch_size: int = 2
  num_parallel_calls: int = 16
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE

  # Priority options
  importance_sampling_exponent: float = 0.6
  priority_exponent: float = 0.9
  max_priority_weight: float = 0.9

  actor_backend: str = 'cpu'

  # GSM Flags
  amdp_rmax_factor: float = 200.
  n_sigmas_threshold_for_goal_creation: int = 0
  prob_augmenting_bonus_constant: float = 0.1
  use_pessimistic_graph_for_planning: bool = True
  off_policy_edge_threshold: float = 0.75
  max_vi_iterations: int = 10

  # When this is <= 0, we use mean + n * std as the novelty threshold.
  novelty_threshold_for_goal_creation: float = -1.

  # When this is -1, it means that we use sum_sampling in GoalSampler.
  goal_space_size: int = 100

  task_goal_probability: float = 0.
  
  use_planning_in_evaluator: bool = False
  should_switch_goal: bool = False
  subgoal_sampler_default_behavior: str = 'graph_search'
  option_timeout: int = 400

  use_exploration_vf_for_expansion: bool = True
  use_intermediate_difficulty: bool = False
  use_uvfa_reachability: bool = False
  num_goals_to_replay: int = 5