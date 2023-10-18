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

"""CFN wrapper around an R2D2 base policy."""


from acme.agents.jax import actor_core as actor_core_lib
from typing import Callable, Generic, Mapping, Optional, Tuple

from acme.agents.jax.cfn import networks as cfn_networks
from acme.jax import networks as networks_lib
from acme.agents.jax.cfn.learning import CFNTrainingState
from acme.wrappers.observation_action_reward import OAR
from acme.agents.jax.cfn.networks import compute_cfn_reward
from acme.agents.jax.r2d2.actor import R2D2Policy, R2D2ActorState
from acme.jax import utils

import jax
import jax.numpy as jnp

CFNExtras = Mapping[str, jnp.ndarray]

def get_actor_core(
    networks: cfn_networks.CFNNetworks,
    rl_actor_core: R2D2Policy,
    intrinsic_reward_scale: float,
    extrinsic_reward_scale: float,
    use_reward_normalization: bool = False,
    cfn_output_dimensions: int = 20
) -> R2D2Policy:
  """Returns ActorCore for R2D2 that adds the intrinsic reward to the OAR."""

  def get_intrinsic_reward(
      observation: OAR, cfn_state: CFNTrainingState) -> float:

    cfn_intrinsic_reward = compute_cfn_reward(
      predictor_params=cfn_state.params,
      target_params=cfn_state.target_params,
      transitions=observation,
      networks=networks,
      random_prior_mean=cfn_state.random_prior_mean,
      random_prior_std=jnp.sqrt(cfn_state.random_prior_var + 1e-12)
    )
    
    if use_reward_normalization:
      norm_reward = cfn_intrinsic_reward - cfn_state.reward_mean
      norm_reward /= jnp.sqrt(jnp.maximum(cfn_state.reward_var, 1e-12))
      return norm_reward[0][0][0]

    return cfn_intrinsic_reward
  
  def modify_obs_with_intrinsic_reward(
      observation: OAR, intrinsic_reward: float) -> OAR:
    combined_reward = (extrinsic_reward_scale * observation.reward) \
      + (intrinsic_reward_scale * intrinsic_reward)
    return observation._replace(reward=combined_reward)

  def select_action(params: networks_lib.Params,  # RL Agent Params (e.g, R2D2)
                    observation: networks_lib.Observation,
                    state: R2D2ActorState[actor_core_lib.RecurrentState],
                    cfn_state: CFNTrainingState):

    intrinsic_reward = get_intrinsic_reward(observation, cfn_state)
    modified_oar = modify_obs_with_intrinsic_reward(
      observation, state.prev_intrinsic_reward)

    action, state = rl_actor_core.select_action(params, modified_oar, state)
    state = state.replace(prev_intrinsic_reward=intrinsic_reward)

    return action, state

  def get_extras(
      state: R2D2ActorState[actor_core_lib.RecurrentState]) -> CFNExtras:
    direct_rl_extras = rl_actor_core.get_extras(state)
    
    coin_flips = 2 * jax.random.bernoulli(state.rng, shape=(cfn_output_dimensions,)) - 1
    coin_flips = coin_flips.astype(jnp.float32)
    cfn_extras = dict(coin_flips=coin_flips)

    return {**direct_rl_extras, **cfn_extras}

  return actor_core_lib.ActorCore(init=rl_actor_core.init,
                                  select_action=select_action,
                                  get_extras=get_extras)