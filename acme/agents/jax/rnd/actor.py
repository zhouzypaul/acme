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

"""R2D2 actor."""


from acme.agents.jax import actor_core as actor_core_lib

from acme.agents.jax.rnd import networks as rnd_networks
from acme.jax import networks as networks_lib
from acme.agents.jax.rnd.learning import RNDTrainingState
from acme.wrappers.observation_action_reward import OAR
from acme.agents.jax.rnd.networks import compute_rnd_reward
from acme.agents.jax.r2d2.actor import R2D2Policy, R2D2ActorState
from acme.jax import utils

import jax.numpy as jnp

def get_actor_core(
    networks: rnd_networks.RNDNetworks,
    rl_actor_core: R2D2Policy,
    intrinsic_reward_scale: float,
    extrinsic_reward_scale: float,
    condition_actor_on_intrinsic_reward: bool,
) -> R2D2Policy:
  """Returns ActorCore for R2D2."""

  def get_intrinsic_reward(
      observation: OAR, rnd_state: RNDTrainingState) -> float:

    new_obs = utils.add_batch_dim(
                utils.add_batch_dim(
                  utils.add_batch_dim(observation.observation)))
    
    observation = observation._replace(observation=new_obs)

    unscaled_reward_int = compute_rnd_reward(
      predictor_params=rnd_state.params,
      target_params=rnd_state.target_params,
      transitions=observation,
      networks=networks,
      observation_mean=rnd_state.observation_mean,
      observation_var=rnd_state.observation_var
    )
    
    norm_reward = unscaled_reward_int - rnd_state.reward_mean
    norm_reward /= jnp.sqrt(jnp.maximum(rnd_state.reward_var, 1e-12))
    return norm_reward[0][0][0]
  
  def modify_obs_with_intrinsic_reward(
      observation: OAR, intrinsic_reward: float) -> OAR:
    combined_reward = (extrinsic_reward_scale * observation.reward) \
      + (intrinsic_reward_scale * intrinsic_reward)
    return observation._replace(reward=combined_reward)

  def select_action(params: networks_lib.Params,  # RL Agent Params (e.g, R2D2)
                    observation: networks_lib.Observation,
                    state: R2D2ActorState[actor_core_lib.RecurrentState],
                    rnd_state: RNDTrainingState):
    if condition_actor_on_intrinsic_reward:
      intrinsic_reward = get_intrinsic_reward(observation, rnd_state)
      observation = modify_obs_with_intrinsic_reward(
        observation, state.prev_intrinsic_reward)

    action, state = rl_actor_core.select_action(params, observation, state)
    state = state.replace(prev_intrinsic_reward=intrinsic_reward)

    return action, state

  return actor_core_lib.ActorCore(init=rl_actor_core.init, select_action=select_action,
                                  get_extras=rl_actor_core.get_extras)
