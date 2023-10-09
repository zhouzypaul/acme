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

"""Simple JAX actors."""

from typing import Generic, Optional

from acme import adders
from acme import core
from acme import types
from acme.agents.jax import actor_core
from acme.jax import networks as network_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.wrappers.observation_action_reward import OAR
import dm_env
import jax


class GenericActor(core.Actor, Generic[actor_core.State, actor_core.Extras]):
  """A generic actor implemented on top of ActorCore.

  An actor based on a policy which takes observations and outputs actions. It
  also adds experiences to replay and updates the actor weights from the policy
  on the learner.
  """

  def __init__(
      self,
      actor: actor_core.ActorCore[actor_core.State, actor_core.Extras],
      random_key: network_lib.PRNGKey,
      variable_client: Optional[variable_utils.VariableClient],
      adder: Optional[adders.Adder] = None,
      jit: bool = True,
      backend: Optional[str] = 'cpu',
      per_episode_update: bool = False,
  ):
    """Initializes a feed forward actor.

    Args:
      actor: actor core.
      random_key: Random key.
      variable_client: The variable client to get policy parameters from.
      adder: An adder to add experiences to.
      jit: Whether or not to jit the passed ActorCore's pure functions.
      backend: Which backend to use when jitting the policy.
      per_episode_update: if True, updates variable client params once at the
        beginning of each episode
    """
    self._random_key = random_key
    self._variable_client = variable_client
    self._adder = adder
    self._state = None

    # Unpack ActorCore, jitting if requested.
    if jit:
      self._init = jax.jit(actor.init, backend=backend)
      self._policy = jax.jit(actor.select_action, backend=backend)
    else:
      self._init = actor.init
      self._policy = actor.select_action
    self._get_extras = actor.get_extras
    self._per_episode_update = per_episode_update

  @property
  def _params(self):
    params = self._variable_client.params if self._variable_client else []
    return params

  def select_action(self,
                    observation: network_lib.Observation) -> types.NestedArray:

    action, self._state = self._policy(self._params, observation, self._state)
    return utils.to_numpy(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._random_key, key = jax.random.split(self._random_key)
    self._state = self._init(key)
    if self._adder:
      self._adder.add_first(timestep)
    if self._variable_client and self._per_episode_update:
      self._variable_client.update_and_wait()

  def observe(self, action: network_lib.Action, next_timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add(
          action, next_timestep, extras=self._get_extras(self._state))

  def update(self, wait: bool = False):
    if self._variable_client and not self._per_episode_update:
      self._variable_client.update(wait)


# TODO(ab/sl): Refactor so that there is a GenericIntrinsicActor and RNDActor and CFNActor

class GenericIntrinsicActor(GenericActor):
  """[StaleRewards] Actor that uses r_int for action selection & adds it to replay."""

  def __init__(
    self,
    intrinsic_reward_scale: float,
    extrinsic_reward_scale: float,
    condition_actor_on_intrinsic_reward: bool,
    *args,
    **kwargs
  ):
    self._intrinsic_reward_scale = intrinsic_reward_scale
    self._extrinsic_reward_scale = extrinsic_reward_scale
    self._condition_actor_on_intrinsic_reward = condition_actor_on_intrinsic_reward
    super().__init__(*args, **kwargs)
  
  @property
  def _params(self):
    params = self._variable_client.params[0] if self._variable_client else []
    return params
  
  @property
  def _rnd_state(self):
    assert len(self._variable_client.params) == 2, "Assuming params, rnd_state."
    params = self._variable_client.params[1] if self._variable_client else []
    return params
  
  def select_action(self,
                    observation: network_lib.Observation) -> types.NestedArray:

    action, self._state = self._policy(
      self._params, observation, self._state, self._rnd_state)
    return utils.to_numpy(action)
  
  def observe(self, action: network_lib.Action, next_timestep: dm_env.TimeStep,
              extras: Optional[dict] = None):
    if self._adder:
      combined_reward = (self._extrinsic_reward_scale * next_timestep.reward) \
          + (self._intrinsic_reward_scale * self._state.prev_intrinsic_reward)
      next_timestep = next_timestep._replace(reward=combined_reward)

      # s_{t+1} = <o_{t+1}, a_t, r_t> 
      # where r_t = r_ext_t + r_int_{t-1}
      if self._condition_actor_on_intrinsic_reward:
        next_oar: OAR = next_timestep.observation
        next_oar = next_oar._replace(reward=next_timestep.reward)
        next_timestep = next_timestep._replace(observation=next_oar)

      self._adder.add(
          action, next_timestep,
          extras=self._get_extras(self._state) if extras is None else extras)


class CFNIntrinsicActor(GenericIntrinsicActor):
  def __init__(
      self,
      cfn_variable_client: variable_utils.VariableClient,
      cfn_adder: adders.Adder,
      *args,
      **kwargs,
  ):
    self._cfn_adder = cfn_adder
    self._cfn_variable_client = cfn_variable_client
    super().__init__(*args, **kwargs)

  @property
  def _params(self):
    params = self._variable_client.params if self._variable_client else []
    return params

  @property
  def _rnd_state(self):
    # since cfn.py  get_variables returns list of len 1, its unpacked by VariableClient
    params = self._cfn_variable_client.params if self._cfn_variable_client else []
    return params
  
  def update(self, wait: bool = False):
    if self._cfn_variable_client:
      self._cfn_variable_client.update(wait)
    return super().update(wait)

  # TODO(sl): add per-episode support for CfnVarClient
  def observe_first(self, timestep: dm_env.TimeStep):
    if self._cfn_adder:
      self._cfn_adder.add_first(timestep)
    return super().observe_first(timestep)
  
  def observe(self, action: network_lib.Action, next_timestep: dm_env.TimeStep):
    # TODO(ab/sl): pass in random_key to get_extras, then update it locally, so its advances

    # Split RNG so that bernoulli isnt same every time
    rng, new_rng = jax.random.split(self._state.rng)
    self._state = self._state.replace(rng=rng)
    extras = self._get_extras(self._state)
    # Update again to make sure we don't use same 
    self._state = self._state.replace(rng=new_rng)
    if self._cfn_adder:
      self._cfn_adder.add(action, next_timestep, extras=extras)
    return super().observe(action, next_timestep, extras=extras)
