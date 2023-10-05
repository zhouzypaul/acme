"""CFN wrapper around the CoreRL learner."""


import functools
import time
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.agents.jax.cfn import networks as cfn_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import reverb_utils
from acme.jax.utils import PrefetchingSplit
from acme.utils.paths import get_save_directory

from acme.wrappers.observation_action_reward import OAR
import acme.agents.jax.cfn.plotting as plotting_utils

import os
import numpy as np
import jax.numpy as jnp
import optax
import reverb


class CFNTrainingState(NamedTuple):
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  params: networks_lib.Params
  target_params: networks_lib.Params
  steps: int
  reward_mean: float
  reward_var: float
  reward_squared_mean: float
  random_prior_mean: jnp.ndarray
  random_prior_var: jnp.ndarray
  random_prior_squared_mean: jnp.ndarray
  states_updated_on: int
  # random_key: networks_lib.PRNGKey

# TODO(ab/sl): mostly need to implement _process_sample() here

class GlobalTrainingState(NamedTuple):
  """Contains training state of the RND learner."""
  rewarder_state: CFNTrainingState
  learner_state: Any

class CFNLearner(acme.Learner):
  """CFN Learner."""

  def __init__(
      self,
      direct_rl_learner_factory: Callable[[Any, Iterator[reverb.ReplaySample]],
                                          acme.Learner],
      iterator: Iterator[reverb.ReplaySample],
      optimizer: optax.GradientTransformation,
      cfn_network: cfn_networks.CFNNetworks,
      rng_key: jnp.ndarray,
      grad_updates_per_batch: int,
      is_sequence_based: bool,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      intrinsic_reward_coefficient=0.001,
      extrinsic_reward_coefficient=1.0,
      use_stale_rewards: bool = False,
      cfn=None,
      value_plotting_freq: int = 1000, # Set to -1 to disable
      ):

    if not use_stale_rewards:
      raise NotImplementedError("Not Implemented (Error)")

    self._direct_rl_learner = direct_rl_learner_factory(
        cfn_network.direct_rl_networks, iterator)
    
    self._cfn = cfn
    self._rng_key = rng_key
    self._cfn_network = cfn_network
    self._value_plotting_freq = value_plotting_freq

    # import ipdb; ipdb.set_trace()
    base_dir = get_save_directory()
    self._plotting_dir = os.path.join(base_dir, 'plots', 'spatial_value_plots')
    os.makedirs(self._plotting_dir, exist_ok=True)

  def step(self):
    self._direct_rl_learner.step()

    if self._value_plotting_freq > 0 and self._cfn and \
      utils.get_from_first_device(self._direct_rl_learner._state).steps % self._value_plotting_freq == 0:
      self._make_spatial_vf_plot()

  def get_variables(self, names: List[str]) -> List[Any]:
    return self._direct_rl_learner.get_variables(names)

  def save(self) -> NamedTuple:
    return self._direct_rl_learner.save()

  def restore(self, state: NamedTuple):
    return self._direct_rl_learner.restore(state)
  
  def _make_spatial_vf_plot(self):
    def get_recurrent_state(batch_size=None):
      return self._cfn_network.direct_rl_networks.init_recurrent_state(self._rng_key, batch_size)

    def get_agent_state():
      return utils.get_from_first_device(self._direct_rl_learner._state)
    
    agent_state = get_agent_state()
    params = agent_state.params
    hashes_to_oar_tuples = self._cfn.get_hash2obs()
    if hashes_to_oar_tuples:
      lstm_state = get_recurrent_state(len(hashes_to_oar_tuples))

      hashes, batch_oarg = self._create_observation_tensor(hashes_to_oar_tuples)

      q_values, _ = self._cfn_network.direct_rl_networks.unroll(
        params,
        self._rng_key,
        batch_oarg,
        lstm_state
      )

      values = q_values.max(axis=-1)[0]  # (1, B, |A|) -> (1, B) -> (B,)

      assert len(values.shape) == 1, values.shape
      assert values.shape[0] == len(hashes), (values.shape, len(hashes))

      plotting_utils.plot_spatial_values(
        hash2value={key: value for key, value in zip(hashes, values)},
        save_path=os.path.join(self._plotting_dir, f'vf_{agent_state.steps}.png'),
        split_by_direction=False
      )

  def _create_observation_tensor(self, hash2obs):
    hashes = list(hash2obs.keys())
    observations = jnp.asarray([hash2obs[key][0] for key in hashes], dtype=jnp.float32)
    actions = jnp.asarray([hash2obs[key][1] for key in hashes], dtype=jnp.int32)
    rewards = jnp.asarray([hash2obs[key][2] for key in hashes], dtype=jnp.float32)
    return hashes, OAR(observations[jnp.newaxis, ...],
                       actions[jnp.newaxis, ...],
                       rewards[jnp.newaxis, ...])
