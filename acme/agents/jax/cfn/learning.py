"""CFN wrapper around the CoreRL learner."""


import functools
import time
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

import jax
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
from acme.jax import variable_utils

from acme.wrappers.observation_action_reward import OAR
import acme.agents.jax.cfn.plotting as plotting_utils
from acme.agents.jax.cfn.networks import compute_cfn_reward

import pickle
import os
import numpy as np
import jax.numpy as jnp
import optax
import reverb
import rlax

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
      tx_pair=rlax.SIGNED_HYPERBOLIC_PAIR,
      cfn_variable_client: Optional[variable_utils.VariableClient] = None,
      cfn_var_to_std_eps: float = 1e-12,
      ):
    
    if use_stale_rewards:
      assert cfn_variable_client is None, "Don't need to maintain copy of CFN params."

    updated_iterator = iterator

    if not use_stale_rewards:
      updated_iterator = (self._process_sample(sample) for sample in iterator)

    self._direct_rl_learner = direct_rl_learner_factory(
        cfn_network.direct_rl_networks, updated_iterator)
    
    self._cfn = cfn
    self._rng_key = rng_key
    self._cfn_network = cfn_network
    self._value_plotting_freq = value_plotting_freq
    self._tx_pair = tx_pair
    self._is_sequence_based = is_sequence_based
    self._cfn_variable_client = cfn_variable_client
    self._cfn_var_to_std_eps = cfn_var_to_std_eps
    self._intrinsic_reward_scale = intrinsic_reward_coefficient
    self._extrinsic_reward_scale = extrinsic_reward_coefficient

    def batched_intrinsic_reward(transitions, cfn_state):
      new_func = jax.tree_util.Partial(
        compute_cfn_reward,
        predictor_params=cfn_state.params,
        target_params=cfn_state.target_params,
        networks=cfn_network,
        random_prior_mean=cfn_state.random_prior_mean,
        random_prior_std=jnp.sqrt(cfn_state.random_prior_var + cfn_var_to_std_eps)
      )
      return jax.vmap(new_func, in_axes=0)(transitions=transitions)[jnp.newaxis, ...]
    
    self._compute_intrinsic_reward = jax.jit(batched_intrinsic_reward)

    # import ipdb; ipdb.set_trace()
    base_dir = get_save_directory()
    self._plotting_dir = os.path.join(base_dir, 'plots', 'spatial_value_plots')
    os.makedirs(self._plotting_dir, exist_ok=True)
    self._spread_plotting_dir = os.path.join(base_dir, 'plots', 'max_value_spread_plots')
    os.makedirs(self._spread_plotting_dir, exist_ok=True)
    self._spread_list = []


  def step(self):
    self._direct_rl_learner.step()
    self.update_cfn_state()

    if self._value_plotting_freq > 0 and self._cfn and \
      utils.get_from_first_device(self._direct_rl_learner._state).steps % self._value_plotting_freq == 0:
      self._make_spatial_vf_plot()

  def _process_sample(self, sample: reverb.ReplaySample) -> reverb.ReplaySample:
    """Uses the replay sample to train and update its reward.

    Args:
      sample: Replay sample to train on.

    Returns:
      The sample replay sample with an updated reward.
    """
    t0 = time.time(); print('About to start processing sample')
    # Sample (s, a, r, s') from replay buffer
    is_prefetch = isinstance(sample, PrefetchingSplit)
    
    if is_prefetch:
      prefetch_split_sample = sample
      sample = sample.device

    transitions = reverb_utils.replay_sample_to_sars_transition(
        sample, is_sequence=self._is_sequence_based)
    extrinsic_reward = transitions.reward

    # Compute the intrinsic reward for s using CFN networks
    oar = transitions.observation

    # Squeeze the leading device dimension which is always 1
    assert oar.observation.shape[0] == 1, oar.observation.shape
    oar = oar._replace(
      observation=oar.observation.squeeze(axis=0),
      action=oar.action.squeeze(axis=0),
      reward=oar.reward.squeeze(axis=0))

    intrinsic_reward = self._compute_intrinsic_reward(oar, self._cfn_state)

    reward_shapes = intrinsic_reward.shape, extrinsic_reward.shape
    assert intrinsic_reward.shape == extrinsic_reward.shape, reward_shapes

    # Update the reward in the sample and return
    combined_reward = (self._extrinsic_reward_scale * extrinsic_reward) \
        + (self._intrinsic_reward_scale * intrinsic_reward)
    
    updated_sample = sample._replace(
      data=sample.data._replace(reward=combined_reward))
    
    if is_prefetch:
      updated_sample = PrefetchingSplit(
        device=updated_sample, host=prefetch_split_sample.host)
    print(f'CFN learner took {time.time() - t0} seconds to process sample.')
    return updated_sample

  def get_variables(self, names: List[str]) -> List[Any]:
    return self._direct_rl_learner.get_variables(names)
  
  @property
  def _cfn_state(self) -> CFNTrainingState:
    params = self._cfn_variable_client.params if self._cfn_variable_client else []
    return params
  
  def update_cfn_state(self, wait: bool = False):
    if self._cfn_variable_client:
      self._cfn_variable_client.update(wait)

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

      q_values = self._tx_pair.apply_inv(q_values)

      values = q_values.max(axis=-1)[0]  # (1, B, |A|) -> (1, B) -> (B,)

      value_diff = values.max() - values.min()
      self._spread_list.append(value_diff)
      plotting_utils.plot_quantity_over_iteration(self._spread_list,
                                            save_path=os.path.join(self._spread_plotting_dir, 'value_spread.png'),
                                            quantity_name='Value Spread')
      with open(os.path.join(self._spread_plotting_dir, 'value_spread.pkl'), 'wb+') as f:
        pickle.dump(self._spread_list, f)

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
