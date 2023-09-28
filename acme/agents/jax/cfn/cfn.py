"""Main CFN run-loop."""

import collections
import os

import threading
import time
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

from absl import logging
import acme
from acme.agents.jax.cfn import networks as cfn_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import async_utils
from acme.utils import counting
from acme.utils import loggers
from acme.wrappers.observation_action_reward import OAR
from acme.agents.jax.cfn.learning import CFNTrainingState
from acme.agents.jax.cfn.networks import compute_cfn_reward

import jax
import numpy as np
import jax.numpy as jnp
import optax
import reverb
import tree

import acme.agents.jax.cfn.plotting as plotting_utils


_PMAP_AXIS_NAME = 'data'
# This type allows splitting a sample between the host and device, which avoids
# putting item keys (uint64) on device for the purposes of priority updating.
CFNReplaySample = utils.PrefetchingSplit


class CFN(acme.Learner):
  """Coin Flip Network that learns and maintains a pseudocount function."""

  def __init__(self,
               networks: cfn_networks.CFNNetworks,
               random_key: networks_lib.PRNGKey,
               max_priority_weight: float,
               iterator: Iterator[CFNReplaySample],
               optimizer: optax.GradientTransformation,
               replay_client: Optional[reverb.Client] = None,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None,
               bonus_plotting_freq = 500  # set to np.inf to disable plotting
  ):


    def loss(
        params: networks_lib.Params,
        rp_params: networks_lib.Params,
        # key_grad: networks_lib.PRNGKey,
        sample: reverb.ReplaySample,
        random_prior_mean: jnp.ndarray,
        random_prior_std: jnp.ndarray
    ) -> Tuple[jnp.float32, Tuple[jnp.ndarray, jnp.ndarray]]:
      """Computes the CFN loss for a batch of sequences."""

      # TODO(ab/sl): indexing is awkward and unreadable rn
      observation = sample.data.observation.observation[:, 0, ...]

      rp_output = networks.target.apply(rp_params, observation, sample.data.action)
      normalized_rp_output = (rp_output - random_prior_mean) / random_prior_std
      predictor_output = networks.predictor.apply(params, observation, sample.data.action)

      pred = predictor_output + normalized_rp_output
      
      target = sample.data.extras['coin_flips']
      target = target.squeeze(1)  # TODO(ab/sl): indexing is awkward and unreadable rn

      assert pred.shape == target.shape, (pred.shape, target.shape)
      
      # TODO(ab/sl)
      dummy_priorities = jnp.ones((random_prior_mean.shape[0],), dtype=jnp.float32)

      intrinsic_reward = networks.get_reward(
        predictor_output=predictor_output,
        normalized_rp_output=normalized_rp_output,
        original_reward=0.)

      return optax.l2_loss(pred, target).mean(), (rp_output, dummy_priorities, intrinsic_reward)


    def sgd_step(
      state: CFNTrainingState,
      samples: reverb.ReplaySample
    ) -> Tuple[CFNTrainingState, jnp.ndarray, Dict[str, jnp.ndarray]]:
      """Performs an update step, averaging over pmap replicas."""
      grad_fn = jax.value_and_grad(loss, has_aux=True)
      # key, key_grad = jax.random.split(state.random_key)
      (loss_value, (random_prior_output, priorities, r_int)), gradients = grad_fn(
        state.params,
        state.target_params,
        # key_grad,
        samples,
        state.random_prior_mean,
        jnp.sqrt(state.random_prior_var))

      # Average gradients over pmap replicas before optimizer update.
      # gradients = jax.lax.pmean(gradients, _PMAP_AXIS_NAME)

      # Apply optimizer updates.
      updates, new_opt_state = optimizer.update(gradients, state.optimizer_state)
      new_params = optax.apply_updates(state.params, updates)

      # Periodically update target networks.
      steps = state.steps + 1

      new_cfn_state = state._replace(
        optimizer_state=new_opt_state,
        params=new_params,
        steps=steps,
        # random_key=key,
      )

      new_cfn_state = update_mean_variance_stats(new_cfn_state, random_prior_output, r_int)
      return new_cfn_state, priorities, {'loss': loss_value}


    def update_priorities(  # TODO(ab/sl)
        keys_and_priorities: Tuple[jnp.ndarray, jnp.ndarray]):
      pass

    def update_mean_variance_stats(
        state: CFNTrainingState,
        random_prior_output: networks_lib.NetworkOutput,
        unscaled_intrinsic_reward: networks_lib.NetworkOutput
      ) -> CFNTrainingState:
      # num_transitions = jnp.prod(jnp.array(unscaled_intrinsic_reward.shape))
      # import ipdb; ipdb.set_trace()
      num_transitions = unscaled_intrinsic_reward.shape[0]
      new_states_updated_on = state.states_updated_on + num_transitions
      delta_reward = (unscaled_intrinsic_reward - state.reward_mean)
      delta_reward = delta_reward.mean()
      new_reward_mean = state.reward_mean + (num_transitions * delta_reward / new_states_updated_on)
      delta_reward_squared = (unscaled_intrinsic_reward**2 - state.reward_squared_mean)
      new_reward_squared_mean = state.reward_squared_mean + (num_transitions*delta_reward_squared.mean() / new_states_updated_on)
      new_reward_var = new_reward_squared_mean - new_reward_mean**2
      
      assert len(state.random_prior_mean.shape) == 1
      assert random_prior_output.shape == (num_transitions, state.random_prior_mean.shape[0])

      delta_rp = (random_prior_output - state.random_prior_mean)
      new_rp_mean = state.random_prior_mean + (num_transitions * delta_rp.mean(axis=0) / new_states_updated_on)
      # delta_rp_squared = (delta_rp**2 - state.random_prior_squared_mean)
      delta_rp_squared = (random_prior_output**2 - state.random_prior_squared_mean)
      new_rp_squared_mean = state.random_prior_squared_mean + (num_transitions * delta_rp_squared.mean(axis=0) / new_states_updated_on)
      new_rp_var = new_rp_squared_mean - (new_rp_mean ** 2)

      return state._replace(
        reward_mean=new_reward_mean,
        reward_var=new_reward_var,
        reward_squared_mean=new_reward_squared_mean,
        random_prior_mean=new_rp_mean,
        random_prior_var=new_rp_var,
        random_prior_squared_mean=new_rp_squared_mean,
        states_updated_on=new_states_updated_on,
      )
    
    # Internalise components, hyperparameters, logger, counter, and methods.
    self._iterator = iterator
    self._replay_client = replay_client
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner',
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        time_delta=1.,
        steps_key=self._counter.get_steps_key())

    self._sgd_step = jax.pmap(sgd_step, axis_name=_PMAP_AXIS_NAME)
    self._async_priority_updater = async_utils.AsyncExecutor(update_priorities)

    # Initialise and internalise training state (parameters/optimiser state).
    random_key, key_init = jax.random.split(random_key)
    init_pred_params = networks.predictor.init(key_init)
    init_target_params = networks.target.init(key_init)
    opt_state = optimizer.init(init_pred_params)

    # Log how many parameters the network has.
    sizes = tree.map_structure(jnp.size, init_pred_params)
    logging.info('Total number of params: %d',
                 sum(tree.flatten(sizes.values())))
    
    self._state = utils.replicate_in_all_devices(
      CFNTrainingState(
        optimizer_state=opt_state,
        params=init_pred_params,
        target_params=init_target_params,
        steps=0,
        reward_mean=0.,
        reward_var=1.,
        reward_squared_mean=0.,
        random_prior_mean=jnp.zeros((20,), dtype=jnp.float32),  # TODO(ab/sl): pass in output_dims
        random_prior_var=jnp.ones((20,), dtype=jnp.float32),
        random_prior_squared_mean=jnp.zeros((20,), dtype=jnp.float32),
        states_updated_on=0))
    
    # Keep track of ground-truth counts for debugging/visualizations.
    self._hash2obs = {}  # map goal hash to obs
    self._hash2counts = collections.defaultdict(int)
    self._count_dict_lock = threading.Lock()
    self._networks = networks
    self._bonus_plotting_freq = bonus_plotting_freq

    # TODO(ab/sl): Make plotting dirs based on acme_id
    os.makedirs('plots/cfn_plots/spatial_bonuses', exist_ok=True)
    os.makedirs('plots/cfn_plots/true_vs_approx_scatterplots', exist_ok=True)

  def step(self):
    prefetching_split = next(self._iterator)
    # The split_sample method passed to utils.sharded_prefetch specifies what
    # parts of the objects returned by the original iterator are kept in the
    # host and what parts are prefetched on-device.
    # In this case the host property of the prefetching split contains only the
    # replay keys and the device property is the prefetched full original
    # sample.
    keys = prefetching_split.host
    samples: reverb.ReplaySample = prefetching_split.device

    # Do a batch of SGD.
    start = time.time()
    self._state, priorities, metrics = self._sgd_step(self._state, samples)
    # Take metrics from first replica.
    metrics = utils.get_from_first_device(metrics)
    # Update our counts and record it.
    counts = self._counter.increment(steps=1, time_elapsed=time.time() - start)

    # Update priorities in replay.
    if self._replay_client:
      self._async_priority_updater.put((keys, priorities))

    # Attempt to write logs.
    self._logger.write({**metrics, **counts})

    if self._state.steps > 0 and \
        self._state.steps % self._bonus_plotting_freq == 0 and \
        len(self._hash2obs) > 1:
      self._make_cfn_bonus_plots()

  def _make_cfn_bonus_plots(self):
    state: CFNTrainingState = utils.get_from_first_device(self._state)
    hashes, oar = self._create_observation_tensor()
    
    observation_tensor = oar.observation
    assert len(observation_tensor.shape) == 4, observation_tensor.shape
    assert observation_tensor.shape[1:] == (84, 84, 3), observation_tensor.shape

    predicted_bonuses = compute_cfn_reward(
      state.params,
      state.target_params,
      oar,
      self._networks,
      state.random_prior_mean,
      jnp.sqrt(state.random_prior_var)
    )
    
    assert predicted_bonuses.shape == (len(hashes),), predicted_bonuses.shape

    approx_bonus_info = {k: v for k, v in zip(hashes, predicted_bonuses)}
    plotting_utils.plot_spatial_count_or_bonus(
      true_count_info=self._hash2counts,
      approx_bonus_info=approx_bonus_info,
      save_path=f'plots/cfn_plots/spatial_bonuses/spatial_bonus_{state.steps}.png'
    )
    plotting_utils.plot_true_vs_approx_bonus(
      true_count_info=self._hash2counts,
      approx_bonus_info=approx_bonus_info,
      save_path=f'plots/cfn_plots/true_vs_approx_scatterplots/bonus_scatterplot_{state.steps}.png'
    )
      
  # TODO(ab/sl): add the count dictionaries so that they checkpoint correctly.
  def get_variables(self, names: List[str]) -> List[networks_lib.Params]:
    del names  # There's only one available set of params in this agent.
    # Return first replica of parameters.
    # Sam changed so it has whole state, there's a chance that's bad with first_device thing 
    return utils.get_from_first_device([self._state])
  
  def get_hash2obs(self) -> Mapping[Tuple, Tuple]:
    keys = list(self._hash2obs.keys())
    return {k: self._hash2obs[k] for k in keys}

  def save(self) -> CFNTrainingState:
    # Serialize only the first replica of parameters and optimizer state.
    return utils.get_from_first_device(self._state)

  def restore(self, state: CFNTrainingState):
    self._state = utils.replicate_in_all_devices(state)

  def update_ground_truth_counts(self, hash2obs, hash2counts):
    self._update_obs_dict(hash2obs)
    self._update_count_dict(hash2counts)

  def _update_count_dict(self, hash2count: Dict):
    with self._count_dict_lock:
      for obs_hash in hash2count:
        self._hash2counts[obs_hash] += hash2count[obs_hash]
  
  def _update_obs_dict(self, hash2obs: Dict):
    for obs_hash in hash2obs:
      self._hash2obs[obs_hash] = hash2obs[obs_hash]

  def _create_observation_tensor(self):
    hashes = list(self._hash2obs.keys())
    observations = jnp.asarray([self._hash2obs[key][0] for key in hashes], dtype=jnp.float32)
    actions = jnp.asarray([self._hash2obs[key][1] for key in hashes], dtype=jnp.int32)
    rewards = jnp.asarray([self._hash2obs[key][2] for key in hashes], dtype=jnp.float32)
    return hashes, OAR(observations, actions, rewards)
