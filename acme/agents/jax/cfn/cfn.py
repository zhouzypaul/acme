"""Main CFN run-loop."""

import collections
import os

import pickle
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
from acme.utils.paths import get_save_directory
from acme.wrappers.observation_action_reward import OAR
from acme.agents.jax.cfn.learning import CFNTrainingState
from acme.agents.jax.cfn.networks import compute_cfn_reward
from acme.agents.jax.cfn import norm_lib

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
               cfn_replay_table_name: str,
               replay_client: Optional[reverb.Client] = None,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None,
               bonus_plotting_freq = 500,  # set to -1 to disable plotting.
               clip_random_prior_output: float = -1,
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

      if clip_random_prior_output > 0:
        normalized_rp_output = jnp.clip(
          normalized_rp_output,
          -clip_random_prior_output, clip_random_prior_output)

      pred = predictor_output + jax.lax.stop_gradient(normalized_rp_output)
      
      target = sample.data.extras['coin_flips']
      target = target.squeeze(1)  # TODO(ab/sl): indexing is awkward and unreadable rn

      assert pred.shape == target.shape, (pred.shape, target.shape)

      intrinsic_reward = networks.get_reward(
        predictor_output=predictor_output,
        normalized_rp_output=normalized_rp_output,
        original_reward=0.)
      
      times_sampled = jnp.maximum(sample.info.times_sampled, 1)
      seen_priority = 1. / times_sampled
      novelty_priority = jnp.clip(intrinsic_reward ** 2, 0., 1.)
      
      priorities = (0.5 * seen_priority) + (0.5 * novelty_priority)
      assert seen_priority.shape == novelty_priority.shape, (seen_priority.shape, novelty_priority.shape)

      log_dict = {
        'batch_rint_mean': intrinsic_reward.mean(),
        'batch_rint_min': intrinsic_reward.min(),
        'batch_rint_max': intrinsic_reward.max(),
        'max_abs_pred': jnp.abs(pred).max(),
        'max_abs_rp': jnp.abs(normalized_rp_output).max(),
        'max_priority': priorities.max(),
        'min_priority': priorities.min(),
        'mean_std': random_prior_std.mean(),
        'mean_inverse_std': (1. / random_prior_std).mean(),
        'max_inverse_std': (1. / random_prior_std).max(),
        'max_seen_priority': seen_priority.max(),
        'max_novelty_priority': novelty_priority.max(),
        'max_times_sampled': times_sampled.max(),
        'min_times_sampled': jnp.min(sample.info.times_sampled),
      }

      return optax.l2_loss(pred, target).mean(), (rp_output, priorities, intrinsic_reward, log_dict)

    def sgd_step(
      state: CFNTrainingState,
      samples: reverb.ReplaySample
    ) -> Tuple[CFNTrainingState, jnp.ndarray, Dict[str, jnp.ndarray]]:
      """Performs an update step, averaging over pmap replicas."""
      grad_fn = jax.value_and_grad(loss, has_aux=True)
      # key, key_grad = jax.random.split(state.random_key)
      (loss_value, (random_prior_output, priorities, r_int, log_dict)), gradients = grad_fn(
        state.params,
        state.target_params,
        # key_grad,
        samples,
        state.random_prior_mean,
        jnp.sqrt(state.random_prior_var + 1e-12))

      # Average gradients over pmap replicas before optimizer update.
      gradients = jax.lax.pmean(gradients, _PMAP_AXIS_NAME)

      # Apply optimizer updates.
      updates, new_opt_state = optimizer.update(gradients, state.optimizer_state, params=state.params)
      new_params = optax.apply_updates(state.params, updates)

      # Periodically update target networks.
      steps = state.steps + 1

      new_cfn_state = CFNTrainingState(
        optimizer_state=new_opt_state,
        params=new_params,
        target_params= state.target_params,
        steps=steps,
        reward_mean=state.reward_mean,
        reward_var=state.reward_var,
        reward_second_moment=state.reward_second_moment,
        random_prior_mean=state.random_prior_mean,
        random_prior_var=state.random_prior_var,
        random_prior_second_moment=state.random_prior_second_moment,
        batches_updated_on=state.batches_updated_on,
      )

      new_cfn_state = update_mean_variance_stats(new_cfn_state, random_prior_output, r_int)
      return new_cfn_state, priorities, {
        'loss': loss_value,
        **log_dict,
        'random_prior_mean': new_cfn_state.random_prior_mean.mean(),
        'random_prior_var': new_cfn_state.random_prior_var.mean(),
        'batches_updated_on': new_cfn_state.batches_updated_on,
        'reward_mean': new_cfn_state.reward_mean,
        'reward_variance': new_cfn_state.reward_var,
      }

    def update_priorities(
        keys_and_priorities: Tuple[jnp.ndarray, jnp.ndarray]):
      keys, priorities = keys_and_priorities
      keys, priorities = tree.map_structure(
          # Fetch array and combine device and batch dimensions.
          lambda x: utils.fetch_devicearray(x).reshape((-1,) + x.shape[2:]),
          (keys, priorities))
      replay_client.mutate_priorities(  # pytype: disable=attribute-error
          table=cfn_replay_table_name,
          updates=dict(zip(keys, priorities)))

    def update_mean_variance_stats(
        state: CFNTrainingState,
        random_prior_output: networks_lib.NetworkOutput,
        unscaled_intrinsic_reward: networks_lib.NetworkOutput
      ) -> CFNTrainingState:
      
      new_reward_mean, new_reward_second_moment, _ = norm_lib.welford(
        state.reward_mean,
        state.reward_second_moment,
        state.batches_updated_on,
        unscaled_intrinsic_reward)
      new_rp_mean, new_rp_second_moment, new_n_batches = norm_lib.welford(
        state.random_prior_mean,
        state.random_prior_second_moment,
        state.batches_updated_on,
        random_prior_output)
      new_reward_var = new_reward_second_moment / new_n_batches
      new_rp_var = new_rp_second_moment / new_n_batches

      new_state = CFNTrainingState(
        optimizer_state=state.optimizer_state,
        params=state.params,
        target_params=state.target_params,
        steps=state.steps,
        reward_mean=new_reward_mean,
        reward_var=new_reward_var,
        reward_second_moment=new_reward_second_moment,
        random_prior_mean=new_rp_mean,
        random_prior_var=new_rp_var,
        random_prior_second_moment=new_rp_second_moment,
        batches_updated_on=new_n_batches,
      )

      return new_state
    
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
        reward_second_moment=0.,
        random_prior_mean=jnp.zeros((20,), dtype=jnp.float32),  # TODO(ab/sl): pass in output_dims
        random_prior_var=jnp.ones((20,), dtype=jnp.float32),
        random_prior_second_moment=jnp.zeros((20,), dtype=jnp.float32),
        batches_updated_on=0))
    
    # Keep track of ground-truth counts for debugging/visualizations.
    self._hash2obs = {}  # map goal hash to obs
    self._hash2counts = collections.defaultdict(int)
    self._count_dict_lock = threading.Lock()
    self._networks = networks
    self._bonus_plotting_freq = bonus_plotting_freq
    self._bonus_prediction_errors = []
    self._num_unique_states_visited = []

    # Create logging directories.
    base_dir = get_save_directory()
    self._spatial_plotting_dir = os.path.join(base_dir, 'plots', 'spatial_bonus')
    self._scatter_plotting_dir = os.path.join(base_dir, 'plots', 'true_vs_approx_scatterplots')
    self._scalar_plotting_dir = os.path.join(base_dir, 'plots', 'scalar_plots')
    os.makedirs(self._spatial_plotting_dir, exist_ok=True)
    os.makedirs(self._scatter_plotting_dir, exist_ok=True)
    os.makedirs(self._scalar_plotting_dir, exist_ok=True)
    print(f'Created CFN Object with clipping factor: {clip_random_prior_output}')

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

    # Attempt to write logs.
    self._logger.write({**metrics, **counts, 'cfn_iteration_time': time.time() - start})

    # Update priorities in replay.
    if self._replay_client:
      self._async_priority_updater.put((keys, priorities))

    if self._bonus_plotting_freq > 0 and self._state.steps > 0 and \
        self._state.steps % self._bonus_plotting_freq == 0 and \
        len(self._hash2obs) > 1:
      self._make_cfn_bonus_plots()

  def _make_cfn_bonus_plots(self):
    state: CFNTrainingState = utils.get_from_first_device(self._state)
    hashes, oar = self._create_observation_tensor()

    observation_tensor = oar.observation

    predicted_bonuses = compute_cfn_reward(
      state.params,
      state.target_params,
      oar,
      self._networks,
      state.random_prior_mean,
      jnp.sqrt(state.random_prior_var + 1e-12)
    )
    
    assert predicted_bonuses.shape == (len(hashes),), predicted_bonuses.shape

    approx_bonus_info = {k: v for k, v in zip(hashes, predicted_bonuses)}
    plotting_utils.plot_spatial_count_or_bonus(
      true_count_info=self._hash2counts,
      approx_bonus_info=approx_bonus_info,
      save_path=os.path.join(self._spatial_plotting_dir, f'spatial_bonus_{state.steps}.png'),
    )
    plotting_utils.plot_true_vs_approx_bonus(
      true_count_info=self._hash2counts,
      approx_bonus_info=approx_bonus_info,
      save_path=os.path.join(self._scatter_plotting_dir, f'bonus_scatterplot_{state.steps}.png'),
    )

    bonus_prediction_error = plotting_utils.compute_bonus_prediction_error(
      true_count_info=self._hash2counts,
      approx_bonus_info=approx_bonus_info)
    self._bonus_prediction_errors.append(bonus_prediction_error)
    plotting_utils.plot_quantity_over_iteration(self._bonus_prediction_errors,
                                           save_path=os.path.join(self._scalar_plotting_dir, 'mse.png'),
                                           quantity_name='MSE')
    with open(os.path.join(self._scalar_plotting_dir, 'mse.pkl'), 'wb+') as f:
      pickle.dump(self._bonus_prediction_errors, f)

    self._num_unique_states_visited.append(len(self._hash2obs))
    plotting_utils.plot_quantity_over_iteration(self._num_unique_states_visited,
                                                save_path=os.path.join(self._scalar_plotting_dir, 'num_obs.png'),
                                                quantity_name='num_unique_states_visited')
    with open(os.path.join(self._scalar_plotting_dir, 'num_visited_states.pkl'), 'wb+') as f:
      pickle.dump(self._num_unique_states_visited, f)
      
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
