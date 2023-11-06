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

"""RND learner implementation."""

import functools
import time
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.agents.jax.rnd import networks as rnd_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import reverb_utils
from acme.jax.utils import PrefetchingSplit
import jax
import jax.numpy as jnp
import optax
import reverb


class RNDTrainingState(NamedTuple):
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  params: networks_lib.Params
  target_params: networks_lib.Params
  steps: int
  reward_mean: float
  reward_var: float
  reward_squared_mean: float
  observation_mean: jnp.ndarray
  observation_var: jnp.ndarray
  observation_squared_mean: jnp.ndarray
  states_updated_on: int


class GlobalTrainingState(NamedTuple):
  """Contains training state of the RND learner."""
  rewarder_state: RNDTrainingState
  learner_state: Any


RNDLoss = Callable[[networks_lib.Params, networks_lib.Params, types.Transition],
                   float]


def rnd_update_step(
    state: RNDTrainingState, transitions: types.Transition,
    loss_fn: RNDLoss, optimizer: optax.GradientTransformation
) -> Tuple[RNDTrainingState, Dict[str, jnp.ndarray]]:
  """Run an update steps on the given transitions.

  Args:
    state: The learner state.
    transitions: Transitions to update on.
    loss_fn: The loss function.
    optimizer: The optimizer of the predictor network.

  Returns:
    A new state and metrics.
  """
  loss, grads = jax.value_and_grad(loss_fn)(
      state.params,
      state.target_params,
      transitions=transitions,
      observation_mean=state.observation_mean,
      observation_var=state.observation_var,)

  update, optimizer_state = optimizer.update(grads, state.optimizer_state)
  params = optax.apply_updates(state.params, update)

  new_state = RNDTrainingState(
      optimizer_state=optimizer_state,
      params=params,
      target_params=state.target_params,
      steps=state.steps + 1,
      reward_mean=state.reward_mean,
      reward_var=state.reward_var,
      reward_squared_mean=state.reward_squared_mean,
      observation_mean=state.observation_mean,
      observation_var=state.observation_var,
      observation_squared_mean=state.observation_squared_mean,
      states_updated_on=state.states_updated_on,
  )
  return new_state, {'rnd_loss': loss, 'rnd_reward_mean': state.reward_mean, 'rnd_reward_var': state.reward_var}


def rnd_loss(
    predictor_params: networks_lib.Params,
    target_params: networks_lib.Params,
    transitions: types.Transition,
    networks: rnd_networks.RNDNetworks,
    observation_mean: jnp.ndarray,
    observation_var: jnp.ndarray,
) -> float:
  """The Random Network Distillation loss.

  See https://arxiv.org/pdf/1810.12894.pdf A.2

  Args:
    predictor_params: Parameters of the predictor
    target_params: Parameters of the target
    transitions: Transitions to compute the loss on.
    networks: RND networks

  Returns:
    The MSE loss as a float.
  """

  safe_observation_var = jnp.maximum(observation_var, 1e-6)
  # We're going to normalize by std because its obviously right
  whitened_observation = (transitions.observation - observation_mean) / jnp.sqrt(safe_observation_var)
  whitened_clipped_observation = jnp.clip(whitened_observation, -5., 5.)

  target_output = networks.target.apply(target_params,
                                        whitened_clipped_observation,
                                        transitions.action)
  predictor_output = networks.predictor.apply(predictor_params,
                                              whitened_clipped_observation,
                                              transitions.action)
  return jnp.mean(jnp.square(target_output - predictor_output))

# TODO(ab/sl): Verify that RND learning is using GPU.

class RNDLearner(acme.Learner):
  """RND learner."""

  def __init__(
      self,
      direct_rl_learner_factory: Callable[[Any, Iterator[reverb.ReplaySample]],
                                          acme.Learner],
      iterator: Iterator[reverb.ReplaySample],
      optimizer: optax.GradientTransformation,
      rnd_network: rnd_networks.RNDNetworks,
      rng_key: jnp.ndarray,
      grad_updates_per_batch: int,
      is_sequence_based: bool,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      intrinsic_reward_coefficient=0.001,
      extrinsic_reward_coefficient=1.0,
      use_stale_rewards: bool = False):

    self._is_sequence_based = is_sequence_based

    # No need to recompute intrinsic rewards when this is true.
    self._use_stale_rewards = use_stale_rewards

    target_key, predictor_key = jax.random.split(rng_key)
    target_params = rnd_network.target.init(target_key)
    predictor_params = rnd_network.predictor.init(predictor_key)
    optimizer_state = optimizer.init(predictor_params)

    self._state = RNDTrainingState(
        optimizer_state=optimizer_state,
        params=predictor_params,
        target_params=target_params,
        steps=0,
        reward_mean=0,
        reward_var=1,
        reward_squared_mean=0,
        observation_mean=0,
        observation_var=1,
        observation_squared_mean=0,
        states_updated_on=0,
        )

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner',
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key=self._counter.get_steps_key())

    loss = functools.partial(rnd_loss, networks=rnd_network)
    self._update = functools.partial(rnd_update_step,
                                     loss_fn=loss,
                                     optimizer=optimizer)
    self._update = utils.process_multiple_batches(self._update,
                                                  grad_updates_per_batch)
    self._update = jax.jit(self._update)

    # self._get_reward = jax.jit(
    self._get_unscaled_intrinsic_reward = jax.jit(
        functools.partial(
            rnd_networks.compute_rnd_reward, networks=rnd_network))

    self._update_obs_and_reward_norm = jax.jit(
      self._update_obs_and_reward_norm_prejit)

    # There's some logic to breaking it up like this so the first batch is normalized, etc.
    # We don't do it yet but its a possibility for later.
    self._update_obs_norm_and_count = jax.jit(
      self._update_obs_norm_and_count_prejit)
    self._update_reward_norm = jax.jit(
      self._update_reward_norm_prejit)

    # Generator expression that works the same as an iterator.
    # https://pymbook.readthedocs.io/en/latest/igd.html#generator-expressions
    updated_iterator = (self._process_sample(sample) for sample in iterator)

    self._direct_rl_learner = direct_rl_learner_factory(
        rnd_network.direct_rl_networks, updated_iterator)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

    # set weightings
    self.intrinsic_reward_coefficient = intrinsic_reward_coefficient
    self.extrinsic_reward_coefficient = extrinsic_reward_coefficient

  def _update_obs_norm_and_count_prejit(self, state, transitions) -> RNDTrainingState:
    num_transitions = jnp.prod(jnp.array(transitions.observation.shape[3:]))
    new_states_updated_on = state.states_updated_on + num_transitions
    delta_observation_entries = (transitions.observation - state.observation_mean)
    new_observation_mean = state.observation_mean + (num_transitions * delta_observation_entries.mean(axis=(0,1,2)) / new_states_updated_on)
    delta_observation_squared_entries = (transitions.observation**2 - state.observation_squared_mean)
    new_observation_squared_mean = state.observation_squared_mean + (num_transitions*delta_observation_squared_entries.mean(axis=(0,1,2)) / new_states_updated_on)
    new_observation_var = new_observation_squared_mean - new_observation_mean**2

    new_state = RNDTrainingState(
        optimizer_state=state.optimizer_state,
        params=state.params,
        target_params=state.target_params,
        steps=state.steps,
        reward_mean=state.reward_mean,
        reward_var=state.reward_var,
        reward_squared_mean=state.reward_squared_mean,
        observation_mean=new_observation_mean,
        observation_var=new_observation_var,
        observation_squared_mean=new_observation_squared_mean,
        states_updated_on=new_states_updated_on,
    )

    return new_state
  
  def _update_reward_norm_prejit(self, state, unscaled_intrinsic_reward) -> RNDTrainingState:
    num_transitions = jnp.prod(jnp.array(unscaled_intrinsic_reward.shape))
    # new_states_updated_on = state.states_updated_on + num_transitions # This has already been done
    delta_reward_entries = (unscaled_intrinsic_reward - state.reward_mean)
    delta_reward = delta_reward_entries.mean()
    new_reward_mean = state.reward_mean + (num_transitions * delta_reward / state.states_updated_on)
    delta_reward_squared_entries = (unscaled_intrinsic_reward**2 - state.reward_squared_mean)
    new_reward_squared_mean = state.reward_squared_mean + (num_transitions*delta_reward_squared_entries.mean() / state.states_updated_on)
    new_reward_var = new_reward_squared_mean - new_reward_mean**2

    new_state = RNDTrainingState(
        optimizer_state=state.optimizer_state,
        params=state.params,
        target_params=state.target_params,
        steps=state.steps,
        reward_mean=new_reward_mean,
        reward_var=new_reward_var,
        reward_squared_mean=new_reward_squared_mean,
        observation_mean=state.observation_mean,
        observation_var=state.observation_var,
        observation_squared_mean=state.observation_squared_mean,
        states_updated_on=state.states_updated_on,
    )

    return new_state

  def _update_obs_and_reward_norm_prejit(self, state, transitions, unscaled_intrinsic_reward) -> RNDTrainingState:

    # Updates reward_norm
    num_transitions = jnp.prod(jnp.array(unscaled_intrinsic_reward.shape))
    new_states_updated_on = state.states_updated_on + num_transitions
    delta_reward_entries = (unscaled_intrinsic_reward - state.reward_mean)
    delta_reward = delta_reward_entries.mean()
    new_reward_mean = state.reward_mean + (num_transitions * delta_reward / new_states_updated_on)
    delta_reward_squared_entries = (unscaled_intrinsic_reward**2 - state.reward_squared_mean)
    new_reward_squared_mean = state.reward_squared_mean + (num_transitions*delta_reward_squared_entries.mean() / new_states_updated_on)
    new_reward_var = new_reward_squared_mean - new_reward_mean**2

    # For now lets not do obs norm. I'll need access to the shape maybe?
    # I don't know, I'll at least need to know which are the batch elements.
    # shape is (1, batch_size, 49, 84, 84, 3). So, average over first 3.
    # Should I broadcast manually or trust it to do it? Let's trust it.
    assert len(transitions.observation.shape) == 6
    delta_observation_entries = (transitions.observation - state.observation_mean)
    new_observation_mean = state.observation_mean + (num_transitions * delta_observation_entries.mean(axis=(0,1,2)) / new_states_updated_on)
    delta_observation_squared_entries = (transitions.observation**2 - state.observation_squared_mean)
    new_observation_squared_mean = state.observation_squared_mean + (num_transitions*delta_observation_squared_entries.mean(axis=(0,1,2)) / new_states_updated_on)
    new_observation_var = new_observation_squared_mean - new_observation_mean**2

    new_state = RNDTrainingState(
        optimizer_state=state.optimizer_state,
        params=state.params,
        target_params=state.target_params,
        steps=state.steps,
        reward_mean=new_reward_mean,
        reward_var=new_reward_var,
        reward_squared_mean=new_reward_squared_mean,
        observation_mean=new_observation_mean,
        observation_var=new_observation_var,
        observation_squared_mean=new_observation_squared_mean,
        states_updated_on=new_states_updated_on,
    )

    return new_state


  def _process_sample(self, sample: reverb.ReplaySample) -> reverb.ReplaySample:
    """Uses the replay sample to train and update its reward.

    Args:
      sample: Replay sample to train on.

    Returns:
      The sample replay sample with an updated reward.
    """

    is_prefetch = isinstance(sample, PrefetchingSplit)
    if is_prefetch:
      prefetch_split_sample = sample
      sample = sample.device

    # TODO(ab/sl): Bug - we are changing r_t but not the reward part of OAR.

    # transitions = (
    #   s_t = <o_t, a_{t-1}, r_{t-1}>,
    #   a_t,
    #   r_t = R_ext(o_{t+1}) + R_int(o_t),
    #   s_{t+1} = <o_{t+1}, a_t, r_t>
    # )
    transitions = reverb_utils.replay_sample_to_sars_transition(
        sample, is_sequence=self._is_sequence_based)

    extrinsic_reward = transitions.reward

    # We might be dealing with an OAR tuple, in which case
    if hasattr(transitions.observation, 'observation'):
      transitions = transitions.observation # its already an OAR.

    # Update obs norm before doing grads
    # self._state = self._update_obs_norm_and_count(self._state, transitions)

    # Ideally we do a decent number of steps before starting this part, so that normalization is correct. We could skip reward for a bit as well.
    self._state, metrics = self._update(self._state, transitions)
    
    # combined inrtinsic extrinsic
    # TODO(sam): refactor so it doesn't look like we are adding r_ext twice.
    unscaled_intrinsic_rewards = self._get_unscaled_intrinsic_reward(self._state.params, self._state.target_params,
                               transitions,
                               observation_mean=self._state.observation_mean, observation_var=self._state.observation_var)

    # Update reward norm before scaling rewards
    # self._state = self._update_reward_norm(self._state, unscaled_intrinsic_rewards)

    # Not using safe reward_var because it'll never drop to zero, but can be very small sometimes.
    normalized_intrinsic_rewards = (unscaled_intrinsic_rewards - self._state.reward_mean) / jnp.sqrt(self._state.reward_var) # doing sqrt because it seems more reasonable, and that's what paper says

    rewards_logging_dict = {
      'unscaled_rnd_batch_mean': unscaled_intrinsic_rewards.mean().item(),
      'normalized_rnd_batch_mean': normalized_intrinsic_rewards.mean().item(),
      'unscaled_rnd_batch_var': unscaled_intrinsic_rewards.var().item(),
      'normalized_rnd_batch_var': normalized_intrinsic_rewards.var().item(),
    }

    # Update of rewards has to happen after calculated, but maybe observation can happen before.
    # Possibly we should have a bleed in time as well. Definitely.
    # Just going to do them at the same time because that's how it used to be,
    # and it doesn't seem like it should matter after a certain point.
    self._state = self._update_obs_and_reward_norm(self._state, transitions, unscaled_intrinsic_rewards)

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time
    counts = self._counter.increment(steps=1, walltime=elapsed_time)

    # Attempts to write the logs.
    self._logger.write({**metrics, **counts, **rewards_logging_dict})

    # sample = sample._replace(data=sample.data._replace(reward=rewards))
    if not self._use_stale_rewards:
      combined_rewards = (normalized_intrinsic_rewards * self.intrinsic_reward_coefficient) + \
        (extrinsic_reward * self.extrinsic_reward_coefficient)
      sample = sample._replace(data=sample.data._replace(reward=combined_rewards))
    
    if is_prefetch:
      sample = PrefetchingSplit(device=sample, host=prefetch_split_sample.host) # its a NamedTuple

    return sample

  def step(self):
    self._direct_rl_learner.step()

  def get_variables(self, names: List[str]) -> List[Any]:
    rnd_variables = {
      'rnd_training_state': self._state
    }

    learner_names = [name for name in names if name not in rnd_variables]
    learner_dict = {}
    if learner_names:
      learner_dict = dict(
          zip(learner_names,
              self._direct_rl_learner.get_variables(learner_names)))

    variables = [
        rnd_variables.get(name, learner_dict.get(name, None)) for name in names
    ]
    return variables

  def save(self) -> GlobalTrainingState:
    return GlobalTrainingState(
        rewarder_state=self._state,
        learner_state=self._direct_rl_learner.save())

  def restore(self, state: GlobalTrainingState):
    self._state = state.rewarder_state
    self._direct_rl_learner.restore(state.learner_state)
