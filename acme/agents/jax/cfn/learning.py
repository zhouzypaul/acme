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
import jax
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
      ):

    if not use_stale_rewards:
      raise NotImplementedError("Not Implemented (Error)")

    self._direct_rl_learner = direct_rl_learner_factory(
        cfn_network.direct_rl_networks, iterator)

  def step(self):
    self._direct_rl_learner.step()

  def get_variables(self, names: List[str]) -> List[Any]:
    return self._direct_rl_learner.get_variables(names)

  def save(self) -> NamedTuple:
    return self._direct_rl_learner.save()

  def restore(self, state: NamedTuple):
    return self._direct_rl_learner.restore(state)

  
  

