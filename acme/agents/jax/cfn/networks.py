"""Network choices for CFN objective."""


import dataclasses
import functools
from typing import Callable, Generic, Tuple, TypeVar

from acme import specs
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils

import jax
import haiku as hk
import jax.numpy as jnp

DirectRLNetworks = TypeVar('DirectRLNetworks')


@dataclasses.dataclass
class CFNNetworks(Generic[DirectRLNetworks]):
  """Container of CFN networks factories."""
  target: networks_lib.FeedForwardNetwork
  predictor: networks_lib.FeedForwardNetwork
  # Function from predictor output, target output, and original reward to reward
  get_reward: Callable[
      [networks_lib.NetworkOutput, networks_lib.NetworkOutput, jnp.ndarray],
      jnp.ndarray]
  direct_rl_networks: DirectRLNetworks = None


def flips2bonus(
    prediction: networks_lib.NetworkOutput) -> networks_lib.NetworkOutput:
  return jnp.sqrt(jnp.mean(prediction ** 2, axis=-1))


def cfn_reward_fn(
    predictor_output: networks_lib.NetworkOutput,
    normalized_rp_output: networks_lib.NetworkOutput,
    original_reward: jnp.ndarray,
    intrinsic_reward_coefficient: float = 1.0,
    extrinsic_reward_coefficient: float = 0.0,
) -> jnp.ndarray:
  intrinsic_reward = flips2bonus(predictor_output + normalized_rp_output)
  # import ipdb; ipdb.set_trace()
  return (intrinsic_reward_coefficient * intrinsic_reward) + \
         (extrinsic_reward_coefficient * original_reward)


def make_networks(
    spec: specs.EnvironmentSpec,
    direct_rl_networks: DirectRLNetworks,
    layer_sizes: Tuple[int, ...] = (256, 256),
    intrinsic_reward_coefficient: float = 1.0,
    extrinsic_reward_coefficient: float = 0.0,
    use_orthogonal_initialization: bool = False,
) -> CFNNetworks[DirectRLNetworks]:
  """Creates networks used by the agent and returns CFNNetworks.

  Args:
    spec: Environment spec.
    direct_rl_networks: Networks used by a direct rl algorithm.
    layer_sizes: Layer sizes.
    intrinsic_reward_coefficient: Multiplier on intrinsic reward.
    extrinsic_reward_coefficient: Multiplier on extrinsic reward.

  Returns:
    The RND networks.
  """

  assert intrinsic_reward_coefficient == 1.0, 'we do scaling elsewhere'
  assert extrinsic_reward_coefficient == 0.0, 'we do scaling elsewhere'


  class ConvolutionalNetworkClass(hk.Module):
    """CNN Torso followed by a fully-connected linear layer."""

    def __init__(self, name = 'convolutional_network_class'):
      super().__init__(name=name)
      w_init = None
      if use_orthogonal_initialization:
        w_init = hk.initializers.Orthogonal(scale=jnp.sqrt(2))
      to_float = spec.observations.observation.dtype == jnp.uint8
      self._network = hk.Sequential([
        networks_lib.CFNAtariTorso(w_init=w_init, to_float=to_float),
        # TODO(ab/sl): test if fc_hidden is a good idea
        # hk.Linear(256),
        # jax.nn.relu,
        hk.Linear(20, w_init=w_init)
      ])
      # self.batched = hk.BatchApply(self._network, num_dims=1)

    def __call__(self, obs):
      # Here obs has shape (1, 84, 84)
      # return self.batched(obs) # hopefully we just need another index here...
      return self._network(obs)


  def _conv_linear_network(obs, act):
    del act
    network = ConvolutionalNetworkClass()
    return network(obs)

  target = hk.without_apply_rng(hk.transform(_conv_linear_network))
  predictor = hk.without_apply_rng(hk.transform(_conv_linear_network))

  # Create dummy observations and actions to create network parameters.
  dummy_obs = utils.zeros_like(spec.observations)
  from acme.wrappers.observation_action_reward import OAR

  if isinstance(dummy_obs, OAR):
    print('using OAR so I need to extract the observation')
    dummy_obs = dummy_obs.observation

  # For sequences we need to add time. Why is it 3? So its batch_size, sequence length, and then there's some extra dimension
  # that's always one. But, it comes from Reverb so I don't think its anything I did.
  # TODO(ab): Will we have a time dimension when we sample from the CFN replay buffer?
  # import ipdb; ipdb.set_trace()
  dummy_obs = utils.add_batch_dim(dummy_obs)  # (1, 84, 84, 3)
  # dummy_obs = utils.add_batch_dim(dummy_obs)
  # dummy_obs = utils.add_batch_dim(dummy_obs) # after 3 of these it has the right shape at least...

  print('[CFN/networks] dummy-obs shape: ', dummy_obs.shape)

  return CFNNetworks(
      target=networks_lib.FeedForwardNetwork(
          lambda key: target.init(key, dummy_obs, ()), target.apply),
      predictor=networks_lib.FeedForwardNetwork(
          lambda key: predictor.init(key, dummy_obs, ()), predictor.apply),
      direct_rl_networks=direct_rl_networks,
      get_reward=functools.partial(
          cfn_reward_fn,
          intrinsic_reward_coefficient=intrinsic_reward_coefficient,
          extrinsic_reward_coefficient=extrinsic_reward_coefficient))


def compute_cfn_reward(predictor_params: networks_lib.Params,
                       target_params: networks_lib.Params,
                       transitions: types.Transition,
                       networks: CFNNetworks,
                       random_prior_mean: jnp.ndarray,
                       random_prior_std: jnp.ndarray) -> jnp.ndarray:
  """Computes the intrinsic CFN reward for a given a batch of transitions.

  Args:
    predictor_params: Parameters of the CFN predictor network.
    target_params: Parameters of the random prior network.
    transitions: The samples to compute rewards for.
    networks: CFN networks

  Returns:
    The rewards corresponding to each of the input transitions.
  """
  assert random_prior_mean.shape == random_prior_std.shape
  
  learning_network_output = networks.predictor.apply(predictor_params,
                                                     transitions.observation,
                                                     transitions.action)
  random_prior_output = networks.target.apply(target_params,
                                              transitions.observation,
                                              transitions.action)
  # TODO(ab/sl): Maybe we should be limiting how small std can be like we did
  # https://github.com/samlobel/CFN/blob/main/intrinsic_motivation/intrinsic_rewards.py#L1167
  # Or maybe we clip it. Note that RND BBE doesn't do the above, though that's for reward, which is clipped
  # at a different time. 
  normalized_rp_output = (random_prior_output - random_prior_mean) / random_prior_std
  
  return networks.get_reward(learning_network_output, normalized_rp_output, transitions.reward)
