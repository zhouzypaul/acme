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

"""RND Builder."""

from typing import Callable, Generic, Iterator, List, Optional

from acme import adders
from acme import core
from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import builders
from acme.agents.jax.rnd import config as rnd_config
from acme.agents.jax.rnd import learning as rnd_learning
from acme.agents.jax.rnd import networks as rnd_networks
from acme.jax import networks as networks_lib
from acme.agents.jax.actors import GenericIntrinsicActor
from acme.agents.jax.rnd.actor import get_actor_core as get_rnd_actor_core
from acme.jax.types import Policy
from acme.utils import counting
from acme.utils import loggers
from acme.jax import variable_utils
import jax
import optax
import reverb


class RNDBuilder(Generic[rnd_networks.DirectRLNetworks, Policy],
                 builders.ActorLearnerBuilder[rnd_networks.RNDNetworks, Policy,
                                              reverb.ReplaySample]):
  """RND Builder."""

  def __init__(
      self,
      rl_agent: builders.ActorLearnerBuilder[rnd_networks.DirectRLNetworks,
                                             Policy, reverb.ReplaySample],
      config: rnd_config.RNDConfig,
      logger_fn: Callable[[], loggers.Logger] = lambda: None,
  ):
    """Implements a builder for RND using rl_agent as forward RL algorithm.

    Args:
      rl_agent: The standard RL agent used by RND to optimize the generator.
      config: A config with RND HPs.
      logger_fn: a logger factory for the rl_agent's learner.
    """
    self._rl_agent = rl_agent
    self._config = config
    self._logger_fn = logger_fn

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: rnd_networks.RNDNetworks[rnd_networks.DirectRLNetworks],
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    direct_rl_learner_key, rnd_learner_key = jax.random.split(random_key)

    counter = counter or counting.Counter()
    direct_rl_counter = counting.Counter(counter, 'direct_rl')

    def direct_rl_learner_factory(
        networks: rnd_networks.DirectRLNetworks,
        dataset: Iterator[reverb.ReplaySample]) -> core.Learner:
      return self._rl_agent.make_learner(
          direct_rl_learner_key,
          networks,
          dataset,
          logger_fn=lambda name: self._logger_fn('direct_rl_learner'), # I guess you need a name for the RL learner's logger...
          environment_spec=environment_spec,
          replay_client=replay_client,
          counter=direct_rl_counter)

    optimizer = optax.adam(learning_rate=self._config.predictor_learning_rate)

    return rnd_learning.RNDLearner(
        direct_rl_learner_factory=direct_rl_learner_factory,
        iterator=dataset,
        optimizer=optimizer,
        rnd_network=networks,
        rng_key=rnd_learner_key,
        is_sequence_based=self._config.is_sequence_based,
        grad_updates_per_batch=self._config.num_sgd_steps_per_step,
        counter=counter,
        intrinsic_reward_coefficient=self._config.intrinsic_reward_coefficient,
        extrinsic_reward_coefficient=self._config.extrinsic_reward_coefficient,
        logger=logger_fn('learner'),
        use_stale_rewards=self._config.use_stale_rewards)

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: Policy,
  ) -> List[reverb.Table]:
    return self._rl_agent.make_replay_tables(environment_spec, policy)

  def make_dataset_iterator(  # pytype: disable=signature-mismatch  # overriding-return-type-checks
      self,
      replay_client: reverb.Client) -> Optional[Iterator[reverb.ReplaySample]]:
    return self._rl_agent.make_dataset_iterator(replay_client)

  def make_adder(self, replay_client: reverb.Client,
                 environment_spec: Optional[specs.EnvironmentSpec],
                 policy: Optional[Policy]) -> Optional[adders.Adder]:
    return self._rl_agent.make_adder(replay_client, environment_spec, policy)

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: Policy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
      force_cpu: bool = False
  ) -> core.Actor:
    if self._config.use_stale_rewards:
      variable_client = variable_utils.VariableClient(
        variable_source,
        key=['actor_variables', 'rnd_training_state'],
        update_period=self._rl_agent._config.variable_update_period)

      actor_backend = 'cpu' if force_cpu else self._rl_agent._config.actor_backend
      print('actor backend', actor_backend)

      return GenericIntrinsicActor(
        self._config.intrinsic_reward_coefficient,
        self._config.extrinsic_reward_coefficient,
        actor=policy,
        random_key=random_key,
        variable_client=variable_client,
        adder=adder,
        condition_actor_on_intrinsic_reward=self._config.condition_actor_on_intrinsic_reward,
        use_stale_rewards=self._config.use_stale_rewards,
      )
    return self._rl_agent.make_actor(random_key, policy, environment_spec,
                                     variable_source, adder)

  def make_policy(self,
                  networks: rnd_networks.RNDNetworks,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool = False) -> actor_core_lib.FeedForwardPolicy:
    """Construct the policy."""
    rl_actor_core = self._rl_agent.make_policy(
      networks.direct_rl_networks, environment_spec, evaluation)
    if self._config.use_stale_rewards:
      return get_rnd_actor_core(
        networks,
        rl_actor_core,
        self._config.intrinsic_reward_coefficient,
        self._config.extrinsic_reward_coefficient,
        self._config.condition_actor_on_intrinsic_reward)
    return rl_actor_core
