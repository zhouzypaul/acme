"""CFN Builder"""

from typing import Callable, Generic, Iterator, List, Optional

from acme import adders
from acme import core
from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import builders
from acme.agents.jax.builders import Networks, Policy
from acme.agents.jax.cfn import config as cfn_config
from acme.agents.jax.cfn import learning as cfn_learning
from acme.agents.jax.cfn import networks as cfn_networks
from acme.agents.jax.cfn.cfn import CFN
from acme.jax import networks as networks_lib
from acme.agents.jax.actors import CFNIntrinsicActor
from acme.agents.jax.cfn.actor import get_actor_core as get_cfn_actor_core
from acme.datasets import reverb as datasets
from acme.jax.types import Policy
from acme.utils import counting
from acme.utils import loggers
from acme.jax import variable_utils
from acme.jax import utils
import jax
import optax
import reverb
from reverb import rate_limiters

from acme.adders import reverb as adders_reverb
from acme.adders.reverb import structured
from acme.agents.jax.r2d2.actor import R2D2Policy
from reverb import structured_writer as sw


def _make_cfn_adder_config(step_spec, table_name):
  return structured.create_n_step_transition_config(
        step_spec,
        n_step=1,
        # n_step=2,
        table=table_name)



class CFNBuilder(Generic[cfn_networks.DirectRLNetworks, Policy],
                 builders.ActorLearnerBuilder[cfn_networks.CFNNetworks, Policy,
                                              reverb.ReplaySample]):
  """CFN Builder."""

  def __init__(
      self,
      rl_agent: builders.ActorLearnerBuilder[cfn_networks.DirectRLNetworks,
                                             Policy, reverb.ReplaySample],
      config: cfn_config.CFNConfig,
      logger_fn: Callable[[], loggers.Logger] = lambda: None,
  ):
    self._rl_agent = rl_agent
    self._config = config
    self._logger_fn = logger_fn

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: cfn_networks.CFNNetworks[cfn_networks.DirectRLNetworks],
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
      cfn: Optional[CFN] = None
  ) -> core.Learner:
    direct_rl_learner_key, cfn_learner_key = jax.random.split(random_key)

    counter = counter or counting.Counter()
    direct_rl_counter = counting.Counter(counter, 'direct_rl')

    def direct_rl_learner_factory(
        networks: cfn_networks.DirectRLNetworks,
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

    return cfn_learning.CFNLearner(
        direct_rl_learner_factory=direct_rl_learner_factory,
        iterator=dataset,
        optimizer=optimizer,
        cfn_network=networks,
        rng_key=cfn_learner_key,
        grad_updates_per_batch=self._config.num_sgd_steps_per_step,
        is_sequence_based=self._config.is_sequence_based,
        counter=counter,
        logger=logger_fn('learner'),
        intrinsic_reward_coefficient=self._config.intrinsic_reward_coefficient,
        extrinsic_reward_coefficient=self._config.extrinsic_reward_coefficient,
        use_stale_rewards=self._config.use_stale_rewards,
        cfn=cfn,
        value_plotting_freq=self._config.value_plotting_freq,)

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
      force_cpu: bool = False,
      cfn_variable_source: Optional[core.VariableSource] = None,
      cfn_adder: Optional[adders.Adder] = None
  ) -> core.Actor:
    if self._config.use_stale_rewards:
      q_learner_variable_client = variable_utils.VariableClient(
        variable_source,
        key=['actor_variables'],
        update_period=self._rl_agent._config.variable_update_period
      )
      cfn_variable_client = variable_utils.VariableClient(
        cfn_variable_source,
        key=[""],
        update_period=self._config.variable_update_period
      ) if cfn_variable_source else None

      actor_backend = 'cpu' if force_cpu else self._rl_agent._config.actor_backend
      print('actor backend', actor_backend)

      return CFNIntrinsicActor(
        cfn_variable_client=cfn_variable_client,
        intrinsic_reward_scale=self._config.intrinsic_reward_coefficient,
        extrinsic_reward_scale=self._config.extrinsic_reward_coefficient,
        actor=policy,
        random_key=random_key,
        variable_client=q_learner_variable_client,
        adder=adder,
        cfn_adder=cfn_adder
      )

    return self._rl_agent.make_actor(random_key, policy, environment_spec,
                                     variable_source, adder)
  
  def make_policy(self,
    networks: Networks,
    environment_spec: specs.EnvironmentSpec,
    evaluation: bool = False
  ) -> R2D2Policy:
    rl_actor_core = self._rl_agent.make_policy(
      networks.direct_rl_networks, environment_spec, evaluation)
    if self._config.use_stale_rewards:
      return get_cfn_actor_core(
        networks, rl_actor_core, self._config.intrinsic_reward_coefficient,
        self._config.extrinsic_reward_coefficient, self._config.use_reward_normalization,
        self._config.cfn_output_dimensions)
    return rl_actor_core

  def make_cfn_object(
      self,
      random_key: networks_lib.PRNGKey,
      networks: cfn_networks.CFNNetworks[cfn_networks.DirectRLNetworks],
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ):
    optimizer = optax.adam(learning_rate=self._config.cfn_learning_rate)
    return CFN(
      networks=networks,
      random_key=random_key,
      max_priority_weight=self._config.max_priority_weight,
      iterator=dataset,
      optimizer=optimizer,
      cfn_replay_table_name=self._config.cfn_replay_table_name,
      replay_client=replay_client,
      counter=counter,
      logger=logger_fn('cfn_object'),
      bonus_plotting_freq=self._config.bonus_plotting_freq,
    )
  
  def make_cfn_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: R2D2Policy
  ) -> List[reverb.Table]:
    """Creates reverb tables for the algorithm."""
    if self._config.samples_per_insert:
      samples_per_insert_tolerance = (
          self._config.samples_per_insert_tolerance_rate *
          self._config.samples_per_insert)
      error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
      limiter = rate_limiters.SampleToInsertRatio(
          min_size_to_sample=self._config.min_replay_size,
          samples_per_insert=self._config.samples_per_insert,
          error_buffer=error_buffer)
    else:
      print(f'[CFNBuilder] Using MinSize Rate Limiter')
      limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)
    dummy_actor_state = policy.init(jax.random.PRNGKey(0))
    extras_spec = policy.get_extras(dummy_actor_state)
    print(f'[CFNBuilder] Extras Spec = {extras_spec}')
    step_spec = self.get_step_spec(policy, environment_spec)
    signature = sw.infer_signature(
      configs=_make_cfn_adder_config(
                  step_spec,
                  self._config.cfn_replay_table_name),
      step_spec=step_spec)
    return [
        reverb.Table(
            name=self._config.cfn_replay_table_name,
            # sampler=reverb.selectors.Uniform(),
            sampler=reverb.selectors.Prioritized(
              self._config.priority_exponent),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=signature
          )
    ]

  def make_cfn_dataset_iterator(self, replay_client):
    """Creates a dataset iterator to use for learning."""
    dataset = datasets.make_reverb_dataset(
        table=self._config.cfn_replay_table_name,
        server_address=replay_client.server_address,
        batch_size=self._batch_size_per_device,
        prefetch_size=self._config.prefetch_size)

    return utils.multi_device_put(
        dataset.as_numpy_iterator(),
        jax.local_devices(),
        split_fn=utils.keep_key_on_host)

  def make_cfn_adder(self,
      replay_client: reverb.Client,
      environment_spec: Optional[specs.EnvironmentSpec],
      policy: Optional[R2D2Policy],
  ) -> Optional[adders.Adder]:
    """Creates an adder which handles observations."""
    # del environment_spec, policy
    # return adders_reverb.NStepTransitionAdder(
    #     priority_fns={self._config.cfn_replay_table_name: None},  # NOTE: initial priority is 1.
    #     client=replay_client,
    #     n_step=1,
    #     discount=self._rl_agent._config.discount)
    step_spec = self.get_step_spec(policy, environment_spec)
    adder_config = _make_cfn_adder_config(
      step_spec,
      self._config.cfn_replay_table_name)
    return adders_reverb.StructuredAdder(
      client=replay_client,
      max_in_flight_items=5,
      configs=adder_config,
      step_spec=step_spec
    )

  def get_step_spec(self, policy, environment_spec):
    dummy_actor_state = policy.init(jax.random.PRNGKey(0))
    extras_spec = policy.get_extras(dummy_actor_state)
    return structured.create_step_spec(
        environment_spec=environment_spec, extras_spec=extras_spec)

  @property
  def _batch_size_per_device(self) -> int:
    """Splits batch size across all learner devices evenly."""
    # TODO(bshahr): Using jax.device_count will not be valid when colocating
    # learning and inference.
    return self._config.batch_size // jax.device_count()
