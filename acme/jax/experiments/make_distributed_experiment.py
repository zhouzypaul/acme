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

"""Program definition for a distributed layout based on a builder."""

import itertools
import math
from typing import Any, List, Optional

from acme import core
from acme import environment_loop
from acme import specs
from acme.agents.jax import actor_core
from acme.agents.jax import builders
from acme.jax import inference_server as inference_server_lib
from acme.jax import networks as networks_lib
from acme.jax import savers
from acme.jax import utils
from acme.jax import variable_utils
from acme.jax.experiments import config
from acme.jax import snapshotter
from acme.utils import counting
from acme.utils import lp_utils
import jax
import launchpad as lp
import reverb
import functools

ActorId = int
InferenceServer = inference_server_lib.InferenceServer[
    actor_core.SelectActionFn]


def make_distributed_experiment(
    experiment: config.ExperimentConfig[builders.Networks, Any, Any],
    num_actors: int,
    *,
    inference_server_config: Optional[
        inference_server_lib.InferenceServerConfig
    ] = None,
    num_learner_nodes: int = 1,
    num_actors_per_node: int = 1,
    num_inference_servers: int = 1,
    multiprocessing_colocate_actors: bool = False,
    multithreading_colocate_learner_and_reverb: bool = False,
    make_snapshot_models: Optional[
        config.SnapshotModelFactory[builders.Networks]
    ] = None,
    name: str = 'agent',
    program: Optional[lp.Program] = None,
    split_actor_specs=False,
) -> lp.Program:
  """Builds a Launchpad program for running the experiment.

  Args:
    experiment: configuration of the experiment.
    num_actors: number of actors to run.
    inference_server_config: If provided we will attempt to use
      `num_inference_servers` inference servers for selecting actions.
      There are two assumptions if this config is provided:
      1) The experiment's policy is an `ActorCore` and a
      `TypeError` will be raised if not.
      2) The `ActorCore`'s `select_action` method runs on
      unbatched observations.
    num_learner_nodes: number of learner nodes to run. When using multiple
      learner nodes, make sure the learner class does the appropriate pmap/pmean
      operations on the loss/gradients, respectively.
    num_actors_per_node: number of actors per one program node. Actors within
      one node are colocated in one or multiple processes depending on the value
      of multiprocessing_colocate_actors.
    num_inference_servers: number of inference servers to serve actors. (Only
      used if `inference_server_config` is provided.)
    multiprocessing_colocate_actors: whether to colocate actor nodes as
      subprocesses on a single machine. False by default, which means colocate
      within a single process.
    multithreading_colocate_learner_and_reverb: whether to colocate the learner
      and reverb nodes in one process. Not supported if the learner is spread
      across multiple nodes (num_learner_nodes > 1). False by default, which
      means no colocation.
    make_snapshot_models: a factory that defines what is saved in snapshots.
    name: name of the constructed program. Ignored if an existing program is
      passed.
    program: a program where agent nodes are added to. If None, a new program is
      created.

  Returns:
    The Launchpad program with all the nodes needed for running the experiment.
  """

  if multithreading_colocate_learner_and_reverb and num_learner_nodes > 1:
    raise ValueError(
        'Replay and learner colocation is not yet supported when the learner is'
        ' spread across multiple nodes (num_learner_nodes > 1). Please contact'
        ' Acme devs if this is a feature you want. Got:'
        '\tmultithreading_colocate_learner_and_reverb='
        f'{multithreading_colocate_learner_and_reverb}'
        f'\tnum_learner_nodes={num_learner_nodes}.')


  def build_replay():
    """The replay storage."""
    dummy_seed = 1
    spec = (
        experiment.environment_spec or
        specs.make_environment_spec(experiment.environment_factory(dummy_seed)))
    network = experiment.network_factory(spec)
    policy = config.make_policy(
        experiment=experiment,
        networks=network,
        environment_spec=spec,
        evaluation=False)
    return experiment.builder.make_replay_tables(spec, policy)

  def build_cfn_replay() -> List[reverb.Table]:
    """Replay storage for storing CFN transitions (single step with coins)."""
    dummy_seed = 1
    spec = (
        experiment.environment_spec or
        specs.make_environment_spec(experiment.environment_factory(dummy_seed)))
    network = experiment.network_factory(spec)
    policy = config.make_policy(
        experiment=experiment,
        networks=network,
        environment_spec=spec,
        evaluation=False)
    return experiment.builder.make_cfn_replay_tables(spec, policy)

  def build_model_saver(variable_source: core.VariableSource):
    assert experiment.checkpointing
    environment = experiment.environment_factory(0)
    spec = specs.make_environment_spec(environment)
    networks = experiment.network_factory(spec)
    models = make_snapshot_models(networks, spec)
    # TODO(raveman): Decouple checkpointing and snapshotting configs.
    return snapshotter.JAXSnapshotter(
        variable_source=variable_source,
        models=models,
        path=experiment.checkpointing.directory,
        subdirectory='snapshots',
        add_uid=experiment.checkpointing.add_uid)

  def build_counter():
    counter = counting.Counter(known_fields=('actor_steps', 'actor_episodes', 'learner_steps', )) # Various things we'd like in all
    if experiment.checkpointing:
      checkpointing = experiment.checkpointing
      counter = savers.CheckpointingRunner(
          counter,
          key='counter',
          subdirectory='counter',
          time_delta_minutes=checkpointing.time_delta_minutes,
          directory=checkpointing.directory,
          add_uid=checkpointing.add_uid,
          max_to_keep=checkpointing.max_to_keep,
          keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
          checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
      )
    return counter

  def build_learner(
      random_key: networks_lib.PRNGKey,
      replay: reverb.Client,
      counter: Optional[counting.Counter] = None,
      primary_learner: Optional[core.Learner] = None,
      cfn=None
  ):
    """The Learning part of the agent."""

    dummy_seed = 1
    spec = (
        experiment.environment_spec or
        specs.make_environment_spec(experiment.environment_factory(dummy_seed)))

    # Creates the networks to optimize (online) and target networks.
    networks = experiment.network_factory(spec)

    iterator = experiment.builder.make_dataset_iterator(replay)
    # make_dataset_iterator is responsible for putting data onto appropriate
    # training devices, so here we apply prefetch, so that data is copied over
    # in the background.
    iterator = utils.prefetch(iterable=iterator, buffer_size=1)
    counter = counting.Counter(counter, 'learner')
    learner = experiment.builder.make_learner(random_key, networks, iterator,
                                              experiment.logger_factory, spec,
                                              replay, counter, cfn)

    if experiment.checkpointing:
      if primary_learner is None:
        checkpointing = experiment.checkpointing
        learner = savers.CheckpointingRunner(
            learner,
            key='learner',
            subdirectory='learner',
            time_delta_minutes=5,
            directory=checkpointing.directory,
            add_uid=checkpointing.add_uid,
            max_to_keep=checkpointing.max_to_keep,
            keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
            checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
        )
      else:
        learner.restore(primary_learner.save())
        # NOTE: This initially synchronizes secondary learner states with the
        # primary one. Further synchronization should be handled by the learner
        # properly doing a pmap/pmean on the loss/gradients, respectively.

    return learner
  
  def build_cfn(
      random_key: networks_lib.PRNGKey,
      replay: reverb.Client,
      counter: Optional[counting.Counter] = None,
  ):
    dummy_seed = 1
    spec = (
        experiment.environment_spec or
        specs.make_environment_spec(experiment.environment_factory(dummy_seed)))
    networks = experiment.network_factory(spec)
    iterator = experiment.builder.make_cfn_dataset_iterator(replay)
    cfn_counter = counting.Counter(counter, 'cfn')
    cfn_obj = experiment.builder.make_cfn_object(
      random_key,
      networks,
      iterator,
      experiment.logger_factory,
      replay,
      cfn_counter)
    if experiment.checkpointing:
      checkpointing = experiment.checkpointing
      cfn_obj = savers.CheckpointingRunner(
        cfn_obj,
        key='cfn_object',
        subdirectory='cfn_object',
        time_delta_minutes=5,
        directory=checkpointing.directory,
        add_uid=checkpointing.add_uid,
        max_to_keep=checkpointing.max_to_keep,
        keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
        checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
      )
    return cfn_obj

  def build_inference_server(
      inference_server_config: inference_server_lib.InferenceServerConfig,
      variable_source: core.VariableSource,
      server_number=0, # to distribute GPUs
  ) -> InferenceServer:
    """Builds an inference server for `ActorCore` policies."""
    dummy_seed = 1
    spec = (
        experiment.environment_spec or
        specs.make_environment_spec(experiment.environment_factory(dummy_seed)))
    networks = experiment.network_factory(spec)
    policy = config.make_policy(
        experiment=experiment,
        networks=networks,
        environment_spec=spec,
        evaluation=False,
    )
    if not isinstance(policy, actor_core.ActorCore):
      raise TypeError(
          f'Using InferenceServer with policy of unsupported type:'
          f'{type(policy)}. InferenceServer only supports `ActorCore` policies.'
      )
    inference_server_devices = jax.local_devices()

    return InferenceServer(
        handler=jax.jit(
            jax.vmap(
                policy.select_action,
                in_axes=(None, 0, 0),
                # Note on in_axes: Params will not be batched. Only the
                # observations and actor state will be stacked along a new
                # leading axis by the inference server.
            ),),
        variable_source=variable_source,
        # devices=jax.local_devices(),
        devices=inference_server_devices,
        config=inference_server_config,
    )

  def build_actor(
      random_key: networks_lib.PRNGKey,
      replay: reverb.Client,
      variable_source: core.VariableSource,
      counter: counting.Counter,
      actor_id: ActorId,
      inference_server: Optional[InferenceServer],
      cfn_variable_source: Optional[core.VariableSource] = None,
      cfn_replay: Optional[reverb.Client] = None
  ) -> environment_loop.EnvironmentLoop:
    """The actor process."""

    environment_key, actor_key = jax.random.split(random_key)
    # Create environment and policy core.

    # Environments normally require uint32 as a seed.
    environment = experiment.environment_factory(
        utils.sample_uint32(environment_key))
    environment_spec = specs.make_environment_spec(environment)

    networks = experiment.network_factory(environment_spec)
    policy_network = config.make_policy(
        experiment=experiment,
        networks=networks,
        environment_spec=environment_spec,
        evaluation=False)
    if inference_server is not None:
      policy_network = actor_core.ActorCore(
          init=policy_network.init,
          select_action=inference_server.handler,
          get_extras=policy_network.get_extras,
      )
      variable_source = variable_utils.ReferenceVariableSource()

    adder = experiment.builder.make_adder(replay, environment_spec,
                                          policy_network)
    
    if experiment.is_cfn:
      cfn_adder = experiment.builder.make_cfn_adder(cfn_replay,
                                                    environment_spec,
                                                    policy_network)
      
      actor = experiment.builder.make_actor(actor_key, policy_network,
                                            environment_spec, variable_source,
                                            adder,
                                            cfn_variable_source=cfn_variable_source,
                                            cfn_adder=cfn_adder)
    else:
      actor = experiment.builder.make_actor(actor_key, policy_network,
                                            environment_spec, variable_source,
                                            adder)

    # Create logger and counter.
    counter = counting.Counter(counter, 'actor')
    logger = experiment.logger_factory('actor', counter.get_steps_key(),
                                       actor_id)
    # Create the loop to connect environment and agent.
    return environment_loop.EnvironmentLoop(
        environment, actor, counter, logger, observers=experiment.observers,
        cfn=cfn_variable_source if experiment.is_cfn else None)

  if not program:
    program = lp.Program(name=name)

  key = jax.random.PRNGKey(experiment.seed)

  checkpoint_time_delta_minutes: Optional[int] = (
      experiment.checkpointing.replay_checkpointing_time_delta_minutes
      if experiment.checkpointing else None)
  replay_node = lp.ReverbNode(
      build_replay, checkpoint_time_delta_minutes=checkpoint_time_delta_minutes)
  replay = replay_node.create_handle()


  if experiment.is_cfn:
    cfn_replay_node = lp.ReverbNode(
      build_cfn_replay, checkpoint_time_delta_minutes=checkpoint_time_delta_minutes)
    cfn_replay = cfn_replay_node.create_handle()

    with program.group('cfn_replay'):
      program.add_node(cfn_replay_node)
  else:
    cfn_replay_node, cfn_replay = None, None

  counter = program.add_node(lp.CourierNode(build_counter), label='counter')

  if experiment.max_num_actor_steps is not None:
    program.add_node(
        lp.CourierNode(lp_utils.StepsLimiter, counter,
                       experiment.max_num_actor_steps),
        label='counter')
  if experiment.is_cfn:
    cfn_key, key = jax.random.split(key)
    cfn_node = lp.CourierNode(build_cfn, cfn_key, cfn_replay, counter)
    cfn = cfn_node.create_handle()

    with program.group('cfn'):
      program.add_node(cfn_node)
  else:
    cfn_node, cfn = None, None

  learner_key, key = jax.random.split(key)
  learner_node = lp.CourierNode(build_learner, learner_key, replay, counter, cfn=cfn)
  learner = learner_node.create_handle()
  variable_sources = [learner]

  if multithreading_colocate_learner_and_reverb:
    program.add_node(
        lp.MultiThreadingColocation([learner_node, replay_node]),
        label='learner')
  else:
    program.add_node(replay_node, label='replay')

    with program.group('learner'):
      program.add_node(learner_node)

      # Maybe create secondary learners, necessary when using multi-host
      # accelerators.
      # Warning! If you set num_learner_nodes > 1, make sure the learner class
      # does the appropriate pmap/pmean operations on the loss/gradients,
      # respectively.
      for _ in range(1, num_learner_nodes):
        learner_key, key = jax.random.split(key)
        variable_sources.append(
            program.add_node(
                lp.CourierNode(
                    build_learner, learner_key, replay,
                    primary_learner=learner, cfn=cfn)))
        # NOTE: Secondary learners are used to load-balance get_variables calls,
        # which is why they get added to the list of available variable sources.
        # NOTE: Only the primary learner checkpoints.
        # NOTE: Do not pass the counter to the secondary learners to avoid
        # double counting of learner steps.

  if inference_server_config is not None:
    num_actors_per_server = math.ceil(num_actors / num_inference_servers)
    with program.group('inference_server'):
      inference_nodes = []
      for _ in range(num_inference_servers):
        inference_nodes.append(
            program.add_node(
                lp.CourierNode(
                    build_inference_server,
                    inference_server_config,
                    learner,
                    courier_kwargs={'thread_pool_size': num_actors_per_server
                                   })))
  else:
    num_inference_servers = 1
    inference_nodes = [None]

  num_actor_nodes, remainder = divmod(num_actors, num_actors_per_node)
  num_actor_nodes += int(remainder > 0)

  # Here we'll want to assign each one to its own CPU somehow.
  # How should we? I guess we could pass in something like actor-cpu-range
  # and divide evenly. Question is, how do we instantiate that?
  # The answer is something along the lines of 

  from contextlib import nullcontext
  actor_context = nullcontext() if split_actor_specs else program.group('actor')
  # with program.group('actor'):
  with actor_context:
    # Create all actor threads.
    *actor_keys, key = jax.random.split(key, num_actors + 1)

    # Create (maybe colocated) actor nodes.
    for node_id, variable_source, inference_node in zip(
        range(num_actor_nodes),
        itertools.cycle(variable_sources),
        itertools.cycle(inference_nodes),
    ):
      actor_node_context = program.group(f'actor_{node_id}') if split_actor_specs else nullcontext()
      with actor_node_context:
        colocation_nodes = []

        first_actor_id = node_id * num_actors_per_node
        for actor_id in range(
            first_actor_id, min(first_actor_id + num_actors_per_node, num_actors)
        ):
          actor = lp.CourierNode(
              build_actor,
              actor_keys[actor_id],
              replay,
              variable_source,
              counter,
              actor_id,
              inference_node,
              cfn,
              cfn_replay
          )
          colocation_nodes.append(actor)

        if len(colocation_nodes) == 1:
          program.add_node(colocation_nodes[0])
        elif multiprocessing_colocate_actors:
          program.add_node(lp.MultiProcessingColocation(colocation_nodes))
        else:
          program.add_node(lp.MultiThreadingColocation(colocation_nodes))

  for evaluator in experiment.get_evaluator_factories():
    evaluator_key, key = jax.random.split(key)
    program.add_node(
        lp.CourierNode(evaluator, evaluator_key, learner, counter,
                       experiment.builder.make_actor, cfn),
        label='evaluator')

  if make_snapshot_models and experiment.checkpointing:
    program.add_node(
        lp.CourierNode(build_model_saver, learner), label='model_saver')

  return program
