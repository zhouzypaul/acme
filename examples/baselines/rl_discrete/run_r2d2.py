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

"""Example running R2D2 on discrete control tasks."""

import os
import signal
from absl import flags
from acme.agents.jax import r2d2
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import dm_env
import launchpad as lp
from datetime import datetime
from local_resources import get_local_resources
from acme.utils.experiment_utils import make_experiment_logger
import functools
import rlax
start_time = datetime.now()

# Flags which modify the behavior of the launcher.
flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('env_name', 'MontezumaRevenge', 'What environment to run.')
flags.DEFINE_integer('seed', 0, 'Random seed (experiment).')
flags.DEFINE_integer('num_actors', 64, 'Num actors if running distributed')
flags.DEFINE_boolean('one_cpu_per_actor', False, 'If we pin each actor to a different CPU')
flags.DEFINE_integer('num_actors_per_node', 1, 'Actors per node (not sure what this means yet)')
flags.DEFINE_boolean('multiprocessing_colocate_actors', False, 'Not sure, maybe whether to put actors in different processes?')
flags.DEFINE_boolean('learner_on_cpu', False, 'For testing whether learner on GPU makes inference faster')
flags.DEFINE_integer('num_steps', 50_000_000,
                     'Number of environment steps to run for.')
flags.DEFINE_float('spi', 1.0,
                     'Number of samples per insert. 0 means does not constrain, other values do.')
flags.DEFINE_list("actor_gpu_ids", ["-1"], "Which GPUs to use for actors. Actors select GPU in round-robin fashion")
flags.DEFINE_list("learner_gpu_ids", ["0"], "Which GPUs to use for learner. Gets all")
flags.DEFINE_string("cfn_gpu_id", "1", "which GPU to use for CFN optimization.")
flags.DEFINE_float('cfn_spi', 8.0,
                     'Number of samples per insert. 0 means does not constrain, other values do.')
flags.DEFINE_integer('cfn_min_replay_size', 2048, 'When CFN training starts')
flags.DEFINE_integer('cfn_max_replay_size', 1_000_000, 'Size of CFN replay buffer')
flags.DEFINE_float('cfn_learning_rate', 1e-4, 'Learning rate for CFN') 
flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')
flags.DEFINE_string('acme_dir', '~/acme', 'Directory to do acme logging')
flags.DEFINE_integer('learner_batch_size', 32, 'Learning batch size. 8 is best for local training, 32 fills up 3090. Paper is 64')
flags.DEFINE_boolean('use_rnd', False, 'Whether to use RND')
flags.DEFINE_boolean('use_cfn', False, 'Whether to use CFN')
flags.DEFINE_integer('checkpointing_freq', 5, 'Checkpointing Frequency in Minutes')
flags.DEFINE_integer('min_replay_size', 10_000, 'When training from replay starts')

flags.DEFINE_float('intrinsic_reward_coefficient', 0.001, 'weight given to intrinsic reward for RND')
flags.DEFINE_float('extrinsic_reward_coefficient', 1.0, 'weight given to extrinsic reward for RND (default to 0, so only use intrinsic)')  # NOTE: NGU paper uses 0.3
flags.DEFINE_float('rnd_learning_rate', 1e-4, 'Learning rate for RND')  # NOTE: NGU paper is 5e-4
flags.DEFINE_string('terminal', 'tmux_session', 'Either terminal or current_terminal')
flags.DEFINE_float('r2d2_learning_rate', 1e-4, 'Learning rate for R2D2')  # NOTE: NGU paper is 2e-4
# These are different from paper to here, so will add as hypers
flags.DEFINE_float('target_update_period', 1200, 'How often to update target network') # NOTE: paper is 2500
flags.DEFINE_float('variable_update_period', 100, 'How often to update actor variables') # NOTE: paper is 400
flags.DEFINE_boolean('use_stale_rewards', False, 'Use stale rewards for RND')
flags.DEFINE_integer('burn_in_length', 8, 'How long to burn in in replay to get good core state (paper is 20/40)')
flags.DEFINE_integer('trace_length', 40, 'Length of sequence to fetch/train on (paper is 80)')
flags.DEFINE_integer('sequence_period', 20, 'How often to start a new sequence. Sequences are repeated in dataset. Should be half of trace_length (paper is 40)')

# MiniGrid Config
flags.DEFINE_integer('max_episode_steps', 1_000, 'Episode timeout')

# Plotting config
# Setting both to -1 disables plotting
flags.DEFINE_integer('cfn_bonus_plotting_freq', 1_000, 'How often to make CFN plots. -1 disables')
flags.DEFINE_integer('cfn_value_plotting_freq', 1_000, 'How often to make CFN plots. -1 disables')

flags.DEFINE_bool('condition_actor_on_intrinsic_reward', False, 'Whether to condition actor LSTM on intrinsic reward')
flags.DEFINE_bool('use_identity_tx', False, 'Whether to use undo R2D2s hyperbolic squash.')

FLAGS = flags.FLAGS

def make_rnd_builder(r2d2_builder):
    from acme.agents.jax import rnd
    # import ipdb; ipdb.set_trace()
    rnd_config = rnd.RNDConfig(
        is_sequence_based=True, # Probably
        intrinsic_reward_coefficient=FLAGS.intrinsic_reward_coefficient,
        extrinsic_reward_coefficient=FLAGS.extrinsic_reward_coefficient,
        predictor_learning_rate=FLAGS.rnd_learning_rate,
        use_stale_rewards=FLAGS.use_stale_rewards,
        condition_actor_on_intrinsic_reward=FLAGS.condition_actor_on_intrinsic_reward
    )
    logger_fn = functools.partial(make_experiment_logger, save_dir=FLAGS.acme_dir)
    builder = rnd.RNDBuilder(
        rl_agent=r2d2_builder,
        config=rnd_config,
        logger_fn=logger_fn)
    return builder


def make_cfn_builder(r2d2_builder):
  from acme.agents.jax.cfn import config as cfn_config
  from acme.agents.jax.cfn import builder as cfn_builder
  cfn_config = cfn_config.CFNConfig(
    use_stale_rewards=FLAGS.use_stale_rewards,
    is_sequence_based=True,
    samples_per_insert=FLAGS.cfn_spi,
    min_replay_size=FLAGS.cfn_min_replay_size,
    max_replay_size=FLAGS.cfn_max_replay_size,
    cfn_learning_rate=FLAGS.cfn_learning_rate,
    intrinsic_reward_coefficient=FLAGS.intrinsic_reward_coefficient,
    extrinsic_reward_coefficient=FLAGS.extrinsic_reward_coefficient,
    bonus_plotting_freq=FLAGS.cfn_bonus_plotting_freq,
    value_plotting_freq=FLAGS.cfn_value_plotting_freq,
    condition_actor_on_intrinsic_reward=FLAGS.condition_actor_on_intrinsic_reward
  )
  logger_fn = functools.partial(make_experiment_logger, save_dir=FLAGS.acme_dir)
  builder = cfn_builder.CFNBuilder(
    rl_agent=r2d2_builder,
    config=cfn_config,
    logger_fn=logger_fn
  )
  return builder


def build_experiment_config():
  """Builds R2D2 experiment config which can be executed in different ways."""
  batch_size = FLAGS.learner_batch_size

  # The env_name must be dereferenced outside the environment factory as FLAGS
  # cannot be pickled and pickling is necessary when launching distributed
  # experiments via Launchpad.
  env_name = FLAGS.env_name
  max_episode_steps = FLAGS.max_episode_steps

  if FLAGS.use_cfn and not FLAGS.use_stale_rewards:
    raise Exception("Fresh CFN not supported at this time. Please try again later.")

  # Create an environment factory.
  def environment_factory(seed: int) -> dm_env.Environment:
    del seed
    return helpers.make_atari_environment(
        level=env_name,
        sticky_actions=True,
        zero_discount_on_life_loss=False,
        oar_wrapper=True,
        num_stacked_frames=1,
        flatten_frame_stack=True,
        grayscaling=False)
  
  def minigrid_environment_factory(seed: int) -> dm_env.Environment:
    del seed  # NOTE: not supporting different seeds on the env for now.
    return helpers.make_minigrid_environment(
      level_name=env_name,
      max_episode_len=max_episode_steps,
      oar_wrapper=True
    )

  def visgrid_environment_factory(seed: int) -> dm_env.Environment:
    return helpers.make_visgrid_environment(oar_wrapper=True)

  actor_backend = "cpu" if FLAGS.actor_gpu_ids == ["-1"] else "gpu"
  tx = rlax.IDENTITY_PAIR if FLAGS.use_identity_tx else rlax.SIGNED_HYPERBOLIC_PAIR
  config = r2d2.R2D2Config(
      burn_in_length=FLAGS.burn_in_length,
      trace_length=FLAGS.trace_length,
      sequence_period=FLAGS.sequence_period,
      min_replay_size=FLAGS.min_replay_size,
      batch_size=batch_size,
      prefetch_size=1,
      # samples_per_insert=1.0,
      samples_per_insert= FLAGS.spi,
      evaluation_epsilon=1e-3,
      learning_rate=FLAGS.r2d2_learning_rate,
      target_update_period=FLAGS.target_update_period,
      variable_update_period=FLAGS.variable_update_period,
      actor_jit=True,
      actor_backend=actor_backend,
      tx_pair=tx,
  )

  # # Configure the agent.

  checkpointing_config = experiments.CheckpointingConfig(directory=FLAGS.acme_dir,
                                                         time_delta_minutes=FLAGS.checkpointing_freq)

  print('hardcoded save dir to see if this is where we need it')
  # def temp_logger_factory():
  #   return functools.partial(make_experiment_logger, save_dir="~/acme_experiment_utils_again")

  # logger_factory = functools.partial(create_experiment_logger_factory, save_dir="~/acme_experiment_utils")

  agent_builder = r2d2.R2D2Builder(config)
  if FLAGS.use_rnd:
    agent_builder = make_rnd_builder(agent_builder)

    def network_factory(env_spec):
      from acme.agents.jax import rnd
      r2d2_networks = r2d2.make_atari_networks(env_spec)
      rnd_networks = rnd.make_networks(env_spec, direct_rl_networks=r2d2_networks)
      return rnd_networks
  elif FLAGS.use_cfn:
    agent_builder = make_cfn_builder(agent_builder)

    def network_factory(env_spec):
      from acme.agents.jax.cfn.networks import make_networks
      r2d2_networks = r2d2.make_atari_networks(env_spec)
      cfn_networks = make_networks(env_spec, r2d2_networks)
      return cfn_networks
  else:
    network_factory = r2d2.make_atari_networks


  checkpointing_config = experiments.CheckpointingConfig(directory=FLAGS.acme_dir)

  if 'minigrid' in env_name.lower():
    env_factory = minigrid_environment_factory
  elif 'visgrid' in env_name.lower():
    env_factory = visgrid_environment_factory
  else:
    env_factory = environment_factory

  make_bonus_plots = FLAGS.cfn_bonus_plotting_freq > 0 and FLAGS.cfn_value_plotting_freq > 0

  return experiments.ExperimentConfig(
      # builder=r2d2.R2D2Builder(config),
      builder=agent_builder,
      # network_factory=r2d2.make_atari_networks,
      network_factory=network_factory,
      environment_factory=env_factory,
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps,
      checkpointing=checkpointing_config,
      logger_factory=functools.partial(make_experiment_logger, save_dir=FLAGS.acme_dir),
      is_cfn=FLAGS.use_cfn,
      make_bonus_plots=make_bonus_plots,)


def sigterm_log_endtime_handler(_signo, _stack_frame):
  """
  log end time gracefully on SIGTERM
  we use SIGTERM because this is what acme throws when the experiments end by reaching nun_steps
  """
  end_time = datetime.now()
  # log start and end time 
  # log_dir = os.path.expanduser(os.path.join('~/acme', FLAGS.acme_id))
  log_dir = os.path.expanduser(os.path.join(FLAGS.acme_dir, FLAGS.acme_id))
  # don't print because it will be lost, especially because acme stops experiment by throwing an Error when reaching num_steps
  from helpers import save_start_and_end_time
  save_start_and_end_time(log_dir, start_time, end_time)

  # log the command used
  from helpers import save_command_used
  save_command_used(log_dir)

  # log git stuff
  from helpers import is_under_git_control, save_git_information
  if is_under_git_control():
      save_git_information(log_dir)


def main(_):
  config = build_experiment_config()
  if FLAGS.run_distributed:
    num_actors = FLAGS.num_actors
    num_actors_per_node = FLAGS.num_actors_per_node
    launch_type = FLAGS.lp_launch_type

    local_resources = get_local_resources(launch_type)

    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=num_actors,
        num_actors_per_node=num_actors_per_node,
        multiprocessing_colocate_actors=FLAGS.multiprocessing_colocate_actors,
        split_actor_specs=True,
        )

    lp.launch(program,
              xm_resources=lp_utils.make_xm_docker_resources(program),
              local_resources=local_resources,
              terminal=FLAGS.terminal)
  else:
    experiments.run_experiment(experiment=config)
  

if __name__ == '__main__':
  signal.signal(signal.SIGTERM, sigterm_log_endtime_handler)
  app.run(main)

