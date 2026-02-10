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
start_time = datetime.now()


# Flags which modify the behavior of the launcher.
flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('env_name', 'Pong', 'What environment to run.')
flags.DEFINE_integer('seed', 0, 'Random seed (experiment).')
flags.DEFINE_integer('num_steps', 50_000_000,
                     'Number of environment steps to run for. Number of frames is 4x this')
flags.DEFINE_integer('num_actors', 64, 'Number of actors to use')
flags.DEFINE_integer('spi', 0, 'Samples per insert')
flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')

FLAGS = flags.FLAGS

def make_logger_without_timestamp(label: str,
                                   steps_key: str = None,
                                   task_instance: int = 0,
                                   save_dir: str = None):
  """Custom logger that saves to save_dir without adding timestamp subdirectories."""
  from acme.utils import loggers
  del task_instance
  if steps_key is None:
    steps_key = f'{label}_steps'
  
  # Create terminal logger
  terminal_logger = loggers.terminal.TerminalLogger(label=label)
  
  # Create CSV logger with add_uid=False to prevent timestamp directories
  csv_logger = loggers.csv.CSVLogger(
      directory_or_file=save_dir,
      label=label,
      add_uid=False  # This prevents timestamp directory creation
  )
  
  # Combine loggers
  logger = loggers.aggregators.Dispatcher([terminal_logger, csv_logger])
  logger = loggers.filters.NoneFilter(logger)
  logger = loggers.filters.TimeFilter(logger, time_delta=1.0)
  
  return logger

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
    logger_fn = functools.partial(make_logger_without_timestamp, save_dir=os.path.join(FLAGS.acme_dir, FLAGS.acme_id))
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
    use_reward_normalization=FLAGS.cfn_use_reward_normalization,
    bonus_plotting_freq=FLAGS.cfn_bonus_plotting_freq,
    value_plotting_freq=FLAGS.cfn_value_plotting_freq,
    condition_actor_on_intrinsic_reward=FLAGS.condition_actor_on_intrinsic_reward,
    cfn_var_to_std_epsilon=FLAGS.cfn_var_to_std_epsilon,
    cfn_use_forgetting=FLAGS.cfn_use_forgetting,
  )
  save_config(cfn_config,
              os.path.join(FLAGS.acme_dir, FLAGS.acme_id, 'cfn_config.json'))
  
  logger_fn = functools.partial(make_logger_without_timestamp, save_dir=os.path.join(FLAGS.acme_dir, FLAGS.acme_id))
  builder = cfn_builder.CFNBuilder(
    rl_agent=r2d2_builder,
    config=cfn_config,
    logger_fn=logger_fn
  )
  return builder


def build_experiment_config():
  """Builds R2D2 experiment config which can be executed in different ways."""
  batch_size = 32

  # The env_name must be dereferenced outside the environment factory as FLAGS
  # cannot be pickled and pickling is necessary when launching distributed
  # experiments via Launchpad.
  env_name = FLAGS.env_name

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

  # Commented out - not needed for MontezumaRevenge experiments
  # def visgrid_environment_factory(seed: int) -> dm_env.Environment:
  #   return helpers.make_visgrid_environment(oar_wrapper=True)

  # Commented out - not needed for MontezumaRevenge experiments
  # def taxi_environment_factory(seed: int) -> dm_env.Environment:
  #   return helpers.make_taxi_environment(
  #     max_steps=max_episode_steps,
  #     seed=seed,
  #     goal_conditioned=False,
  #     oar_wrapper=True,
  #     grid_size=taxi_grid_size,
  #   )

  # Commented out - not needed for MontezumaRevenge experiments
  # def sokoban_environment_factory(seed: int) -> dm_env.Environment:
  #   return helpers.make_sokoban_environment(
  #     level_name=env_name,
  #     seed=seed,
  #     goal_conditioned=False,
  #     to_float=False,
  #   )

  actor_backend = "cpu" if FLAGS.actor_gpu_ids == ["-1"] else "gpu"
  tx = rlax.IDENTITY_PAIR if FLAGS.use_identity_tx else rlax.SIGNED_HYPERBOLIC_PAIR
  config = r2d2.R2D2Config(
      burn_in_length=8,
      trace_length=40,
      sequence_period=20,
      min_replay_size=10_000,
      batch_size=batch_size,
      prefetch_size=1,
      samples_per_insert=FLAGS.spi,
      evaluation_epsilon=1e-3,
      learning_rate=1e-4,
      target_update_period=1200,
      variable_update_period=100,
  )

  return experiments.ExperimentConfig(
      builder=r2d2.R2D2Builder(config),
      network_factory=r2d2.make_atari_networks,
      environment_factory=environment_factory,
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps,
      checkpointing=checkpointing_config,
      logger_factory=functools.partial(make_logger_without_timestamp, save_dir=os.path.join(FLAGS.acme_dir, FLAGS.acme_id)),
      is_cfn=FLAGS.use_cfn,
      make_bonus_plots=make_bonus_plots,)


def sigterm_log_endtime_handler(_signo, _stack_frame):
  """
  log end time gracefully on SIGTERM
  we use SIGTERM because this is what acme throws when the experiments end by reaching nun_steps
  """
  end_time = datetime.now()
  # log start and end time 
  log_dir = os.path.expanduser(os.path.join('~/acme', FLAGS.acme_id))
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
  FLAGS.append_flags_into_file('/tmp/temp_flags')  # hack: so that subprocesses can load FLAGS
  config = build_experiment_config()
  if FLAGS.run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=FLAGS.num_actors if lp_utils.is_local_run() else 80
    )
    lp.launch(program, 
              xm_resources=lp_utils.make_xm_docker_resources(program),
              local_resources=_get_local_resources(FLAGS.lp_launch_type),
              terminal='tmux_session')
  else:
    experiments.run_experiment(experiment=config)
  

if __name__ == '__main__':
  signal.signal(signal.SIGTERM, sigterm_log_endtime_handler)
  app.run(main)

