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
import argparse


# While this is not a good practice, we need to set the logging directory before importing launchpad.
# And we cannot parse FLAGS because absl insists on parsing them only once via app.run(main).
# The "parse_known_args" method allows us to parse only the arguments we need for determining the log_dir.
parser = argparse.ArgumentParser()
parser.add_argument('--acme_id', type=str, default=None, help='Experiment identifier to use for Acme.')
parser.add_argument('--acme_dir', type=str, default='~/acme', help='Directory to do acme logging')
args = parser.parse_known_args()[0]
lp_log_dir = os.path.join(args.acme_dir, args.acme_id, 'terminal_output')
os.environ["LAUNCHPAD_LOGGING_DIR"] = lp_log_dir

import signal
from absl import flags
from acme.agents.jax import r2d2
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
from acme.utils import utils
import dm_env
import launchpad as lp
from datetime import datetime
import rlax
import functools
from acme.utils.experiment_utils import make_experiment_logger

start_time = datetime.now()


# Flags which modify the behavior of the launcher.
flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('lp_launch_type', 'local_mp', 'Launch type for Launchpad.')
flags.DEFINE_string('terminal', 'tmux_session', 'Terminal for Launchpad.')
flags.DEFINE_string('env_name', 'MiniGrid-Empty-8x8-v0', 'What environment to run.')
flags.DEFINE_integer('seed', 0, 'Random seed (experiment).')
flags.DEFINE_integer('num_steps', 50_000_000,
                     'Number of environment steps to run for. Number of frames is 4x this')
flags.DEFINE_integer('num_actors', 64, 'Number of actors to use')
flags.DEFINE_integer('spi', 0, 'Samples per insert')
flags.DEFINE_string('acme_dir', 'local_testing', 'Directory to do acme logging')
flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')
flags.DEFINE_integer('max_episode_steps', 1_000, 'Episode timeout')

# Novelty search flags.
flags.DEFINE_float('intrinsic_reward_coefficient', 0.001, 'weight given to intrinsic reward for RND')
flags.DEFINE_float('extrinsic_reward_coefficient', 1.0, 'weight given to extrinsic reward for RND (default to 0, so only use intrinsic)')
flags.DEFINE_float('rnd_learning_rate', 1e-4, 'Learning rate for RND')
flags.DEFINE_boolean('use_stale_rewards', True, 'Use stale rewards for RND')
flags.DEFINE_bool('use_rnd', False, help='Whether to use RND or not')
flags.DEFINE_bool('use_cfn', True, help='Whether to use CFN or not')
flags.DEFINE_float('cfn_learning_rate', 1e-4, 'Learning rate for CFN optimizer')
flags.DEFINE_integer('cfn_spi', 0, 'Samples per insert for CFN optimization')
flags.DEFINE_integer('cfn_policy_spi', 8, 'Samples per insert for the exploration policy')
flags.DEFINE_integer('cfn_max_replay_size', 2_000_000, 'Max replay size for CFN optimization')

# GSM flags.
flags.DEFINE_float('amdp_rmax_factor', 2., 'Rmax factor for AMDP')


FLAGS = flags.FLAGS


def build_experiment_config():
  """Builds R2D2 experiment config which can be executed in different ways."""
  batch_size = 32

  # The env_name must be dereferenced outside the environment factory as FLAGS
  # cannot be pickled and pickling is necessary when launching distributed
  # experiments via Launchpad.
  env_name = FLAGS.env_name
  max_episode_steps = FLAGS.max_episode_steps
  
  def environment_factory(seed: int) -> dm_env.Environment:
    return helpers.make_minigrid_environment(
      level_name=env_name,
      max_episode_len=max_episode_steps
    )

  checkpointing_config = experiments.CheckpointingConfig(directory=FLAGS.acme_dir)

  # Configure the agent.
  config = r2d2.R2D2Config(
      burn_in_length=0,  # NOTE(ab): got rid of burn_in
      trace_length=40,
      sequence_period=20,
      min_replay_size=1_000,
      batch_size=batch_size,
      prefetch_size=1,
      samples_per_insert=FLAGS.spi,
      evaluation_epsilon=1e-3,
      learning_rate=1e-4,
      target_update_period=1200,
      variable_update_period=100,
      # The default hyperbolic transform makes the vf preds small (~0.4 max)
      tx_pair=rlax.IDENTITY_PAIR,
      amdp_rmax_factor=FLAGS.amdp_rmax_factor,
  )
  return experiments.ExperimentConfig(
      builder=r2d2.R2D2Builder(config),
      network_factory=r2d2.make_atari_networks,
      environment_factory=environment_factory,
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps,
      logger_factory=functools.partial(make_experiment_logger, save_dir=FLAGS.acme_dir),
      checkpointing=checkpointing_config)


def make_rnd_builder(r2d2_builder):
  from acme.agents.jax import rnd
  # import ipdb; ipdb.set_trace()
  rnd_config = rnd.RNDConfig(
      is_sequence_based=True, # Probably
      intrinsic_reward_coefficient=FLAGS.intrinsic_reward_coefficient,
      extrinsic_reward_coefficient=FLAGS.extrinsic_reward_coefficient,
      predictor_learning_rate=FLAGS.rnd_learning_rate,
      use_stale_rewards=FLAGS.use_stale_rewards
  )
  logger_fn = functools.partial(make_experiment_logger, save_dir=FLAGS.acme_dir)
  builder = rnd.RNDBuilder(
      rl_agent=r2d2_builder,
      config=rnd_config,
      logger_fn=logger_fn)
  return builder


def make_cfn_builder(r2d2_builder):
  from acme.agents.jax.cfn.config import CFNConfig
  from acme.agents.jax.cfn.builder import CFNBuilder
  cfn_config = CFNConfig(
    intrinsic_reward_coefficient=FLAGS.intrinsic_reward_coefficient,
    extrinsic_reward_coefficient=FLAGS.extrinsic_reward_coefficient,
    cfn_learning_rate=FLAGS.cfn_learning_rate,
    use_stale_rewards=FLAGS.use_stale_rewards,
    samples_per_insert=FLAGS.cfn_spi,
    max_replay_size=FLAGS.cfn_max_replay_size,
  )
  logger_fn = functools.partial(make_experiment_logger, save_dir=FLAGS.acme_dir)
  builder = CFNBuilder(
    rl_agent=r2d2_builder,
    config=cfn_config,
    logger_fn=logger_fn)
  return builder


def build_exploration_policy_experiment_config():
  batch_size = 32

  # The env_name must be dereferenced outside the environment factory as FLAGS
  # cannot be pickled and pickling is necessary when launching distributed
  # experiments via Launchpad.
  env_name = FLAGS.env_name
  max_episode_steps = FLAGS.max_episode_steps
  
  def environment_factory(seed: int) -> dm_env.Environment:
    return helpers.make_minigrid_environment(
      level_name=env_name,
      max_episode_len=max_episode_steps,
      goal_conditioned=False,  # This is the reason we have a different env_factory
      seed=seed)
  
  def rnd_network_factory(env_spec):
    from acme.agents.jax import rnd
    r2d2_networks = r2d2.make_vanilla_atari_networks(env_spec)
    return rnd.make_networks(env_spec, direct_rl_networks=r2d2_networks)
  
  def cfn_network_factory(env_spec):
    from acme.agents.jax.cfn import networks as cfn_networks
    r2d2_networks = r2d2.make_vanilla_atari_networks(env_spec)
    return cfn_networks.make_networks(env_spec, direct_rl_networks=r2d2_networks)

  checkpointing_config = experiments.CheckpointingConfig(directory=FLAGS.acme_dir)
  
  # Configure the agent.
  # TODO(ab): pick hyperparameters that make sense for R2D2 + RND.
  config = r2d2.R2D2Config(
      burn_in_length=0,  # NOTE(ab): got rid of burn_in
      trace_length=40,
      sequence_period=20,
      min_replay_size=1_000,
      batch_size=batch_size,
      prefetch_size=1,
      samples_per_insert=FLAGS.cfn_policy_spi,
      evaluation_epsilon=1e-3,
      learning_rate=3e-4,
      target_update_period=1200,
      variable_update_period=100,
      tx_pair=rlax.IDENTITY_PAIR,
      discount=0.99,
  )
  
  if FLAGS.use_rnd:
    agent_builder = make_rnd_builder(r2d2.R2D2Builder(config))
  elif FLAGS.use_cfn:
    agent_builder = make_cfn_builder(r2d2.R2D2Builder(config))
  else:
    raise NotImplementedError()
  
  return experiments.ExperimentConfig(
      builder=agent_builder,
      network_factory=rnd_network_factory if FLAGS.use_rnd else cfn_network_factory,
      environment_factory=environment_factory,
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps,
      logger_factory=functools.partial(make_experiment_logger, save_dir=FLAGS.acme_dir),
      is_cfn=FLAGS.use_cfn,
      checkpointing=checkpointing_config
  )


def _get_local_resources(launch_type):
   assert launch_type in ('local_mp', 'local_mt'), launch_type
   from launchpad.nodes.python.local_multi_processing import PythonProcess
   if launch_type == 'local_mp':
     local_resources = {
       "learner":PythonProcess(env={
         "CUDA_VISIBLE_DEVICES": str(0),
         "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
         "TF_FORCE_GPU_ALLOW_GROWTH": "true",
       }),
       "actor":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
       "evaluator":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(0)}),
       "inference_server":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
       "counter":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
       "replay":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
       "gsm": PythonProcess(env={
         "CUDA_VISIBLE_DEVICES": str(1),  # TODO(ab): Set automatically
         "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
         "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        }),
        "exploration_replay":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
        "exploration_learner": PythonProcess(env={
          "CUDA_VISIBLE_DEVICES": str(1),  # TODO(ab): Set automatically
          "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
          "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        }),
       "cfn": PythonProcess(env={
         "CUDA_VISIBLE_DEVICES": str(1),  # TODO(ab): Set automatically
         "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
         "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        }),
     }
   else:
     local_resources = {}
   return local_resources


def sigterm_log_endtime_handler(_signo, _stack_frame):
  """
  log end time gracefully on SIGTERM
  we use SIGTERM because this is what acme throws when the experiments end by reaching nun_steps
  """
  end_time = datetime.now()
  # log start and end time 
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
  FLAGS.append_flags_into_file('tmp/temp_flags')  # hack: so that subprocesses can load FLAGS
  config = build_experiment_config()
  exploration_config = build_exploration_policy_experiment_config()

  utils.create_log_dir('plots')
  utils.create_log_dir(os.path.join('plots', 'uvfa_plots'))
  utils.create_log_dir(os.path.join('plots', 'target_nodes'))

  if FLAGS.run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=config,
        exploration_experiment=exploration_config,
        num_actors=FLAGS.num_actors if lp_utils.is_local_run() else 80,
        create_goal_space_manager=True
    )
    lp.launch(program, 
              xm_resources=lp_utils.make_xm_docker_resources(program),
              local_resources=_get_local_resources(FLAGS.lp_launch_type),
              terminal=FLAGS.terminal)
  else:
    experiments.run_experiment(experiment=config)
  

if __name__ == '__main__':
  signal.signal(signal.SIGTERM, sigterm_log_endtime_handler)
  app.run(main)

