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
start_time = datetime.now()

# Flags which modify the behavior of the launcher.
flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('env_name', 'Pong', 'What environment to run.')
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
flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')
flags.DEFINE_string('acme_dir', '~/acme', 'Directory to do acme logging')
flags.DEFINE_integer('learner_batch_size', 32, 'Learning batch size. 8 is best for local training, 32 fills up 3090')
flags.DEFINE_boolean('use_rnd', False, 'Whether to use RND')
flags.DEFINE_integer('checkpointing_freq', 5, 'Checkpointing Frequency in Minutes')
flags.DEFINE_integer('min_replay_size', 1000, 'When replay starts')

FLAGS = flags.FLAGS

def make_rnd_builder(r2d2_builder):
    from acme.agents.jax import rnd
    # import ipdb; ipdb.set_trace()
    rnd_config = rnd.RNDConfig(
        is_sequence_based=True, # Probably
    )
    logger_fn = functools.partial(make_experiment_logger, save_dir=FLAGS.acme_dir)
    builder = rnd.RNDBuilder(
        rl_agent=r2d2_builder,
        config=rnd_config,
        logger_fn=logger_fn)
    return builder


def build_experiment_config():
  """Builds R2D2 experiment config which can be executed in different ways."""
  batch_size = FLAGS.learner_batch_size

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

  actor_backend = "cpu" if FLAGS.actor_gpu_ids == ["-1"] else "gpu"
  config = r2d2.R2D2Config(
      burn_in_length=8,
      trace_length=40,
      sequence_period=20,
      min_replay_size=FLAGS.min_replay_size,
      batch_size=batch_size,
      prefetch_size=1,
      # samples_per_insert=1.0,
      samples_per_insert= FLAGS.spi,
      evaluation_epsilon=1e-3,
      learning_rate=1e-4,
      target_update_period=1200,
      variable_update_period=100,
      actor_jit=True,
      actor_backend=actor_backend,
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
  else:
    network_factory = r2d2.make_atari_networks


  checkpointing_config = experiments.CheckpointingConfig(directory=FLAGS.acme_dir)

  return experiments.ExperimentConfig(
      # builder=r2d2.R2D2Builder(config),
      builder=agent_builder,
      # network_factory=r2d2.make_atari_networks,
      network_factory=network_factory,
      environment_factory=environment_factory,
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps,
      checkpointing=checkpointing_config,
      logger_factory=functools.partial(make_experiment_logger, save_dir=FLAGS.acme_dir))


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
    print(local_resources)

    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=num_actors,
        num_actors_per_node=num_actors_per_node,
        multiprocessing_colocate_actors=FLAGS.multiprocessing_colocate_actors,
        split_actor_specs=True,
        )

    lp.launch(program,
              xm_resources=lp_utils.make_xm_docker_resources(program),
              local_resources=local_resources,
              # terminal="current_terminal")
              terminal="tmux_session")
  else:
    experiments.run_experiment(experiment=config)
  

if __name__ == '__main__':
  signal.signal(signal.SIGTERM, sigterm_log_endtime_handler)
  app.run(main)

