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
import rlax

start_time = datetime.now()


# Flags which modify the behavior of the launcher.
flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('env_name', 'MiniGrid-Empty-8x8-v0', 'What environment to run.')
flags.DEFINE_integer('seed', 0, 'Random seed (experiment).')
flags.DEFINE_integer('num_steps', 50_000_000,
                     'Number of environment steps to run for. Number of frames is 4x this')
flags.DEFINE_integer('num_actors', 64, 'Number of actors to use')
flags.DEFINE_integer('spi', 0, 'Samples per insert')
flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')
flags.DEFINE_integer('max_episode_steps', 1_000, 'Episode timeout')

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
      tx_pair=rlax.IDENTITY_PAIR
  )
  return experiments.ExperimentConfig(
      builder=r2d2.R2D2Builder(config),
      network_factory=r2d2.make_atari_networks,
      environment_factory=environment_factory,
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps)


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
       "gsm": PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)})
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
        experiment=config, num_actors=FLAGS.num_actors if lp_utils.is_local_run() else 80,
        create_goal_space_manager=True
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

