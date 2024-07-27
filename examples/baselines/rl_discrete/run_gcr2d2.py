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
import json
import dataclasses
from helpers import save_command_used
from helpers import is_under_git_control, save_git_information
from local_resources import get_local_resources
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
flags.DEFINE_integer('cfn_target_update_period', 1200, 'Target update period for CFN optimization')
flags.DEFINE_float('cfn_clip_random_prior_output', -1., 'Clipping factor for random prior output')

# GSM flags.
flags.DEFINE_float('amdp_rmax_factor', 200., 'Rmax factor for AMDP')
flags.DEFINE_list("actor_gpu_ids", ["-1"], "Which GPUs to use for actors. Actors select GPU in round-robin fashion")
flags.DEFINE_integer("n_sigmas_threshold_for_goal_creation", 1, "Number of sigmas above reward mean for new goal/node creation")
flags.DEFINE_float("prob_augmenting_bonus_constant", 0.1, "Probability of augmenting bonus constant")
flags.DEFINE_bool("use_pessimistic_graph_for_planning", False, "Whether to use pessimistic graph for planning or not")
flags.DEFINE_float("off_policy_edge_threshold", 0.75, "Threshold for off-policy edges")
flags.DEFINE_integer("max_vi_iterations", 50, "Max number of VI iterations for AMDP")
flags.DEFINE_float("novelty_threshold_for_goal_creation", -1., "Threshold for novelty for new goal/node creation")
flags.DEFINE_integer("goal_space_size", -1, "Number of candidate goals for target node sampling. -1 means sum_sampling.")
flags.DEFINE_float("descendant_threshold", 0., "Threshold for enumerating descendant nodes for goal sampling.")

flags.DEFINE_float("task_goal_probability", 0., "Probability of sampling a task goal for behavior generation (0 vector).")
flags.DEFINE_bool("switch_task_expansion_node", False, "Whether to switch the expansion node if it is the task goal.")
flags.DEFINE_string("planner_backup_strategy", "graph_search", "Backup strategy for the planner. One of ['graph_search', 'task'].")
flags.DEFINE_bool("use_planning_in_evaluator", False, "Whether to use planning in the evaluator or not.")
flags.DEFINE_integer('option_timeout', 400, 'Max number of steps for which to pursue a goal.')
flags.DEFINE_bool('use_exploration_vf_for_expansion', False, 'Whether to use exploration value function for expansion or the reward function')
flags.DEFINE_bool('use_decentralized_planner', False, 'In decentralized planning, each actor computes its own plan.')
flags.DEFINE_bool('use_gsm_var_client', False, 'Whether to use the GSM variable client or not')
flags.DEFINE_bool('warmstart_vi', False, 'Whether to warmstart VI with the previous soln of VI.')
flags.DEFINE_float('background_extrinsic_reward_coefficient', 0.0, 'weight given to extrinsic reward for option rf.')

# Environment flags
flags.DEFINE_integer('action_repeat', 4, 'Number of frames to repeat the action for.')
flags.DEFINE_bool('reset_to_laser_room', False, 'Whether to reset Joe to the laser room at the start of the episode (for debugging).')

FLAGS = flags.FLAGS


def make_environment_factory(env_name, max_episode_steps, to_float, action_repeat, reset_to_laser_room):
  
  minigrid_factory = functools.partial(
    helpers.make_minigrid_environment,
    level_name=env_name, max_episode_len=max_episode_steps, to_float=to_float)
  
  montezuma_factory = functools.partial(
    helpers.make_montezuma_environment,
      sticky_actions=False,
      oar_wrapper=False,
      oarg_wrapper=True,
      num_stacked_frames=1,
      flatten_frame_stack=True,
      grayscaling=False,
      to_float=to_float,
      scale_dims=(84, 84),
      max_episode_steps=max_episode_steps,
      action_repeat=action_repeat,
      reset_to_laser_room=reset_to_laser_room
    )
  
  return minigrid_factory if 'MiniGrid' in env_name else montezuma_factory

def build_experiment_config():
  """Builds R2D2 experiment config which can be executed in different ways."""
  batch_size = 32

  # The env_name must be dereferenced outside the environment factory as FLAGS
  # cannot be pickled and pickling is necessary when launching distributed
  # experiments via Launchpad.
  env_name = FLAGS.env_name
  max_episode_steps = FLAGS.max_episode_steps
  action_repeat = FLAGS.action_repeat
  reset_to_laser_room = FLAGS.reset_to_laser_room
  
  def environment_factory(seed: int) -> dm_env.Environment:
    return make_environment_factory(
      env_name,
      max_episode_steps,
      to_float=False,
      action_repeat=action_repeat,
      reset_to_laser_room=reset_to_laser_room
    )(seed=seed, goal_conditioned=True)

  checkpointing_config = experiments.CheckpointingConfig(directory=FLAGS.acme_dir)\
  
  if not FLAGS.use_planning_in_evaluator:
    assert FLAGS.task_goal_probability > 0, "If not planning in eval, then drive behavior with task goal."

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
      n_sigmas_threshold_for_goal_creation=FLAGS.n_sigmas_threshold_for_goal_creation,
      prob_augmenting_bonus_constant=FLAGS.prob_augmenting_bonus_constant,
      use_pessimistic_graph_for_planning=FLAGS.use_pessimistic_graph_for_planning,
      off_policy_edge_threshold=FLAGS.off_policy_edge_threshold,
      max_vi_iterations=FLAGS.max_vi_iterations,
      novelty_threshold_for_goal_creation=FLAGS.novelty_threshold_for_goal_creation,
      goal_space_size=FLAGS.goal_space_size,
      task_goal_probability=FLAGS.task_goal_probability,
      use_planning_in_evaluator=FLAGS.use_planning_in_evaluator,
      should_switch_goal=FLAGS.switch_task_expansion_node,
      subgoal_sampler_default_behavior=FLAGS.planner_backup_strategy,
      option_timeout=FLAGS.option_timeout,
      use_exploration_vf_for_expansion=FLAGS.use_exploration_vf_for_expansion,
      use_decentralized_planner=FLAGS.use_decentralized_planner,
      use_gsm_var_client=FLAGS.use_gsm_var_client,
      warmstart_vi=FLAGS.warmstart_vi,
      descendant_threshold=FLAGS.descendant_threshold,
      background_extrinsic_reward_coefficient=FLAGS.background_extrinsic_reward_coefficient,
  )
  save_config(config, os.path.join(FLAGS.acme_dir, FLAGS.acme_id, 'gc_policy_config.json'))
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
    clip_random_prior_output=FLAGS.cfn_clip_random_prior_output
  )
  save_config(cfn_config, os.path.join(FLAGS.acme_dir, FLAGS.acme_id, 'cfn_config.json'))
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
  target_update_period = FLAGS.cfn_target_update_period
  action_repeat = FLAGS.action_repeat
  reset_to_laser_room = FLAGS.reset_to_laser_room
  
  def environment_factory(seed: int) -> dm_env.Environment:
    return make_environment_factory(
      env_name,
      max_episode_steps,
      to_float=False,
      action_repeat=action_repeat,
      reset_to_laser_room=reset_to_laser_room
    )(seed=seed, goal_conditioned=False)
  
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
      target_update_period=target_update_period,
      variable_update_period=100,
      tx_pair=rlax.IDENTITY_PAIR,
      discount=0.99,
  )
  save_config(
    config, os.path.join(FLAGS.acme_dir, FLAGS.acme_id, 'exploration_policy_config.json'))
  
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


def save_config(config: dataclasses.dataclass, save_path: str):
  """Serialize the config dataclass and dump to a json file."""
  def serialize_dataclass(instance):
    def is_serializable(value):
      try:
        json.dumps(value, indent=4)
        return True
      except (TypeError, OverflowError):
        return False
    return {k: v for k, v in dataclasses.asdict(instance).items() if is_serializable(v)}
  
  config_dict = serialize_dataclass(config)
  json_config_data = json.dumps(config_dict)
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  with open(save_path, 'w') as f:
    f.write(json_config_data)


def main(_):
  config = build_experiment_config()
  exploration_config = build_exploration_policy_experiment_config()

  log_dir = os.path.join(FLAGS.acme_dir, FLAGS.acme_id)
  
  # Save FLAGS
  flags_dict = {name: FLAGS[name].value for name in FLAGS}
  json_flags_data = json.dumps(flags_dict, indent=4)
  with open(os.path.join(log_dir, 'flags.json'), 'w') as f:
    f.write(json_flags_data)

  # log the command used
  save_command_used(log_dir)

  # log git stuff
  if is_under_git_control():
    save_git_information(log_dir)

  if FLAGS.run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=config,
        exploration_experiment=exploration_config,
        num_actors=FLAGS.num_actors if lp_utils.is_local_run() else 80,
        create_goal_space_manager=True
    )
    lp.launch(program, 
              xm_resources=lp_utils.make_xm_docker_resources(program),
              local_resources=get_local_resources(FLAGS.lp_launch_type),
              terminal=FLAGS.terminal)
  else:
    experiments.run_experiment(experiment=config)
  

if __name__ == '__main__':
  signal.signal(signal.SIGTERM, sigterm_log_endtime_handler)
  app.run(main)

