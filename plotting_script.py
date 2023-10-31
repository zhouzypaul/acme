import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# old log dir: /home/akhil/acme/
LOG_DIR = os.path.expanduser('~/git-repos/acme/examples/baselines/rl_discrete/local_testing')


experiment_names = [
  # 'four_rooms_exploration_prob0_uniform_node_selection',
  # 'four_rooms_exploration_prob0_1_novelty_node_selection',
  # 'four_rooms_exploration_after_exploitation2',
  # # 'four_rooms_developing_sparse_nodes_task_goal_prob0pt1_long_traj_her_random_exploration',
  # 'four_rooms_big_refactor',
  # 'four_rooms_big_refactor_no_exploration_rollouts',
  # 'doorkey16x16_expanded_info2_gsm_varsrc1',
  # 'doorkey16x16_expanded_info2_futures1',
  # 'doorkey16x16_baseline',
  # 'four_rooms_gcnetwork_512_512_always_learn_about_task_noveltyHER_amdp_on_gsm',
  # 'four_rooms_gcnetwork_512_512_always_learn_about_task_noveltyHER_expanded_minigrid2',\
  'sparse_node_sparse_edges_dev_doorkey_nsigmas0_add_all_nodes_in_novel_trajectory1',
  'sparse_edges_dev_doorkey1'
]

exp2label = {
  'four_rooms_exploration_after_exploitation2': 'Shared UVFA for Exploration',
  'four_rooms_big_refactor': 'Random Exploration',
  'four_rooms_big_refactor_no_exploration_rollouts': 'No Separate Exploration Rollout',
  'doorkey16x16_expanded_info2_gsm_varsrc1': 'DoorKey VariableSrc',
  'doorkey16x16_expanded_info2_futures1': 'DoorKey Futures',
  'doorkey16x16_baseline': 'DoorKey Baseline',
  'doorkey16x16_gsm_var_src_unexpanded_info': 'DoorKey GSM VariableSrc',
  'four_rooms_gcnetwork_512_512_always_learn_about_task_noveltyHER_amdp_on_gsm': 'AMDP on GSM',
  'four_rooms_gcnetwork_512_512_always_learn_about_task_noveltyHER_expanded_minigrid2': 'AMDP on Actors',
  'sparse_node_sparse_edges_dev_doorkey_nsigmas0_add_all_nodes_in_novel_trajectory1': 'Sparse Traj, Sparse Edge',
  'sparse_edges_dev_doorkey1': 'Sparse Edge'
}


parser = argparse.ArgumentParser()
parser.add_argument('--save_filename', type=str, required=True)
args = parser.parse_args()


def moving_average(a, n=25):
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n-1:] / n


def plot_evaluation_return():
  for experiment_name in experiment_names:
    filename = f'{LOG_DIR}/{experiment_name}/logs/evaluator/logs.csv'
    df = pd.read_csv(filename)
    # plt.plot(df['evaluator_steps'], df['episode_return'], label=exp2label[experiment_name])
    plt.plot(moving_average(df['episode_return'].to_numpy(), n=100), label=exp2label[experiment_name])
  
  plt.legend()
  plt.xlabel('Evaluator Steps')
  plt.ylabel('Episode Return')
  plt.title('Evaluator Return')
  plt.xlim((0, 2000))
  print(f'Saving {args.save_filename}.png')
  plt.savefig(f'{args.save_filename}.png')
  plt.close()


def plot_actor_sps():
  for experiment_name in experiment_names:
    filename = f'/home/akhil/acme/{experiment_name}/logs/actor/logs.csv'
    df = pd.read_csv(filename)
    # plt.plot(df['evaluator_steps'], df['episode_return'], label=exp2label[experiment_name])
    # import ipdb; ipdb.set_trace()
    col = df['steps_per_second'].tolist()
    def f(column):
      nums = []
      for entry in column:
        try:
          number = float(entry)
          nums.append(number)
        except ValueError:
          pass
      return nums
    
    numbers = np.asarray([x for x in col if f(x)])
    print(len(numbers))
    plt.plot(moving_average(numbers, n=100), label=exp2label[experiment_name])
  
  plt.legend()
  plt.xlabel('Evaluator Steps')
  plt.ylabel('Episode Return')
  plt.title('Evaluator Return')
  plt.xlim((0, 2000))
  print(f'Saving {args.save_filename}.png')
  plt.savefig(f'{args.save_filename}.png')
  plt.close()


plot_evaluation_return()
