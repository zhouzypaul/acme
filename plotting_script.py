import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


experiment_names = [
  # 'four_rooms_exploration_prob0_uniform_node_selection',
  # 'four_rooms_exploration_prob0_1_novelty_node_selection',
  'four_rooms_exploration_after_exploitation2',
  # 'four_rooms_developing_sparse_nodes_task_goal_prob0pt1_long_traj_her_random_exploration',
  'four_rooms_big_refactor',
  'four_rooms_big_refactor_no_exploration_rollouts'
]

exp2label = {
  'four_rooms_exploration_after_exploitation2': 'Shared UVFA for Exploration',
  'four_rooms_big_refactor': 'Random Exploration',
  'four_rooms_big_refactor_no_exploration_rollouts': 'No Separate Exploration Rollout'
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
    filename = f'/home/akhil/acme/{experiment_name}/logs/evaluator/logs.csv'
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


plot_evaluation_return()