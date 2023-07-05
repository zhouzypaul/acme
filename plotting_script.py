import argparse
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='32actors-cpu-spi-1-toy')
args = parser.parse_args()


def plot_evaluation_return():
  filename = f'/home/akhil/acme/{args.experiment_name}/logs/evaluator/logs.csv'
  df = pd.read_csv(filename)
  plt.plot(df['evaluator_steps'], df['episode_return'], label='Evaluation return')
  plt.xlabel('Evaluator Steps')
  plt.ylabel('Episode Return')
  plt.savefig(f'{args.experiment_name}_eval_return.png')
  plt.close()
  

def visualize_value_function(path_to_value_matrix: str):
  pass

