import os
import time
import json
import pickle
import random
import psutil
import itertools
import subprocess
import collections
import numpy as np
import matplotlib.pyplot as plt

from acme.utils.paths import get_save_directory


class ModelFreeGSMPlotter:
  def __init__(self, env, time_between_plots=1 * 60):
    self._env = env
    self._time_between_plots = time_between_plots
    
    base_dir = get_save_directory()
    self._checkpoint_path = os.path.join(base_dir, 'plots', 'plotting_vars.pkl')
    self._node_expansion_prob_dir = os.path.join(base_dir, 'plots', 'node_expansion_prob')
    self._gc_learning_curves_plotting_dir = os.path.join(base_dir, 'plots', 'gc_learning_curves')
    self._classifier_positives_plotting_dir = os.path.join(base_dir, 'plots', 'classifier_positives')

    os.makedirs(self._node_expansion_prob_dir, exist_ok=True)
    os.makedirs(self._gc_learning_curves_plotting_dir, exist_ok=True)
    os.makedirs(self._classifier_positives_plotting_dir, exist_ok=True)

  def get_gsm_variables(self):
    try:
      with open(self._checkpoint_path, 'rb') as f:
        state = pickle.load(f)
    except:
      print(f'No checkpoint found at {self._checkpoint_path}')
      return {}
    
    return dict(
      hash2counts=state[0],
      hash2proto=state[1],
      hash2bonus=state[2],
      edge2successes=state[3],
      classifiers=state[4],
      hash2obs=state[5],
      classifier2inferredinfo=state[6],
    )
  
  def __call__(self, episode=0):
    vars = self.get_gsm_variables()
    if vars:
      classifier2positives = self._convert_hash2obs_to_classifier2positives(vars['hash2obs'])
      self._plot_classifier_to_positives(classifier2positives)
      self._print_classifier_inferred_info(vars['classifier2inferredinfo'])
      try:
        self._plot_hash2bonus(vars['hash2bonus'], vars['hash2proto'], episode)
        self._plot_goal_learning_curves(vars['edge2successes'], vars['hash2proto'], episode)
      except Exception as e:
        print(f'Error: {e}')

    self._log_memory_usage(episode)

  def _print_classifier_inferred_info(self, classifier2inferredinfo):
    for classifier, inferred_info in classifier2inferredinfo.items():
      print(f'Classifier {classifier} inferred info: {self._env.binary2info(inferred_info)}')

  def _convert_hash2obs_to_classifier2positives(self, hash2obs):
    """Convert the hash2obs dictionary to a classifier2positives dictionary."""
    classifier2positives = collections.defaultdict(list)
    for hash, observations in hash2obs.items():
      classifier_id = hash[0]
      classifier2positives[classifier_id].extend(list(observations))
    return classifier2positives

  def _plot_classifier_to_positives(self, classifier2positives):
    """Plot the classifier2positives dictionary."""
    for classifier, positives in classifier2positives.items():
      num_images = len(positives)
      
      if num_images == 0:
        continue
      
      cols = int(np.ceil(np.sqrt(num_images)))
      rows = int(np.ceil(num_images / cols))
      
      fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
      if isinstance(axes, plt.Axes):
        axes = [axes]
      else:
        axes = axes.flatten()  # Flatten to iterate easily
      
      for i, oarg in enumerate(positives):
        if i < len(axes):
          axes[i].imshow(oarg.observation[:,:,:3])
          axes[i].axis('off')
      
      # Hide any unused subplots
      for j in range(i+1, len(axes)):
        axes[j].axis('off')
      
      plt.suptitle(f'Classifier {classifier} Positives')
      plt.tight_layout()
      plt.subplots_adjust(top=0.9)  # Adjust title position
      plt.savefig(os.path.join(self._classifier_positives_plotting_dir, f'classifier_{classifier}.png'))
      plt.close()

  def _plot_hash2bonus(self, hash2bonus, hash2proto, episode):
    """Plot the hash2bonus dictionary."""
    hashes = list(hash2bonus.keys())  # list of tuples where each tuple represents the hot idx in the proto-vector.
    values = list(hash2bonus.values())
    one_hot_vectors = [hash2proto[h] for h in hashes]
    infos = [self._env.binary2info(b) for b in one_hot_vectors]
    
    info_val_with_key = [(infos[i], values[i]) for i in range(len(infos)) if infos[i]['has_key']]
    info_val_without_key = [(infos[i], values[i]) for i in range(len(infos)) if not infos[i]['has_key']]
    info_val_with_open_door = [(infos[i], values[i]) for i in range(len(infos)) if 'door4' in infos[i] and infos[i]['door4'] == 'open']

    xs_without_key = [info['player_x'] for info, _ in info_val_without_key]
    ys_without_key = [info['player_y'] for info, _ in info_val_without_key]
    values_without_key = [v for _, v in info_val_without_key]

    plt.figure(figsize=(15, 5))
    
    if not info_val_with_key:
      plt.scatter(xs_without_key, ys_without_key, c=values_without_key)
      plt.colorbar()
    else:
      n_subplots = 2 + (len(info_val_with_open_door) > 0)
      xs_with_key = [info['player_x'] for info, _ in info_val_with_key]
      ys_with_key = [info['player_y'] for info, _ in info_val_with_key]
      values_with_key = [v for _, v in info_val_with_key]
      plt.subplot(1, n_subplots, 1)
      plt.scatter(xs_with_key, ys_with_key, c=values_with_key)
      plt.colorbar()
      plt.title('With Key')
      
      xs_without_key = [info['player_x'] for info, _ in info_val_without_key]
      ys_without_key = [info['player_y'] for info, _ in info_val_without_key]
      values_without_key = [v for _, v in info_val_without_key]
      plt.subplot(1, n_subplots, 2)
      plt.scatter(xs_without_key, ys_without_key, c=values_without_key)
      plt.colorbar()
      plt.title('Without Key')

      if info_val_with_open_door:
        xs_with_open_door = [info['player_x'] for info, _ in info_val_with_open_door]
        ys_with_open_door = [info['player_y'] for info, _ in info_val_with_open_door]
        values_with_open_door = [v for _, v in info_val_with_open_door]
        plt.subplot(1, n_subplots, 3)
        plt.scatter(xs_with_open_door, ys_with_open_door, c=values_with_open_door)
        plt.colorbar()
        plt.title('With Open Door')

    plt.suptitle(f'Hash2Bonus at episode {episode}')
    plt.savefig(os.path.join(self._node_expansion_prob_dir, f'hash2bonus_{episode}.png'))
    plt.close()

  def _plot_goal_learning_curves(self, edge2successes, hash2proto, episode):
    """Grab the destination node in each edge of edge2successes, group them based on whether they are player_pos goals, key goals, or door goals, and plot the learning curves for each group."""
    category_to_mean_success_rate = collections.defaultdict(float)
    category_to_std_error = collections.defaultdict(float)
    category_to_count = collections.defaultdict(int)
    category_to_n_attempts = collections.defaultdict(int)

    node2successes = collections.defaultdict(list)
    for edge, successes in edge2successes.items():
      node2successes[edge[1]].extend(successes)
    
    for dest, successes in node2successes.items():
      info = self._env.binary2info(hash2proto[dest], sparse_info=True)
      def _categorize_goal(info):
        if 'player_y' in info or 'player_x' in info:
          category = 'player_pos'
        elif 'has_key' in info:
          category = 'has_key'
        elif 'key_pos' in info:
          category = 'key_pos'
        elif 'door0' in info:
          category = 'door0'
        elif 'door1' in info:
          category = 'door1'
        elif 'door2' in info:
          category = 'door2'
        elif 'door3' in info:
          category = 'door3'
        elif 'door4' in info:
          category = 'door4'
        elif 'door5' in info:
          category = 'door5'
        elif 'door6' in info:
          category = 'door6'
        elif 'has_ball' in info:
          category = 'has_ball'
        return category
      category = _categorize_goal(info)
      category_to_mean_success_rate[category] += np.mean(successes)
      category_to_std_error[category] += np.std(successes)
      category_to_count[category] += 1
      category_to_n_attempts[category] += len(successes)
    
    categories = []
    mean_success_rates = []
    std_errors = []
    counts = []
    attempts = []
    for category, total_successes in category_to_mean_success_rate.items():
      mean_success_rate = total_successes / category_to_count[category]
      std_error = category_to_std_error[category] / np.sqrt(category_to_count[category])
      categories.append(category)
      mean_success_rates.append(mean_success_rate)
      std_errors.append(std_error)
      counts.append(category_to_count[category])
      attempts.append(category_to_n_attempts[category])
    
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    plt.bar(categories, mean_success_rates)
    plt.xlabel('Category')
    plt.ylabel('Mean Success Rate')
    plt.title('Goal Learning Curves')
    
    plt.subplot(1, 3, 2)
    plt.bar(categories, counts)
    plt.yscale('log')
    plt.xlabel('Category')
    plt.ylabel('Number of goals per category (log scale)')
    plt.title('Goal Counts')

    plt.subplot(1, 3, 3)
    plt.bar(categories, attempts)
    plt.yscale('log')
    plt.xlabel('Category')
    plt.ylabel('Number of attempts (log scale)')
    plt.title('Attempts')
    
    plt.savefig(os.path.join(self._gc_learning_curves_plotting_dir, f'goal_learning_curves_{episode}.png'))
    plt.close()
  
  def _log_memory_usage(self, episode):
    """Log the memory usage at the end of each episode."""
    print(f'Logging memory usage at episode {episode}')
    try:
      # Execute the command, capture the output and error (if any)
      vm = psutil.virtual_memory()
      print(f"Total: {vm.total / (1024**3):.2f} GB, Available: {vm.available / (1024**3):.2f} GB, ",
            f"Used: {vm.used / (1024**3):.2f} GB, Usage: {vm.percent}%")
      
      output = subprocess.check_output(
        'nvidia-smi', stderr=subprocess.STDOUT, shell=True, text=True)
      print(output)
      
    except Exception as e:
      print(f'Error: {e}')

  def run(self):
    for iteration in itertools.count():
      t0 = time.time()
      try:
        self(episode=iteration)
      except Exception as e:
        print(f'Error: {e}')
        continue
      t1 = time.time()
      print(f'Plotted iteration {iteration} in {t1 - t0:.3f} seconds.')
      time.sleep(max(0, self._time_between_plots - (t1 - t0)))
