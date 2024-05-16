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

    os.makedirs(self._node_expansion_prob_dir, exist_ok=True)
    os.makedirs(self._gc_learning_curves_plotting_dir, exist_ok=True)

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
      edge2successes=state[3]
    )
  
  def __call__(self, episode=0):
    vars = self.get_gsm_variables()
    if vars:
      self._plot_hash2bonus(vars['hash2bonus'], vars['hash2proto'], episode)

    self._log_memory_usage(episode)

  def _plot_hash2bonus(self, hash2bonus, hash2proto, episode):
    """Plot the hash2bonus dictionary."""
    hashes = list(hash2bonus.keys())  # list of tuples where each tuple represents the hot idx in the proto-vector.
    values = list(hash2bonus.values())
    one_hot_vectors = [hash2proto[h] for h in hashes]
    infos = [self._env.binary2info(b) for b in one_hot_vectors]
    
    info_val_with_key = [(infos[i], values[i]) for i in range(len(infos)) if infos[i]['has_key']]
    info_val_without_key = [(infos[i], values[i]) for i in range(len(infos)) if not infos[i]['has_key']]
    info_val_with_open_door = [(infos[i], values[i]) for i in range(len(infos)) if 'door1' in infos[i] and infos[i]['door1'] == 'open']

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

  def _plot_goal_learning_curves():
    pass
  
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
      self(episode=iteration)
      t1 = time.time()
      print(f'Plotted iteration {iteration} in {t1 - t0:.3f} seconds.')
      time.sleep(max(0, self._time_between_plots - (t1 - t0)))
