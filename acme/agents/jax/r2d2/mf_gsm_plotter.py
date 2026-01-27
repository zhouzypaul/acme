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
    self._selection_difficulty_dir = os.path.join(base_dir, 'plots', 'selection_difficulty') # Reuse this dir

    os.makedirs(self._node_expansion_prob_dir, exist_ok=True)
    os.makedirs(self._gc_learning_curves_plotting_dir, exist_ok=True)
    os.makedirs(self._classifier_positives_plotting_dir, exist_ok=True)
    os.makedirs(self._selection_difficulty_dir, exist_ok=True)

    # Robust local history tracking
    self._max_bonus_history = []  # List of (timestamp, max_bonus)
    self._mean_bonus_history = [] # List of (timestamp, mean_bonus)

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
      # selection_history ignored as it's empty on learner
    )
  
  def __call__(self, episode=0):
    vars = self.get_gsm_variables()
    if vars:
      classifier2positives = self._convert_hash2obs_to_classifier2positives(vars['hash2obs'])
      
      try:
        self._plot_classifier_to_positives(classifier2positives)
      except Exception as e:
        print(f'Error plotting classifier positives: {e}')
        
      self._print_classifier_inferred_info(vars['classifier2inferredinfo'])
      
      try:
        self._plot_hash2bonus(vars['hash2bonus'], vars['hash2proto'], episode)
      except Exception as e:
        print(f'Error plotting hash2bonus: {e}')

      try:
        self._plot_goal_learning_curves(vars['edge2successes'], vars['hash2proto'], episode)
      except Exception as e:
        print(f'Error plotting learning curves: {e}')
      
      # Robust Bonus Plotting using Learner state
      try:
        if vars['hash2bonus']:
            values = [v for v in vars['hash2bonus'].values() if v is not None]
            if values:
                # Handle potential numpy arrays
                try:
                    clean_values = [float(v) for v in values]
                except:
                     clean_values = [float(v.item()) if hasattr(v, 'item') else 0.0 for v in values]
                
                max_bonus = max(clean_values)
                # mean_bonus = sum(clean_values) / len(clean_values)
                mean_bonus = np.mean(clean_values)
                timestamp = time.time()
                
                self._max_bonus_history.append((timestamp, max_bonus))
                self._mean_bonus_history.append((timestamp, mean_bonus))
                
                self._plot_bonus_stats(episode)
      except Exception as e:
        print(f'Error plotting bonus stats: {e}')

    self._log_memory_usage(episode)

  def _print_classifier_inferred_info(self, classifier2inferredinfo):
    for classifier, inferred_info in classifier2inferredinfo.items():
      # Use internal binary2info logic safely
      try:
          info = self._binary2info(inferred_info)
          print(f'Classifier {classifier} inferred info: {info}')
      except:
          pass

  def _binary2info(self, binary_vector, sparse_info: bool = False):
        """
        Convert a binary vector back into an info dictionary for Montezuma's Revenge.
        Inlined from MontezumaInfoWrapper to ensure robustness against env wrappers.
        """
        # Define the inventory mapping
        inventory_items = ['torch', 'sword', 'sword', 'key', 'key', 'key', 'key', 'hammer']

        # Initialize the info dictionary
        info = {}

        # Decode player_x (first 206 indices)
        player_x_index = np.where(binary_vector[:206] == 1)[0]
        if len(player_x_index) > 0 or not sparse_info:
            info["player_x"] = player_x_index[0] if len(player_x_index) > 0 else -1

        # Decode player_y (indices 206 to 341, normalized to 120-255)
        player_y_index = np.where(binary_vector[206:342] == 1)[0]
        if len(player_y_index) > 0 or not sparse_info:
            info["player_y"] = player_y_index[0] + 120 if len(player_y_index) > 0 else -1

        # Decode room number (indices 342 to 373)
        room_number_index = np.where(binary_vector[342:374] == 1)[0]
        if len(room_number_index) > 0 or not sparse_info:
            info["room_number"] = room_number_index[0] if len(room_number_index) > 0 else -1

        # Decode player state flags
        # Accessing indices safely in case vector is short (though it shouldn't be)
        if len(binary_vector) > 379:
            if not sparse_info or binary_vector[378]:
                info["left_door_open"] = bool(binary_vector[378])
            if not sparse_info or binary_vector[379]:
                info["right_door_open"] = bool(binary_vector[379])

            # Decode inventory (binary string starting from index 380)
            if len(binary_vector) >= 380 + len(inventory_items):
                inventory_binary = binary_vector[380:380 + len(inventory_items)]
                decoded_inventory = [
                    item for bit, item in zip(inventory_binary, inventory_items) if bit
                ]
                if len(decoded_inventory) > 0 or not sparse_info:
                    info["inventory"] = decoded_inventory
        
        return info

  def _convert_hash2obs_to_classifier2positives(self, hash2obs):
    """Convert the hash2obs dictionary to a classifier2positives dictionary."""
    classifier2positives = collections.defaultdict(list)
    for hash_key, observations in hash2obs.items():
      # Hash can be an int (from initialization) or a tuple (from updates)
      if isinstance(hash_key, tuple):
        classifier_id = hash_key[0]
      else:
        classifier_id = hash_key
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
          obs = oarg.observation
          if obs.ndim == 2:
            axes[i].imshow(obs, cmap='gray')
          elif obs.ndim == 3 and obs.shape[2] == 1:
             axes[i].imshow(obs[:,:,0], cmap='gray')
          elif obs.ndim == 3 and obs.shape[2] == 2:
             # Handle 2-channel images (e.g. frame stack of 2) by showing 1st channel
             axes[i].imshow(obs[:,:,0], cmap='gray')
          else:
             axes[i].imshow(obs[:,:,:3])
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
    # Use robust local binary2info
    infos = [self._binary2info(b) for b in one_hot_vectors]
    
    info_val_with_key = [(infos[i], values[i]) for i in range(len(infos)) if infos[i].get('inventory') and 'key' in infos[i]['inventory']]
    info_val_without_key = [(infos[i], values[i]) for i in range(len(infos)) if not (infos[i].get('inventory') and 'key' in infos[i]['inventory'])]
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

  def _plot_bonus_stats(self, episode):
    """Plot the Max and Mean bonus of available goals over time."""
    if not self._max_bonus_history:
        return

    # Unzip history
    # max_history is list of (timestamp, val)
    timestamps, max_vals = zip(*self._max_bonus_history)
    _, mean_vals = zip(*self._mean_bonus_history)
    
    # Calculate relative time (minutes since start of history)
    t0 = timestamps[0]
    times_mins = [(t - t0) / 60.0 for t in timestamps]
    
    plt.figure(figsize=(10, 6))
    
    # Plot Max Bonus (Proxy for what the bandit selects)
    plt.plot(times_mins, max_vals, color='blue', linewidth=2, label='Max Bonus (Best Goal)')
    
    # Plot Mean Bonus (General difficulty of frontier)
    plt.plot(times_mins, mean_vals, color='green', linestyle='--', linewidth=1.5, label='Mean Bonus (All Goals)')
    
    plt.xlabel('Time (minutes)')
    plt.ylabel('Goal Bonus (Implied CFN/Novelty)')
    plt.title(f'Goal Bonus Trends Over Time (Episode {episode})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save to the selection_difficulty dir so user finds it where expected
    plt.savefig(os.path.join(self._selection_difficulty_dir, f'selection_difficulty_{episode}.png'))
    plt.close()
