import os
import math
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

from scipy.sparse import csr_matrix
from typing import Tuple, Optional, Dict
from acme.utils.paths import get_save_directory

import acme.agents.jax.cfn.plotting as cfn_plotting


class GSMPlotter:
  def __init__(self, time_between_plots=1 * 60, key_bit=2, door_bit=3):
    self._time_between_plots = time_between_plots
    self._reward_means = []
    self._reward_variances = []
    self._key_bit = key_bit
    self._door_bit = door_bit
    self._goal_space_sizes = []

    self._make_debug_version_of_skill_graph = False

    base_dir = get_save_directory()
    self._already_plotted_goals = set()
    self._spatial_plotting_dir = os.path.join(base_dir, 'plots', 'spatial_bonus')
    self._scatter_plotting_dir = os.path.join(base_dir, 'plots', 'true_vs_approx_scatterplots')
    self._gcvf_plotting_dir = os.path.join(base_dir, 'plots', 'gcvf_plots')
    self._discovered_goals_dir = os.path.join(base_dir, 'plots', 'discovered_goals')
    self._node_expansion_prob_dir = os.path.join(base_dir, 'plots', 'node_expansion_prob')
    self._hash_bit_plotting_dir = os.path.join(base_dir, 'plots', 'hash_bit_plots')
    self._hash_bit_vf_diff_dir = os.path.join(base_dir, 'plots', 'hash_bit_vf_diff')
    self._skill_graph_plotting_dir = os.path.join(base_dir, 'plots', 'skill_graph')
    self._off_policy_graph_plotting_dir = os.path.join(base_dir, 'plots', 'off_policy_graph')
    self._gsm_iteration_times_dir = os.path.join(base_dir, 'plots', 'gsm_iteration_times')
    self._checkpoint_path = os.path.join(base_dir, 'plots', 'plotting_vars.pkl')
    self._bellman_errors_plotting_dir = os.path.join(base_dir, 'plots', 'bellman_errors')
    self._amdp_vstar_plotting_dir = os.path.join(base_dir, 'plots', 'amdp_vstar')
    self._hash_bit_plotting_dir = os.path.join(base_dir, 'plots', 'hash_bit_plots')
    self._gc_learning_curves_plotting_dir = os.path.join(base_dir, 'plots', 'gc_learning_curves')
    self._novelty_threshold_plotting_dir = os.path.join(base_dir, 'plots', 'novelty_threshold')
    self._key_competence_plotting_dir = os.path.join(base_dir, 'plots', 'key_competence')
    self._door_competence_plotting_dir = os.path.join(base_dir, 'plots', 'door_competence')
    self._on_policy_count_plotting_dir = os.path.join(base_dir, 'plots', 'on_policy_counts')
    self._transition_matrix_dir = os.path.join(base_dir, 'plots', 'transition_matrix')

    os.makedirs(self._spatial_plotting_dir, exist_ok=True)
    os.makedirs(self._scatter_plotting_dir, exist_ok=True)
    os.makedirs(self._gcvf_plotting_dir, exist_ok=True)
    os.makedirs(self._discovered_goals_dir, exist_ok=True)
    os.makedirs(self._hash_bit_plotting_dir, exist_ok=True)
    os.makedirs(self._hash_bit_vf_diff_dir, exist_ok=True)
    os.makedirs(self._node_expansion_prob_dir, exist_ok=True)
    os.makedirs(self._skill_graph_plotting_dir, exist_ok=True)
    os.makedirs(self._off_policy_graph_plotting_dir, exist_ok=True)
    os.makedirs(self._gsm_iteration_times_dir, exist_ok=True)
    os.makedirs(self._bellman_errors_plotting_dir, exist_ok=True)
    os.makedirs(self._amdp_vstar_plotting_dir, exist_ok=True)
    os.makedirs(self._hash_bit_plotting_dir, exist_ok=True)
    os.makedirs(self._gc_learning_curves_plotting_dir, exist_ok=True)
    os.makedirs(self._novelty_threshold_plotting_dir, exist_ok=True)
    os.makedirs(self._key_competence_plotting_dir, exist_ok=True)
    os.makedirs(self._door_competence_plotting_dir, exist_ok=True)
    os.makedirs(self._on_policy_count_plotting_dir, exist_ok=True)
    os.makedirs(self._transition_matrix_dir, exist_ok=True)

  def get_gsm_variables(self):
    try:
      with open(self._checkpoint_path, 'rb') as f:
        state = pickle.load(f)
    except:
      print(f'No checkpoint found at {self._checkpoint_path}')
      return {}
    
    return dict(
      hash2obs=state[0],
      hash2counts=collections.defaultdict(int, state[1]),
      hash2bonus=state[2],
      on_policy_counts=state[3],
      hash2reward=state[4],
      hash2discount=state[5],
      hash2idx=state[6],
      transition_matrix=state[7],
      idx2hash=state[8],
      edges=state[9],
      off_policy_edges=state[10],
      reward_mean=state[11],
      reward_var=state[12],
      hash2bellman=collections.defaultdict(
        lambda: collections.deque(maxlen=50),
        {k: collections.deque(v, maxlen=50) for k, v in state[13].items()}),
      hash2vstar=state[14],
      gsm_iteration_times=state[15],
      edge2success=state[16],
    )

  def __call__(self, episode=0):
    def extract_hash_to_success(edge2success):
      return {edge[1]: edge2success[edge] for edge in edge2success}

    vars = self.get_gsm_variables()
    if vars:
      self._visualize_transition_matrix(vars['transition_matrix'], episode)
      self.visualize_value_function(
        episode,
        vars['hash2idx'],
        vars['transition_matrix'])
      self._make_spatial_bonus_plot(vars['hash2bonus'], episode)
      self._plot_discovered_goals(
        vars['hash2obs'],
        vars['hash2bonus'],
        vars['reward_mean'],
        vars['reward_var'],
        episode)
      self._reward_means.append(vars['reward_mean'])
      self._reward_variances.append(vars['reward_var'])
      self._goal_space_sizes.append(len(vars['hash2obs']))
      self._plot_hash2bonus(vars['hash2bonus'], episode)

      if self._make_debug_version_of_skill_graph:
        self._old_plot_skill_graph(
          vars['edges'],
          vars['off_policy_edges'],
          episode
        )
      else:
        self._new_plot_skill_graph(
          vars['edges'],
          vars['off_policy_edges'],
          vars['hash2bonus'],
          vars['hash2vstar'],
          vars['hash2obs'],
          episode
        )

      self._plot_bellman_errors(vars['hash2bellman'], episode)
      self._plot_spatial_vstar(vars['hash2vstar'], episode)
      self._plot_gsm_iteration_times(vars['gsm_iteration_times'])
      # self._plot_per_goal_success_curves(
      #   extract_hash_to_success(vars['edge2success']), episode)
      self._plot_reward_mean_and_variance(episode=-1)
      # self._plot_key_bit_success_curves(vars['edge2success'], episode)
      # self._plot_door_bit_success_curves(vars['edge2success'], episode)
      self._plot_on_policy_counts(vars['on_policy_counts'], episode)
      # self._plot_task_goal_success_curve(vars['edge2success'])

      # cfn_plotting.plot_average_bonus_for_each_hash_bit(
      #   vars['hash2bonus'],
      #   save_path=os.path.join(self._hash_bit_plotting_dir, f'mean_bonus_{episode}.png'))
      
      # self._plot_avg_value_for_interesting_hash_bits(
      #   vars['hash2idx'], vars['transition_matrix'], episode)
      
      self._log_memory_usage(episode)
      self._plot_goal_space_size()

  def run(self):
    for iteration in itertools.count():
      t0 = time.time()
      self(episode=iteration)
      t1 = time.time()
      print(f'Plotted iteration {iteration} in {t1 - t0:.3f} seconds.')
      time.sleep(max(0, self._time_between_plots - (t1 - t0)))

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

  def _visualize_transition_matrix(self, transition_matrix, episode):
    """Visualize the transition matrix."""
    if isinstance(transition_matrix, np.ndarray):
      plt.imshow(transition_matrix)
    elif isinstance(transition_matrix, csr_matrix):
      plt.imshow(transition_matrix.toarray())
    else:
      print(f'Unknown type for transition matrix: {type(transition_matrix)}')
      return
    plt.colorbar()
    plt.title('Transition Matrix')
    plt.savefig(os.path.join(self._transition_matrix_dir, f'transition_matrix_{episode}.png'))
    plt.close()

  def visualize_value_function(
    self,
    episode: int,
    hash2idx: Dict[Tuple, int],
    transition_matrix: np.ndarray,
  ):

    def _get_value(src: Tuple, dest: Tuple):
      row = hash2idx[src]
      col = hash2idx[dest]
      return transition_matrix[row, col]

    starts = list(hash2idx.keys())
    
    if len(starts) == 0:
      return
    
    selected_start = random.choice(starts)
    
    hash2val = {h: _get_value(selected_start, h) for h in starts}  
    self._make_spatial_plot(hash2val, f'UVFA starting at {selected_start}')
    plt.savefig(os.path.join(self._gcvf_plotting_dir, f'uvfa_{episode}.png'))
    plt.close()

  def _make_spatial_bonus_plot(self, hash2bonus: Dict, episode: int):
    self._make_spatial_plot(hash2bonus, "")
    plt.savefig(os.path.join(self._spatial_plotting_dir, f'spatial_bonus_{episode}.png'))
    plt.close()

  def _plot_discovered_goals(
      self,
      hash2obs,
      hash2bonus,
      reward_mean,
      reward_var,
      episode):
    for goal_hash in _thread_safe_deepcopy(hash2obs):
      if goal_hash not in self._already_plotted_goals:
        obs = hash2obs[goal_hash]
        # TODO(ab): Maybe compute the bonus if it is not already in hash2bonus.
        score = hash2bonus[goal_hash] if goal_hash in hash2bonus else 0.
        filename = f'goal_{goal_hash}_episode_{episode}.png'
        title = f'Score: {score:.3f} ' + \
                f'Mean: {reward_mean:.3f} ' + \
                f'Var: {reward_var:.3f}'
        plt.imshow(obs.observation)
        plt.title(title)
        plt.savefig(os.path.join(self._discovered_goals_dir, filename))
        plt.close()
        
        self._already_plotted_goals.add(goal_hash)

  def _plot_goal_space_size(self):
    """Plot the size of the goal space as a function of episode."""
    plt.plot(self._goal_space_sizes)
    plt.title('Goal Space Size')
    plt.savefig(os.path.join(self._gsm_iteration_times_dir, 'goal_space_size.png'))
    plt.close()

  @staticmethod
  def subplot_grid(N):
    def is_prime(num):
      if num <= 1:
        return False
      for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
          return False
      return True
    
    # adjust for prime N
    if is_prime(N):
      N += 1

    # Start with a square root approximation to find one dimension
    factor = int(math.sqrt(N))
    while N % factor != 0:
      factor -= 1  # Decrease the factor until it divides N
    # Return dimensions that are factors of N
    return factor, N // factor
  
  @staticmethod
  def _make_spatial_plot(hash2val, title):
    hashes = list(hash2val.keys())
    room2spatial = collections.defaultdict(lambda: collections.defaultdict(list))
    for hash in hashes:
      room = hash[2]
      room2spatial[room]['x'].append(hash[0])
      room2spatial[room]['y'].append(hash[1])
      room2spatial[room]['val'].append(hash2val[hash])
    width, height = plt.rcParams['figure.figsize']
    if len(room2spatial) == 0:
      return
    n_rows, n_cols = GSMPlotter.subplot_grid(len(room2spatial))
    plt.figure(figsize=(width * n_cols, height * n_rows))
    for i, room in enumerate(room2spatial):
      plt.subplot(n_rows, n_cols, i + 1)
      plt.scatter(room2spatial[room]['x'], room2spatial[room]['y'], c=room2spatial[room]['val'])
      plt.title(f'Room {room}')
      plt.colorbar()
    plt.suptitle(title)

  def _plot_hash2bonus(self, hash2bonus, episode):
    self._make_spatial_plot(hash2bonus, f'Bonus at GSM Iteration {episode}')
    plt.savefig(os.path.join(self._node_expansion_prob_dir, f'expansion_probs_{episode}.png'))
    plt.close()

  def _old_plot_skill_graph(self, edges, off_policy_edges, episode, include_off_policy_edges=True):
    """Spatially plot the nodes and edges of the skill-graph."""

    def split_edges(edges):
      # Split the edges based on room number
      rooms2edges = collections.defaultdict(list)
      for edge in edges:
        src, dest = edge
        key = src[2], dest[2]
        rooms2edges[key].append(edge)
      return rooms2edges

    def plot_edges(e, color):
      for edge in e:
        x1 = edge[0][0]
        y1 = edge[0][1]
        x2 = edge[1][0]
        y2 = edge[1][1]
        plt.scatter([x1, x2], [y1, y2], color=color)
        plt.plot([x1, x2], [y1, y2], color=color, alpha=0.3)

    def split_then_plot(edges, color):
      rooms2edges = split_edges(edges)
      if not rooms2edges:
        return

      width, height = plt.rcParams['figure.figsize']
      n_rows, n_cols = GSMPlotter.subplot_grid(len(rooms2edges))
      plt.figure(figsize=(width * n_cols, height * n_rows))
      
      for i, rooms in enumerate(rooms2edges):
        plt.subplot(n_rows, n_cols, i + 1)
        plot_edges(rooms2edges[rooms], color=color)
        plt.title(f'Room {rooms[0]} -> {rooms[1]}')
    
    edges = list(edges)
    split_then_plot(edges, color='black')
    plt.savefig(os.path.join(self._skill_graph_plotting_dir, f'online_skill_graph_{episode}.png'))
    plt.close()
    
    off_policy_edges = list(off_policy_edges)
    split_then_plot(off_policy_edges, color='red')
    plt.savefig(os.path.join(self._off_policy_graph_plotting_dir, f'offline_skill_graph_{episode}.png'))
    plt.close()

  def _new_plot_skill_graph(self, edges, off_policy_edges, hash2bonus, hash2vstar, hash2obs, episode, include_off_policy_edges=True):
    """Spatially plot the nodes and edges of the skill-graph in a grid-based layout."""

    def split_edges(edges):
      # Split the edges based on room number
      rooms2edges = collections.defaultdict(list)
      for edge in edges:
        src, dest = edge
        if src[2] == dest[2]:
          key = src[2], dest[2]
          rooms2edges[key].append(edge)
      return rooms2edges

    def plot_edges(ax, e, color, node2val):
      for edge in e:
        x1, y1 = edge[0][0], edge[0][1]
        x2, y2 = edge[1][0], edge[1][1]
        val = (node2val[edge[0]] + node2val[edge[1]]) / 2
        denom = max(node2val.values()) if max(node2val.values()) != 0 else 1
        val = max(0.05, val / denom)
        ax.plot([x1, x2], [y1, y2], color=color, alpha=val)
        ax.scatter([x1, x2], [y1, y2], color=color, alpha=val)

    def create_grid_structure(fig, rooms2edges):
      n_plots = len(rooms2edges)
      grid = plt.GridSpec(3, 7, figure=fig)
      axes_structure = {
        0: grid[0, 2], 1: grid[0, 3], 2: grid[0, 4],
        3: grid[1, 1], 4: grid[1, 2], 5: grid[1, 3], 6: grid[1, 4], 7: grid[1, 5],
        8: grid[2, 1], 9: grid[2, 2], 10: grid[2, 3], 11: grid[2, 4], 12: grid[2, 5], 13: grid[2, 6]
      }
      return axes_structure

    def _node2val(hash2bonus, hash2vstar):
      # Find the top 10 nodes with the highest bonus
      top_nodes = sorted(hash2bonus, key=hash2bonus.get, reverse=True)[:10]

      # hash2vstar is a nested dictionary
      # first it takes a goal node and then it outputs a value function (map from node to val).
      # for each of the top nodes, average together the value functions.

      node2val = collections.defaultdict(float)
      for node in top_nodes:
        for n, v in hash2vstar.get(node, {}).items():
          node2val[n] += v

      return node2val

    def _get_background_images(hash2obs, hash2bonus):
      # For each room, find the obs with the highest bonus
      room2hashobs = collections.defaultdict(list)
      
      for hash, obs in hash2obs.items():
        room = hash[2]
        room2hashobs[room].append((hash, obs))

      room_to_highest_bonus_obs = {}
      
      for room in room2hashobs:
        hash_obs_pairs = room2hashobs[room]
        bonuses = [hash2bonus.get(h, 0) for h, _ in hash_obs_pairs]
        highest_bonus_idx = np.argmax(bonuses)
        room_to_highest_bonus_obs[room] = hash_obs_pairs[highest_bonus_idx][1]

      return room_to_highest_bonus_obs

    def plot_skill_graphs(edges, node2val, room2background, color, save_path):
      rooms2edges = split_edges(edges)
      if not rooms2edges:
        return

      fig = plt.figure(figsize=(20, 12))
      axes_structure = create_grid_structure(fig, rooms2edges)

      for i, (rooms, edges) in enumerate(rooms2edges.items()):
        if i >= 14:  # We only have space for 14 subplots in our structure
          raise ValueError(f'Too many rooms to plot: {len(rooms2edges)}')
        room = rooms[0]
        ax = fig.add_subplot(axes_structure[room])
        plot_edges(ax, edges, color=color, node2val=node2val)
        ax.imshow(room2background[room].observation, alpha=0.3, extent=[0, 150, 125, 300])
        ax.set_title(f'Room {rooms[0]}')
        ax.set_xticks([])
        ax.set_yticks([])

      plt.tight_layout()
      plt.savefig(save_path)
      plt.close()

    # node2val = self._node2val(hash2bonus, hash2vstar)
    node2val = _node2val(hash2bonus, hash2vstar)
    room2background = _get_background_images(hash2obs, hash2bonus)

    # Plot online skill graph
    online_save_path = os.path.join(self._skill_graph_plotting_dir, f'online_skill_graph_{episode}.png')
    plot_skill_graphs(edges, node2val, room2background, color='black', save_path=online_save_path)

    # Plot offline skill graph if include_off_policy_edges is True
    if include_off_policy_edges:
      offline_save_path = os.path.join(self._off_policy_graph_plotting_dir, f'offline_skill_graph_{episode}.png')
      plot_skill_graphs(off_policy_edges, node2val, room2background, color='red', save_path=offline_save_path)
  

  def _plot_bellman_errors(self, hash2bellman: Dict, episode: int):
    """Randomly sample 4 nodes and plot their bellman errors as a function of iteration."""
    nodes = list(hash2bellman.keys())
    
    if nodes:
      selected_nodes = random.sample(nodes, k=min(len(nodes), 4))
      plt.figure(figsize=(14, 14))
      for i, node in enumerate(selected_nodes):
        plt.subplot(2, 2, i + 1)
        plt.plot(hash2bellman[node], marker='o', linestyle='-')
        plt.title(f'Goal: {node}')
      plt.suptitle(f'Max BE vs # VI Iterations at GSM Iteration {episode}')
      plt.savefig(os.path.join(self._bellman_errors_plotting_dir, f'bellman_errors_{episode}.png'))
      plt.close()

  def _plot_per_goal_success_curves(self, hash2success, episode):
    """Make 3x3 subplots of the success curves for 9 randomly sampled goals."""
    nodes = list(hash2success.keys())
    
    if nodes:
      selected_nodes = random.sample(nodes, k=min(len(nodes), 9))
      plt.figure(figsize=(14, 14))
      for i, node in enumerate(selected_nodes):
        plt.subplot(3, 3, i + 1)
        plt.plot(hash2success[node], marker='o', linestyle='-')
        plt.title(f'Goal: {node}')
      plt.suptitle(f'Goal Success Curves at GSM Iteration {episode}')
      plt.savefig(os.path.join(self._gc_learning_curves_plotting_dir, f'success_curves_{episode}.png'))
      plt.close()

  def _plot_success_curve_for_interesting_hash_bit(self, hash_bit, hash_bit_vals, edge2success, save_path):
    """Plot success curves for some edges that transition from low -> high for the given hash_bit."""
    def get_success_curves():
      interesting_success_curves = {}
      for (src, dest), success_curve in edge2success.items():
        if src[hash_bit] not in hash_bit_vals and \
          dest[hash_bit] in hash_bit_vals and \
            len(success_curve) > 10:
          interesting_success_curves[(src, dest)] = success_curve
      return interesting_success_curves
    
    interesting_success_curves = get_success_curves()
    edges = list(interesting_success_curves.keys())
    selected_edges = random.sample(edges, k=min(len(edges), 6))

    if selected_edges:
      plt.figure(figsize=(14, 14))

    for i, edge in enumerate(selected_edges):
      plt.subplot(3, 2, i + 1)
      plt.plot(interesting_success_curves[edge], marker='o', linestyle='-')
      plt.title(f'{edge}')

    if selected_edges:
      plt.savefig(save_path)
      plt.close()

  def _plot_key_bit_success_curves(self, edge2success, episode):
    """Plot success curves for edges that transition from low -> high for the key bit."""
    self._plot_success_curve_for_interesting_hash_bit(
      hash_bit=self._key_bit,
      hash_bit_vals=(1,),
      edge2success=edge2success,
      save_path=os.path.join(self._key_competence_plotting_dir, f'success_curves_key_{episode}.png')
    )
  
  def _plot_door_bit_success_curves(self, edge2success, episode):
    """Plot success curves for edges that transition from low -> high for the door bit."""
    self._plot_success_curve_for_interesting_hash_bit(
      hash_bit=self._door_bit,
      hash_bit_vals=(0, 1),  # Door unlocked or open
      edge2success=edge2success,
      save_path=os.path.join(self._door_competence_plotting_dir, f'success_curves_door_{episode}.png')
    )

  def _plot_task_goal_success_curve(self, hash2success):
    """Plot the success curve for the task goal."""
    def get_curve():
      for goal in hash2success:
        if -1 in goal and not all(goal == -1):
          return hash2success[goal]
    
    def smoothen_curve(curve, n=10):
      return np.convolve(curve, np.ones(n) / n, mode='valid')
      
    curve = get_curve()
    
    if curve is not None:
      plt.plot(smoothen_curve(curve), marker='o', linestyle='-')
      plt.title('Task Goal Success Curve')
      plt.savefig(os.path.join(self._gc_learning_curves_plotting_dir, f'task_goal_success_curve.png'))
      plt.close()

  def _plot_on_policy_counts(self, edge_counts, episode):
    """Plot the counts for the src node in 1 subplot and the dest node in another."""
    def get_total_count(src=None, dest=None) -> int:
      total_count = 0
      for src_node in edge_counts:
        for dest_node in edge_counts[src_node]:
          if src is not None and src_node == src:
            total_count += edge_counts[src][dest_node]
          elif dest is not None and dest_node == dest:
            total_count += edge_counts[src_node][dest]
      return total_count
    
    src_counts = {src: get_total_count(src=src) for src in edge_counts}
    dst_counts = {dst: get_total_count(dest=dst) for dst in edge_counts}
    
    # Delete the entry from src_counts with the highest value
    # so that the colorbar is not dominated by a single node.
    del src_counts[max(src_counts, key=src_counts.get)]

    self._make_spatial_plot(src_counts, title='log(src counts)')
    plt.savefig(os.path.join(self._on_policy_count_plotting_dir, f'src_edge_counts_{episode}.png'))
    plt.close()

    self._make_spatial_plot(dst_counts, title='log(dest counts)')
    plt.savefig(os.path.join(self._on_policy_count_plotting_dir, f'dest_edge_counts_{episode}.png'))
    plt.close()

  def _plot_reward_mean_and_variance(self, episode):
    """Make a plot showing reward mean and shade the region one std dev above it."""
    plt.figure(figsize=(14, 14))
    means = np.asarray(self._reward_means)
    variances = np.asarray(self._reward_variances)
    stds = np.sqrt(variances + 1e-12)
    plt.plot(self._reward_means, marker='o', linestyle='-', linewidth=4)
    plt.fill_between(
      range(len(self._reward_means)),
      means - stds,
      means + stds,
      alpha=0.5
    )
    plt.savefig(
      os.path.join(self._novelty_threshold_plotting_dir,
                   f'reward_mean_and_variance_{episode}.png'))
    plt.close()

  def _plot_spatial_vstar(self, hash2vstar, episode):
    """Spatially plot the AMDP V* for a randomly sampled goal node."""

    nodes = list(hash2vstar.keys())
    
    if nodes:
      node = random.choice(nodes)
      self._make_spatial_plot(hash2vstar[node], f'Goal: {node}')
      plt.savefig(os.path.join(self._amdp_vstar_plotting_dir, f'vstar_{episode}.png'))
      plt.close()
  
  def _plot_gsm_iteration_times(self, gsm_iteration_times, gsm_goal_space_sizes=None):
    plt.figure(figsize=(14, 14))
    plot_gs_sizes = gsm_goal_space_sizes is not None
    if plot_gs_sizes:
      plt.subplot(121)
    plt.plot(gsm_iteration_times)
    plt.title('GSM Iteration Times')
    if plot_gs_sizes:
      plt.subplot(122)
      plt.plot(gsm_goal_space_sizes)
      plt.title('GSM Goal Space Sizes')
    plt.savefig(os.path.join(self._gsm_iteration_times_dir, 'gsm_iteration_times.png'))
    plt.close()

  def _plot_avg_value_for_interesting_hash_bits(self, hash2idx, transition_matrix, episode):
    node_to_node_values = {}
    nodes = list(hash2idx.keys())
    for src in nodes:
      for dest in nodes:
        node_to_node_values[(src, dest)] = transition_matrix[hash2idx[src], hash2idx[dest]]

    cfn_plotting.plot_average_value_for_interesting_hash_bits(
      node_to_node_values,
      src_node_hash_bit=2,   # Key bit
      dest_node_hash_bit=3,  # Door bit 
      src_node_hash_bit_vals=(1,),  # has_key = True
      dest_node_hash_bit_vals=(0, 1),  # door is either open or unlocked
      save_path=os.path.join(self._hash_bit_vf_diff_dir, f'vf_diff_{episode}.png')
    )
  
  @staticmethod
  def plot_amdp_reward_and_value_functions(
    plotting_dir,
    hash2idx,
    reward_vector,
    discount_vector,
    value_vector,
    goal_pos,
    only_plot_vf=True):
    os.makedirs(plotting_dir, exist_ok=True)
    plt.figure(figsize=(14, 14))
    
    def plot_vector(vec, name: str):
      xs = []; ys = []; values = []
      for key in hash2idx:
        xs.append(key[0])
        ys.append(key[1])
        values.append(vec[hash2idx[key]])

      if values:
        plt.scatter(xs, ys, c=values)
        plt.colorbar()
        plt.title(name)

    if not only_plot_vf:
      plt.subplot(131)
      plot_vector(reward_vector, 'Reward Vector')
      plt.subplot(132)
      plot_vector(discount_vector, 'Discount Vector')
      plt.subplot(133)

    plot_vector(value_vector, 'Value Vector')
    plt.suptitle(f'Goal: {goal_pos}')
    plt.savefig(os.path.join(plotting_dir, f'amdp_goal_{goal_pos}.png'))
    plt.close()


# TODO(ab): move this to the timing class.
def plot_gsm_times(path_to_json, save_path):
  with open(path_to_json, 'r') as f:
    data = json.load(f)
  keys_to_ignore = ['step', '_construct_obs_matrix', '_construct_oarg']

  print(f'Keys: {data.keys()}')
  for key in data:
    if key in keys_to_ignore:
      continue
    plt.plot(data[key], label=key.replace('_', ' '))
  plt.xlabel('Iteration')
  plt.ylabel('Time (s)')
  plt.legend()
  plt.savefig(save_path)
  plt.close()


def _thread_safe_deepcopy(d: Dict):
  # Technically size of d can change during list(d.keys()),
  # but low prob and hasn't happened yet.
  keys = list(d.keys())
  return {k: d[k] for k in keys}


def _dict_to_default_dict(nested_dict, inner_default):
  dd = collections.defaultdict(
    lambda: collections.defaultdict(inner_default))
  for key in nested_dict:
    dd[key] = collections.defaultdict(
      inner_default, nested_dict[key])
  return dd


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_to_json', type=str, required=True)
  parser.add_argument('--save_path', type=str, required=True)
  args = parser.parse_args()
  plot_gsm_times(args.path_to_json, args.save_path)

