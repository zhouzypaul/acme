import os
import time
import math
import json
import pickle
import random
import psutil
import itertools
import subprocess
import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
      edge2return=state[18]
    )

  def __call__(self, episode=0):
    def extract_hash_to_success(edge2success):
      return {edge[1]: edge2success[edge] for edge in edge2success}

    vars = self.get_gsm_variables()
    if vars:
      self.visualize_value_function(
        episode,
        vars['hash2idx'],
        vars['transition_matrix'])
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
      self._plot_skill_graph(vars['edges'], vars['off_policy_edges'], vars['edge2return'], episode)
      self._plot_bellman_errors(vars['hash2bellman'], episode)
      self._plot_spatial_vstar(vars['hash2vstar'], episode)
      self._plot_gsm_iteration_times(vars['gsm_iteration_times'])
      self._plot_per_goal_success_curves(
        extract_hash_to_success(vars['edge2success']), episode)
      self._plot_reward_mean_and_variance(episode=-1)
      self._plot_on_policy_counts(vars['on_policy_counts'], episode)
      self._plot_task_goal_success_curve(vars['edge2success'])

      cfn_plotting.plot_average_bonus_for_each_hash_bit(
        vars['hash2bonus'],
        save_path=os.path.join(self._hash_bit_plotting_dir, f'mean_bonus_{episode}.png'))
      
      self._plot_avg_value_for_interesting_hash_bits(
        vars['hash2idx'], vars['transition_matrix'], episode)
      
      self._log_memory_usage(episode)
      self._plot_goal_space_size()

  def run(self):
    for iteration in itertools.count():
      t0 = time.time()
      # try:
      self(episode=iteration)
      # except Exception as e:
      #   print(f'Plotting Subprocess Error: {e}')
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
  def _make_spatial_plot(hash2val, title=""):
    hashes = list(hash2val.keys())
    n_boxes = (len(hashes[0]) - 3) // 2
    n_subplots = n_boxes + 1
    n_rows, n_cols = GSMPlotter.subplot_grid(n_subplots)
    for i in range(n_subplots):
      xy2vals = collections.defaultdict(list)
      for hash in hashes:
        x = hash[2 * i]
        y = hash[2 * i + 1]
        bonus = hash2val[hash]
        xy2vals[(x, y)].append(bonus)
      plt.subplot(n_rows, n_cols, i + 1)
      xs, ys, values = [], [], []
      for (x, y), values_list in xy2vals.items():
        xs.append(x)
        ys.append(y)
        values.append(np.mean(values_list))
      plt.scatter(xs, ys, c=values, s=60, marker='s')
      plt.colorbar()
      title = 'PlayerPos' if i == 0 else f'Box {i}'
      plt.title(title)

  def _plot_hash2bonus(self, hash2bonus, episode):
    # the hashes are (player_x, player_y, box1_x, box1_y, ..., boxn_x, boxn_y, reached).
    # I want a subplot for each box, and one for the player.
    self._make_spatial_plot(hash2bonus)
    plt.savefig(os.path.join(self._node_expansion_prob_dir, f'expansion_probs_{episode}.png'))
    plt.close()

  @staticmethod
  def plot_graph_with_positions(graph, positions, title):
    """
    Plots a graph using specified spatial positions for each node.

    Args:
    graph (nx.Graph): The graph to plot.
    positions (dict): A dictionary mapping nodes to (x, y) coordinates.
    title (str): Title of the plot.
    """
    nx.draw(graph, pos=positions, with_labels=True, node_size=700, node_color='skyblue')
    
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edges(graph, pos=positions)
    nx.draw_networkx_edge_labels(graph, pos=positions, edge_labels=edge_labels)
    
    plt.title(title)
    plt.grid('on')

  def _plot_skill_graph(self, edges, off_policy_edges, edge2return, episode, include_off_policy_edges=True):
    """Spatially plot the nodes and edges of the skill-graph."""

    def create_graphs(transitions, n_boxes, edge2return):
      positions = {}
      graphs = [nx.DiGraph() for _ in range(n_boxes + 1)]

      for i in range(n_boxes + 1):
        for transition in transitions:
          start_xy = transition[0][2 * i:2 * i + 2]
          end_xy = transition[1][2 * i:2 * i + 2]
          # skip self loops
          if start_xy == end_xy:
            continue
          if transition[0] in edge2return and transition[1] in edge2return[transition[0]]:
            w = np.round(edge2return[transition[0]][transition[1]], 3)
            graphs[i].add_edge(start_xy, end_xy, weight=w)
          else:
            graphs[i].add_edge(start_xy, end_xy)
          positions[start_xy] = start_xy
          positions[end_xy] = end_xy
      return graphs, positions
    
    def plot_edges(edges, rewards=None):
      edges = list(edges)
      n_boxes = (len(edges[0][0]) - 3) // 2
      graphs, positions = create_graphs(edges, n_boxes, edge2return)
      n_subplots = n_boxes + 1
      n_rows, n_cols = GSMPlotter.subplot_grid(n_subplots)
      default_figsize = plt.rcParams['figure.figsize']
      
      plt.figure(figsize=(3 * default_figsize[0] * n_cols, default_figsize[1] * n_rows * 2))
      
      for i, graph in enumerate(graphs):
        title = 'PlayerPos' if i == 0 else f'Box {i}'
        plt.subplot(n_rows, n_cols, i + 1)
        self.plot_graph_with_positions(graph, positions, title)

    if edges:
      plot_edges(edges)
      plt.savefig(os.path.join(self._skill_graph_plotting_dir, f'online_skill_graph_{episode}.png'))
      plt.close()

    if off_policy_edges:
      plot_edges(off_policy_edges)
      plt.savefig(os.path.join(self._off_policy_graph_plotting_dir, f'offline_skill_graph_{episode}.png'))
      plt.close()

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
    
    def spatial_plot(hash2counts, title):
      hashes = list(hash2counts.keys())
      xs, ys, counts = [], [], []
      for hash in hashes:
        if hash2counts[hash] > 0:
          xs.append(hash[0])
          ys.append(hash[1])
          counts.append(np.log(hash2counts[hash]))
      plt.scatter(xs, ys, c=counts, s=60, marker='s')
      plt.colorbar()
      plt.title(title)
    
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
    """Spatially plot the AMDP V* for 4 randomly sampled goal nodes."""

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

