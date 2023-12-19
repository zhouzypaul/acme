import os
import time
import json
import pickle
import random
import itertools
import collections
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Optional, Dict
from acme.utils.paths import get_save_directory

import acme.agents.jax.cfn.plotting as cfn_plotting


class GSMPlotter:
  def __init__(self, time_between_plots=60 * 5):
    self._time_between_plots = time_between_plots
    self._reward_means = []
    self._reward_variances = []

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
      on_policy_counts=_dict_to_default_dict(state[3], int),
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
      hash2success=state[16],
    )

  def __call__(self, episode=0):
    vars = self.get_gsm_variables()
    if vars:
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
      self._plot_hash2bonus(vars['hash2bonus'], episode)
      self._plot_skill_graph(vars['edges'], vars['off_policy_edges'], episode)
      self._plot_bellman_errors(vars['hash2bellman'], episode)
      self._plot_spatial_vstar(vars['hash2vstar'], episode)
      self._plot_gsm_iteration_times(vars['gsm_iteration_times'])
      self._plot_per_goal_success_curves(vars['hash2success'], episode)
      self._plot_reward_mean_and_variance(episode=-1)

      cfn_plotting.plot_average_bonus_for_each_hash_bit(
        vars['hash2bonus'],
        save_path=os.path.join(self._hash_bit_plotting_dir, f'mean_bonus_{episode}.png'))
      
      self._plot_avg_value_for_interesting_hash_bits(
        vars['hash2idx'], vars['transition_matrix'], episode)

  def run(self):
    for iteration in itertools.count():
      t0 = time.time()
      self(episode=iteration)
      t1 = time.time()
      print(f'Plotted iteration {iteration} in {t1 - t0:.3f} seconds.')
      time.sleep(max(0, self._time_between_plots - (t1 - t0)))

  def visualize_value_function(
    self,
    episode: int,
    hash2idx: Dict[Tuple, int],
    transition_matrix: np.ndarray,
    task_value_vector: Optional[np.ndarray] = None,
    exploration_value_vector: Optional[np.ndarray] = None,
    n_goal_subplots: int = 4
  ):
    assert n_goal_subplots % 2 == 0, 'Ask for even number of subplots.'
    assert task_value_vector is None or len(task_value_vector.shape) == 1
    assert exploration_value_vector is None or len(exploration_value_vector.shape) == 1

    plot_task_value_fn = task_value_vector is not None
    plot_explore_value_fn = exploration_value_vector is not None

    def _get_value(src: Tuple, dest: Tuple):
      row = hash2idx[src]
      col = hash2idx[dest]
      return transition_matrix[row, col]

    starts = list(hash2idx.keys())
    
    if len(starts) < n_goal_subplots:
      return
    
    selected_starts = random.sample(starts, k=n_goal_subplots)

    plt.figure(figsize=(14, 14))
    n_subplots = n_goal_subplots + \
      int(plot_task_value_fn) + int(plot_explore_value_fn)

    for i, start in enumerate(selected_starts):
      xs = []; ys = []; values = []
      for goal_hash in starts:
        xs.append(goal_hash[0])
        ys.append(goal_hash[1])
        value = _get_value(start, goal_hash)
        values.append(value)
      plt.subplot(n_subplots // 2, n_subplots // 2, i + 1)
      plt.scatter(xs, ys, c=values)
      plt.colorbar()
      plt.title(f'Start State: {start}')

    if plot_task_value_fn:
      xs = []; ys = []; values = []
      for start in starts:
        xs.append(start[0])
        ys.append(start[1])
        values.append(task_value_vector[hash2idx[start]])
      plt.subplot(n_subplots // 2, n_subplots // 2, n_goal_subplots + 1)
      plt.scatter(xs, ys, c=values)
      plt.colorbar()
      plt.title('Task Reward Function')

    if plot_explore_value_fn:
      xs = []; ys = []; values = []
      for start in starts:
        xs.append(start[0])
        ys.append(start[1])
        values.append(exploration_value_vector[hash2idx[start]])
      plt.subplot(n_subplots // 2, n_subplots // 2, n_goal_subplots + 2)
      plt.scatter(xs, ys, c=values)
      plt.colorbar()
      plt.title('Exploration Context')

    plt.savefig(os.path.join(self._gcvf_plotting_dir, f'uvfa_{episode}.png'))
    plt.close()

  def _make_spatial_bonus_plot(self, hash2bonus: Dict, episode: int):
    hashes = list(hash2bonus.keys())
    xs, ys, bonuses = [], [], []
    for hash in hashes:
      xs.append(hash[0])
      ys.append(hash[1])
      bonuses.append(hash2bonus[hash])
    plt.scatter(xs, ys, c=bonuses, s=40, marker='s')
    plt.colorbar()
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

  def _plot_hash2bonus(self, hash2bonus, episode):
    hashes = list(hash2bonus.keys())
    xs, ys, bonuses = [], [], []
    for hash in hashes:
      xs.append(hash[0])
      ys.append(hash[1])
      bonuses.append(hash2bonus[hash])
    plt.scatter(xs, ys, c=bonuses, s=40, marker='s')
    plt.colorbar()
    plt.savefig(os.path.join(self._node_expansion_prob_dir, f'expansion_probs_{episode}.png'))
    plt.close()

  def _plot_skill_graph(self, edges, off_policy_edges, episode, include_off_policy_edges=True):
    """Spatially plot the nodes and edges of the skill-graph."""

    def split_edges(edges, hash_bit):
      """Split edges based on whether the hash bit is on/off for src and dest."""
      no_no = []
      no_yes = []
      yes_no = []
      yes_yes = []
      for edge in edges:
        src_hash = edge[0]
        dest_hash = edge[1]
        if src_hash[hash_bit] == 0 and dest_hash[hash_bit] == 0:
          no_no.append(edge)
        elif src_hash[hash_bit] == 0 and dest_hash[hash_bit] == 1:
          no_yes.append(edge)
        elif src_hash[hash_bit] == 1 and dest_hash[hash_bit] == 0:
          yes_no.append(edge)
        elif src_hash[hash_bit] == 1 and dest_hash[hash_bit] == 1:
          yes_yes.append(edge)
      return no_no, no_yes, yes_no, yes_yes

    def plot_edges(e, color):
      for edge in e:
        x1 = edge[0][0]
        y1 = edge[0][1]
        x2 = edge[1][0]
        y2 = edge[1][1]
        plt.scatter([x1, x2], [y1, y2], color=color)
        plt.plot([x1, x2], [y1, y2], color=color, alpha=0.3)

    def split_then_plot(edges, hash_bit, color):
      no_no, no_yes, yes_no, yes_yes = split_edges(edges, hash_bit=2)
      
      plt.subplot(2, 2, 1)
      plot_edges(no_no, color=color)
      plt.title('No Key -> No Key')
      plt.subplot(2, 2, 2)
      plot_edges(no_yes, color=color)
      plt.title('No Key -> Yes Key')
      plt.subplot(2, 2, 3)
      plot_edges(yes_no, color=color)
      plt.title('Yes Key -> No Key')
      plt.subplot(2, 2, 4)
      plot_edges(yes_yes, color=color)
      plt.title('Yes Key -> Yes Key')
    
    edges = list(edges)
    plt.figure(figsize=(14, 14))
    split_then_plot(edges, hash_bit=2, color='black')
    plt.savefig(os.path.join(self._skill_graph_plotting_dir, f'online_skill_graph_{episode}.png'))
    plt.close()
    
    off_policy_edges = list(off_policy_edges)
    plt.figure(figsize=(14, 14))
    split_then_plot(off_policy_edges, hash_bit=2, color='red')
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
    def plot_vf(hash2val: Dict, name: str):
      xs = []; ys = []; values = []
      for key, val in hash2val.items():
        xs.append(key[0])
        ys.append(key[1])
        values.append(val)

      if values:
        plt.scatter(xs, ys, c=values, s=100, marker='s')
        plt.colorbar()
        plt.title(name)

    nodes = list(hash2vstar.keys())
    
    if nodes:
      selected_nodes = random.sample(nodes, k=min(len(nodes), 4))
      plt.figure(figsize=(14, 14))
      for i, node in enumerate(selected_nodes):
        plt.subplot(2, 2, i + 1)
        plot_vf(hash2vstar[node], name=f'Goal: {node}')
      plt.suptitle(f'AMDP V-Star at GSM Iteration {episode}')
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

