import copy
import ipdb
import pickle
import numpy as np

from typing import Tuple


class AMDP:
  """Abstract MDP over the skill-graph."""
  
  def __init__(
    self, value_dict, reward_dict, target_node, count_dict=None,
    gamma=0.99, max_vi_iterations=10, vi_tol=1e-3, verbose=True):
    """Abstract MDP in goal-space.

    Args:
        value_dict (dict): node to node probs from gcvf
        reward_dict (dict): extrinsic reward for getting to a node
        count_dict (dict, optional): visit counts from the GSM
        target_node (tuple): features/hash of the expansion node
        gamma (float, optional): discount factor. Defaults to 0.99.
        max_vi_iterations (int, optional): Num iters of VI. Defaults to 10.
        vi_tol (float, optional): Convergence thresh for VI. Defaults to 1e-3.
        verbose (bool, optional): Debugging. Defaults to True.
    """
    self._gamma = gamma
    self._value_dict = value_dict
    self._reward_dict = reward_dict
    self._use_count_bonus = True if count_dict else False
    self._count_dict = count_dict  # node features -> int count
    
    self._action_space = self._construct_action_space()
    self._state_space = self._construct_state_space()  # ind -> features
    self._reverse_state_space = {v: k for k, v in self._state_space.items()}
    
    self._transition_matrix = self._transition_func()
    self._reward_vector, self._discount_vector = self._reward_func(target_node)
    self._vf, self._policy = self.solve_abstract_mdp(max_vi_iterations, vi_tol)
    
    if verbose:
      # self.print_abstract_rf()
      self.print_abstract_policy()

  def policy(self, node_features: Tuple[int, int]) -> Tuple[int, int]:
    state = self._reverse_state_space[node_features]
    return self._state_space[self._policy[state]]
  
  def value(self, node_features: Tuple[int, int]) -> float:
    state = self._reverse_state_space[node_features]
    return self._vf[state]
  
  def print_abstract_policy(self):
    for key in self._reverse_state_space:
      print(f'Abstract Policy at {key}: {self.policy(key)}')
      
  def print_abstract_rf(self):
    for key in self._reverse_state_space:
      idx = self._reverse_state_space[key]
      print(f'Abstract R at {key}: {self._reward_vector[idx]}')

  def _construct_action_space(self) -> dict:
    return {i: node for i, node in enumerate(self._value_dict)}
    
  def _construct_state_space(self) -> dict:
    states = copy.deepcopy(self._action_space)
    states[len(states)] = 'death'
    return states  
  
  def _get_value(self, src_node, dest_node, eps=1e-3, c=0.2) -> float:
    """Get the probability of going from src_node to dest_node."""
    bonus = 0.
    value = 0.

    if self._use_count_bonus:
      bonus = 1. / np.sqrt(self._count_dict[dest_node] + eps)

    # TODO(ab): Careful - why would a key not be in this dict?
    if src_node in self._value_dict and dest_node in self._value_dict[src_node]:
      value = self._value_dict[src_node][dest_node]

    return np.clip(value + (c * bonus), 0., 1.)
  
  def _transition_func(self) -> np.ndarray:
    """Construct the transition matrix for the AMDP.
    States and actions are both nodes. With probability
    value_dict[src][dest] you end up in dest, with prob
    (1 - value_dict[src][dest]) you end up in a death node.

    Args:
      value_dict (dict): src -> dest -> value
      
    Returns:
      transition_matrix (np.ndarray): S x A x S -> R
    """
    n_states = len(self._state_space)
    n_actions = len(self._action_space)
    death_state = len(self._state_space) - 1
    transition_matrix = np.zeros(
      (n_states, n_actions, n_states), dtype=np.float32)
    
    for state in self._state_space:
      for action in self._action_space:
        # Self-loop at the death state
        if state == death_state:
          transition_matrix[state][action][state] = 1.
        else:
          dest = action  # specify where you want to go
          node = self._state_space[state]
          goal = self._state_space[dest]
          prob = self._get_value(node, goal)
          transition_matrix[state][dest][dest] = prob
          transition_matrix[state][dest][death_state] = 1. - prob

    return transition_matrix

  def _reward_func(
    self,
    target_node: Tuple[int, int]
    ) -> Tuple[np.float32, np.float32]:
    """Generate a reward function for the the AMDP.

    Args:
      target_node (tuple): features describing the target node

    Returns:
      rewards: np.float32 vector mapping node -> reward
      discounts: np.float32 vector mapping node -> discount factor
    """
    n_states = len(self._state_space)
    rewards = np.zeros((n_states,), dtype=np.float32)
    discounts = np.zeros((n_states,), dtype=np.float32)
    for state in self._state_space:
      node = self._state_space[state]
      is_goal_node = node == target_node
      rewards[state] = 0. if node == 'death' else self._reward_dict[node]
      rewards[state] += int(is_goal_node)
      discounts[state] = int(not is_goal_node) * self._gamma
    return rewards, discounts

  def solve_abstract_mdp(self, n_iterations, tol):
    num_states, num_actions, _ = self._transition_matrix.shape
    values = np.zeros((num_states,), dtype=np.float32)
    
    for _ in range(n_iterations):
      prev_values = np.copy(values)
      assert self._reward_vector.shape == self._discount_vector.shape
      assert self._reward_vector.shape == prev_values.shape
      target = self._reward_vector + (self._discount_vector * prev_values)
      assert target.shape == (num_states,), target.shape
      # TODO(ab): is this bmm correct?
      Q = self._transition_matrix @ target
      assert Q.shape == (num_states, num_actions), Q.shape
      values = np.max(Q, axis=1)
      assert values.shape == (num_states,), values.shape
        
      if np.max(np.abs(values - prev_values)) < tol:
        break
    
    policy = np.argmax(Q, axis=1)
    print('Values: ', values)
    print('Policy: ', policy)
    return values, policy


if __name__ == '__main__':
  def manhattan_distance(n1: Tuple[int, int], n2: Tuple[int, int]) -> int:
    return abs(n1[0] - n2[0]) + abs(n1[1] - n2[1])
  
  def overwrite_value_dict(v_dict: dict) -> dict:
    for n1 in v_dict:
      for n2 in v_dict:
        if n1 == n2:
          v_dict[n1][n2] = 0.
        else:
          v_dict[n1][n2] = 1. / (manhattan_distance(n1, n2))
    return v_dict
  
  filename = 'examples/baselines/rl_discrete/value_matrix_iteration_4956940.pkl'
  with open(filename, 'rb') as f:
    value_dictionary = pickle.load(f)
    
  value_dictionary = overwrite_value_dict(value_dictionary)
  print('Value dict: ', value_dictionary)
  reward_dictionary = {node: float(node == (4, 1)) for node in value_dictionary}
  expansion_node = (3, 3)
  amdp = AMDP(value_dictionary, reward_dictionary, expansion_node)
