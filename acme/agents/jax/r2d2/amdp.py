import random
import numpy as np

from collections import defaultdict
from typing import Dict, Tuple, List
from scipy.sparse import csr_matrix, diags


class AMDP:
  def __init__(
    self,
    transition_tensor,
    hash2idx,
    reward_dict,
    discount_dict,
    count_dict, 
    target_node,
    rmax_factor: float = 2.,
    gamma: float = 0.99,
    max_vi_iterations: int = 10,
    vi_tol: float = 1e-3,
    verbose: bool = False,
    should_switch_goal: bool = False,
    use_sparse_matrix: bool = True,
    hash2vstar: Dict = None,
    rmin_for_death_node: float = 0.
  ):
    self._transition_matrix =  csr_matrix(transition_tensor) if \
      use_sparse_matrix and not isinstance(transition_tensor, csr_matrix) else transition_tensor
    self._hash2idx =  hash2idx
    self._reward_dict =  reward_dict
    self._discount_dict =  discount_dict
    self._count_dict =  count_dict
    self._target_node =  target_node
    self._gamma =  gamma
    self._verbose = verbose
    self._rmax_factor = rmax_factor
    self._use_sparse_matrix = use_sparse_matrix
    self._rmin_for_death_node = rmin_for_death_node

    self._use_reward_matrix = any(isinstance(v, (dict, defaultdict)) for v in reward_dict.values())
    print(f'[AMDP] Using reward matrix: {self._use_reward_matrix} and Rmin: {rmin_for_death_node}')

    # TODO(ab): pass this from GoalSampler rather than recomputing it.
    self._idx2hash = {v: k for k, v in hash2idx.items()}
    
    self._n_states, self._n_actions = transition_tensor.shape
    assert self._n_states == self._n_actions, 'Not special-casing the death node here.'

    if target_node not in self._hash2idx and should_switch_goal:
      goal_node = self.sample_rewarding_node()
      if goal_node is not None:
        print(f'[AMDP] *** Switched target node from {target_node} -> {goal_node} ***')
        target_node = goal_node
        self._target_node = target_node

    self._hash2vstar = hash2vstar
    self._reward_function, self._discount_vector = self._abstract_reward_function(target_node)

    self._vf, self._policy, self.max_bellman_errors = self._solve_abstract_mdp(max_vi_iterations, vi_tol)
    print(f'[AMDP] Solved AMDP[R-Max={rmax_factor}] with {self._policy.shape} abstract states.')
    print(f'[AMDP] Using sparse matrix: {self._use_sparse_matrix}')

  def get_policy(self) -> Dict:
    """Serialize the policy vector into a dictionary with goal_hash -> goal_hash."""
    return {node: self._idx2hash[self._policy[idx]] for node, idx in self._hash2idx.items()}
  
  def get_values(self) -> Dict:
    """Serialize the value function into a dictionary with goal_hash -> value."""
    return {node: self._vf[idx] for node, idx in self._hash2idx.items()}
  
  def sample_rewarding_node(self):
    """If the target node is the task goal, then switch it out for a rewarding node."""
    rewarding_nodes = [node for node, rew in self._reward_dict.items() if rew > 0]
    terminal_rewarding_nodes = [node for node in rewarding_nodes if self._discount_dict[node] == 0]
    if len(terminal_rewarding_nodes) > 0:
      return random.choice(terminal_rewarding_nodes)

  def _construct_extrinsic_reward_matrix(self):
    """Construct the extrinsic reward matrix R(s, s') for the AMDP."""
    # Assume that reward_dict is a nested dictionary of the form: {s: {s': r}}
    reward_matrix = np.zeros((self._n_states, self._n_states), dtype=np.float32)
    for src_node, src_idx in self._hash2idx.items():
      if src_node in self._reward_dict:
        assert isinstance(self._reward_dict[src_node], (dict, defaultdict)), self._reward_dict[src_node]
      for dst_node, dst_idx in self._hash2idx.items():
        reward = self._reward_dict.get(src_node, {}).get(dst_node, 0.)
        reward_matrix[src_idx, dst_idx] = reward * (self._rmax_factor / 2)
    return reward_matrix

  def _construct_extrinsic_reward_vector(self):
    """Construct the extrinsic reward vector for the AMDP."""
    reward_vector = np.zeros((self._n_states,), dtype=np.float32)
    for node, idx in self._hash2idx.items():
      reward_vector[idx] = self._reward_dict.get(node, 0.)
    return reward_vector

  def _construct_intrinsic_reward_vector(self):
    """Construct the intrinsic reward vector for the AMDP."""
    intrinsic_reward_vector = np.zeros((self._n_states,), dtype=np.float32)
    for node, idx in self._hash2idx.items():
      is_goal_node = node == self._target_node
      intrinsic_reward_vector[idx] = int(is_goal_node) * self._rmax_factor
    return intrinsic_reward_vector

  def _construct_continuation_vector(self):
    """Construct the discount vector for the AMDP."""
    continuation_vector = np.zeros((self._n_states,), dtype=np.float32)
    for node, idx in self._hash2idx.items():
      is_goal_node = node == self._target_node
      extrinsic_continuation = self._discount_dict.get(node, 1.)
      intrinsic_continuation = int(not is_goal_node)
      continuation_vector[idx] = intrinsic_continuation * extrinsic_continuation
    return continuation_vector

  def _abstract_reward_function(self, target_node) -> Tuple[np.ndarray, np.ndarray]:
    discount = self._construct_continuation_vector() * self._gamma
    assert discount.shape == (self._n_states,), discount.shape
    
    if self._use_reward_matrix:
      extrinsic_reward_matrix = self._construct_extrinsic_reward_matrix()
      intrinsic_reward_vector = self._construct_intrinsic_reward_vector()
      assert extrinsic_reward_matrix.shape == (self._n_states, self._n_states), extrinsic_reward_matrix.shape
      assert intrinsic_reward_vector.shape == (self._n_states,), intrinsic_reward_vector.shape
      combined = extrinsic_reward_matrix + intrinsic_reward_vector[np.newaxis, :]
      assert combined.shape == (self._n_states, self._n_states), combined.shape
      return combined, discount

    extrinsic_reward_vector = self._construct_extrinsic_reward_vector()
    intrinsic_reward_vector = self._construct_intrinsic_reward_vector()
    assert extrinsic_reward_vector.shape == intrinsic_reward_vector.shape == (self._n_states,), \
      (extrinsic_reward_vector.shape, intrinsic_reward_vector.shape)
    return extrinsic_reward_vector + intrinsic_reward_vector, discount

  def _vector_case_q_update(self, prev_values):
    """New VI update rule that takes advantage of the sparsity of the 
    transition tensor. This allows us to only store (N, N) transition matrices
    rather than (N+1, N, N+1) tranasition tensors."""
    assert self._reward_function.shape == self._discount_vector.shape
    assert self._reward_function.shape == prev_values.shape
    target = self._reward_function + (self._discount_vector * prev_values)
    assert target.shape == (self._n_states,), target.shape
    if self._use_sparse_matrix:
      return self._transition_matrix @ diags(target)
    return self._transition_matrix @ np.diag(target)

  def _matrix_case_q_update(self, prev_values):
    """Update rule when the reward function is a matrix R(s, s')."""
    assert prev_values.shape == (self._n_states,), prev_values.shape
    assert self._reward_function.shape == (self._n_states, self._n_states)
    assert self._discount_vector.shape == (self._n_states,), self._discount_vector.shape
    
    if self._use_sparse_matrix:
      expected_reward = self._transition_matrix.multiply(self._reward_function)
      expected_next_value = self._transition_matrix @ diags(prev_values)
      if self._rmin_for_death_node != 0:
        one_matrix = csr_matrix(np.ones((self._n_states, self._n_states), dtype=np.float32))
        death_prob = one_matrix - self._transition_matrix
        expected_reward = expected_reward + death_prob.multiply(self._rmin_for_death_node)
        expected_next_value = expected_next_value + death_prob.multiply(self._rmin_for_death_node)
      return expected_reward + expected_next_value.multiply(self._discount_vector)
    
    expected_reward = self._transition_matrix * self._reward_function
    expected_next_value = self._transition_matrix @ np.diag(prev_values)
    return expected_reward + (self._discount_vector * expected_next_value)
  
  def _solve_abstract_mdp(self, n_iterations, tol):
    max_bellman_errors = []
    values = self.get_vinit(self._hash2vstar)

    for i in range(n_iterations):
      prev_values = np.copy(values)

      if self._use_reward_matrix:
        Q = self._matrix_case_q_update(prev_values)
      else:
        Q = self._vector_case_q_update(prev_values)
      
      assert Q.shape == (self._n_states, self._n_actions), Q.shape
      values = Q.max(axis=1)
      
      if self._use_sparse_matrix:
        values = values.toarray().flatten()

      assert values.shape == (self._n_states,), values.shape
        
      error = np.max(np.abs(values - prev_values))
      max_bellman_errors.append(error)
      
      if error < tol:
        break

      if self._verbose:
        print(f'[AMDP] VI {i + 1} iters and {error} error.')
    
    if (values == 0).all() or (values == 1).all():
      policy = np.random.randint(
        low=0, high=self._n_actions, size=values.shape
      )
    else:
      policy = self.sparse_q2policy(Q) if self._use_sparse_matrix else self.q2policy(Q)

    assert values.shape == policy.shape == (self._n_states,), (values.shape, policy.shape)

    return values, policy, max_bellman_errors

  def q2policy(self, q_table: np.ndarray) -> np.ndarray:
    """Argmax over Q that accounts for initiation sets and random tie breaking."""
    def randargmax(x):
      """A random tie-breaking argmax."""
      return np.argmax(x + np.random.random(x.shape) * 1e-6, axis=1)
    q_modified = np.where(self._transition_matrix > 0, q_table, -10_000.)
    return randargmax(q_modified)

  def sparse_q2policy(self, q_table: csr_matrix) -> np.ndarray:
    """Argmax over Q that accounts for initiation sets and random tie breaking."""
    def randargmax(x):
      """A random tie-breaking argmax."""
      return np.argmax(x + np.random.random(x.shape) * 1e-6, axis=1)
    nonzero_rows, nonzero_cols = self._transition_matrix.nonzero()
    q_modified = np.ones((self._n_states, self._n_actions), dtype=np.float32) * -10_000.
    q_modified[nonzero_rows, nonzero_cols] = q_table[nonzero_rows, nonzero_cols]
    print(f'[AMDP] Number of nonzero entries: {len(nonzero_rows)} out of {self._n_states * self._n_actions}')
    return randargmax(q_modified)
  
  def get_vinit(self, hash2vstar):
    """Get the initial value function for the AMDP."""
    v0 = np.zeros((self._n_states,), dtype=np.float32)
    if hash2vstar and self._target_node in self._hash2idx:
      print(f'[AMDP] Using hash2vstar to initialize AVF for {self._target_node}')
      for node, idx in self._hash2idx.items():
        v0[idx] = hash2vstar[self._target_node].get(node, 0.)
    return v0

  def get_goal_sequence(
    self, start_node: Tuple, goal_node: Tuple, max_len: int = 20
  ) -> List[Tuple]:
    """Get the sequence of subgoals from start -> goal."""
    i = 0
    current = start_node
    path = [start_node]
    policy = self.get_policy()
    if current in self._hash2idx:
      while goal_node not in path and i < max_len:
        current = policy[current]
        path.append(current)
        i += 1
    return path
