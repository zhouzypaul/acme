import random
import numpy as np

from typing import Dict, Tuple, List, Optional
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
    hash2vstar: Optional[Dict] = None,
    rmax_factor: float = 2.,
    gamma: float = 0.99,
    max_vi_iterations: int = 10,
    vi_tol: float = 1e-3,
    verbose: bool = False,
    should_switch_goal: bool = False,
    use_sparse_matrix: bool = True,
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
    self._hash2vstar = hash2vstar

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

    self._reward_vector, self._discount_vector = self._abstract_reward_function(target_node)
    self._vf, self._policy, self.max_bellman_errors = self._solve_abstract_mdp(max_vi_iterations, vi_tol)
    print(f'[AMDP] Solved AMDP[R-Max={rmax_factor}] with {self._policy.shape} abstract states.')

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
  
  def _abstract_reward_function(self, target_node) -> Tuple[np.ndarray, np.ndarray]:
    """Assign a reward and discount factor to each node in the AMDP."""
    reward_vector = np.zeros((self._n_states,), dtype=np.float32)
    discount_vector = np.zeros((self._n_states,), dtype=np.float32)

    for node, idx in self._hash2idx.items():
      is_goal_node = node == target_node
      extrinsic_reward = (self._rmax_factor // 2) * self._reward_dict.get(node, 0.)
      intrinsic_reward = self._rmax_factor * int(is_goal_node)
      reward_vector[idx] = extrinsic_reward + intrinsic_reward

      extrinsic_discount = self._discount_dict.get(node, 1.)
      intrinsic_discount = int(not is_goal_node)
      discount_vector[idx] = intrinsic_discount * extrinsic_discount * self._gamma

      if discount_vector[idx] == 0 or reward_vector[idx] > 0:
        print(f'[AMDP] State {node} has discount={discount_vector[idx]} & rew={reward_vector[idx]}')
    
    return reward_vector, discount_vector
  
  def _solve_abstract_mdp(self, n_iterations, tol):
    max_bellman_errors = []
    values = self.get_vinit(self._hash2vstar)

    for i in range(n_iterations):
      prev_values = np.copy(values)
      assert self._reward_vector.shape == self._discount_vector.shape
      assert self._reward_vector.shape == prev_values.shape
      target = self._reward_vector + (self._discount_vector * prev_values)
      assert target.shape == (self._n_states,), target.shape

      # New VI update rule that takes advantage of the sparsity of the 
      # transition tensor. This allows us to only store (N, N) transition matrices
      # rather than (N+1, N, N+1) tranasition tensors.
      if self._use_sparse_matrix:
        Q = self._transition_matrix @ diags(target)
      else:
        Q = self._transition_matrix @ np.diag(target)
      
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
    if hash2vstar and self._target_node in self._hash2idx and self._target_node in hash2vstar:
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
