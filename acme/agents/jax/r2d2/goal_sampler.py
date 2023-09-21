"""To pick expansion node in the graph and an abstract policy that reaches it."""

import time
import random
import numpy as np
import jax.numpy as jnp
import networkx as nx
import collections

from typing import List, Tuple, Dict

from acme.wrappers.oar_goal import OARG
# from acme.agents.jax.r2d2.amdp import AMDP
from acme.agents.jax.r2d2.amdp3 import AMDP


class GoalSampler:
  def __init__(
      self,
      goal_dict, 
      count_dict,
      on_policy_edge_count_dict,
      reward_dict,
      discount_dict,
      hash2idx,
      transition_tensor: np.ndarray,
      idx2hash,
      task_goal_probability: float,
      task_goal: OARG,
      exploration_goal: OARG,
      exploration_goal_probability: float = 0.,
      method: str = 'amdp'):
    """Interface layer: takes graph from GSM and gets abstract policy from AMDP."""
    assert method in ('task', 'amdp', 'uniform', 'exploration'), method
    
    self._amdp = None
    self._sampling_method = method
    self._task_goal = task_goal
    self._exploration_goal = exploration_goal
    self._task_goal_probability = task_goal_probability
    self._exploration_goal_probability = exploration_goal_probability

    self.goal_dict = goal_dict
    self.count_dict = count_dict
    self.reward_dict = reward_dict
    self.discount_dict = discount_dict
    self.on_policy_edge_count_dict = on_policy_edge_count_dict
    self.transition_tensor = transition_tensor
    self.hash2idx = hash2idx
    self.idx2hash = idx2hash
    
    self._n_courier_errors = 0
    
    self._task_goal_hash = tuple(task_goal.goals)
    self._exploration_goal_hash = tuple(exploration_goal.goals)

  def begin_episode(self, current_node: Tuple) -> Tuple:
    goal_dict = self.get_candidate_goals(current_node)
    if len(goal_dict) > 1:
      t0 = time.time()
      target_node = self._select_expansion_node(
        current_node, goal_dict, method='novelty')
      print(f'[GoalSampler] Target Node = {target_node}')
      t1 = time.time()
      self._amdp = AMDP(
        transition_tensor=self.transition_tensor,
        hash2idx=self.hash2idx,
        reward_dict=self.reward_dict,
        discount_dict=self.discount_dict,
        count_dict=self.on_policy_edge_count_dict,
        target_node=target_node,
      )
      print(f'[GoalSampler] Took {t1 - t0}s to select expansion node.')
      print(f'[GoalSampler] Took {time.time() - t1}s to create & solve AMDP.')
      goal_sequence = self._amdp.get_goal_sequence(
        start_node=current_node,
        goal_node=target_node
      )
      print(f'[GoalSampler] Goal Sequence: {goal_sequence}')
      return target_node

  def get_candidate_goals(self, current_node: Tuple) -> Dict:
    """Get the possible goals to pursue at the current state."""
    # at_goal = lambda g: (timestep.observation.goals == g).all()
    at_goal = lambda g: all([g1 == g2 for g1, g2 in zip(current_node, g) if g2 >= 0])
    not_special_context = lambda g: g != self._exploration_goal_hash and g != self._task_goal_hash
    return {
      goal: oar for (goal, oar) in self.goal_dict.items()
        if not at_goal(goal) and not_special_context(goal)
    }
  
  def _select_expansion_node(
    self,
    current_node: Tuple,
    goal_dict: dict,
    method: str
  ) -> Tuple:
    if self._sampling_method == 'task' or \
      random.random() < self._task_goal_probability:
      return self._task_goal_hash
    
    if method == 'random':
      potential_goals = list(goal_dict.keys())
      return random.choice(potential_goals)

    if method == 'novelty':
      dist = self._get_target_node_probability_dist(current_node)
      reachables = dist[0] if dist is not None else list(goal_dict.keys())
      probs = dist[1] if dist is not None else np.ones((len(reachables),)) / len(reachables)
      idx = np.random.choice(range(len(reachables)), p=probs)
      return reachables[idx]
    raise NotImplementedError(method)

  def _get_target_node_probability_dist(
    self,
    current_node: Tuple,
    default_behavior: str = 'none'
  ) -> Tuple[List[Tuple], np.ndarray]:
    assert default_behavior in ('task', 'exploration', 'none'), default_behavior
    
    # TODO(ab): handle case where we can be in multiple current nodes.
    t0 = time.time()
    reachable_goals = self.get_descendants(current_node)
    print(f'[GoalSampler] Took {time.time() - t0} to get descendants.')

    if reachable_goals:
      scores = [1. / np.sqrt(self.count_dict[g] + 1) for g in reachable_goals]
      scores = np.asarray(scores)
      probs = scores / scores.sum()
      return reachable_goals, probs
    
    if default_behavior == 'exploration':
      return [self._exploration_goal_hash], np.array([1.], dtype=np.float32)
    if default_behavior == 'task':
      return [self._task_goal_hash], np.array([1.], dtype=np.float32)
  
  def construct_skill_graph(self, threshold=0.) -> nx.DiGraph:
    graph = nx.DiGraph()
    for src in self.hash2idx:
      for dest in self.hash2idx:
        if src != dest:
          row = self.hash2idx[src]
          col = self.hash2idx[dest]
          if self.transition_tensor[row, col] > threshold:
            graph.add_edge(src, dest)
    return graph

  def __get_descendants(self, src_node: Tuple) -> List[Tuple]:
    t0 = time.time()
    graph = self.skill_graph  #self.construct_skill_graph()
    t1 = time.time()
    if src_node not in graph.nodes:
      return []
    reachable_goals = nx.algorithms.dag.descendants(graph, src_node)
    t2 = time.time()
    reachable_goals = list(reachable_goals)
    if self._exploration_goal_hash in reachable_goals:
      reachable_goals.remove(self._exploration_goal_hash)
    if self._task_goal_hash in reachable_goals:
      reachable_goals.remove(self._task_goal_hash)
    t3 = time.time()
    print(f'[GoalSampler] Took {t1-t0}s to construct skill graph.')
    print(f'[GoalSampler] Took {t2-t1}s to compute descendants using nx.')
    print(f'[GoalSampler] Took {t3-t2}s to remove the special contexts from descendants.')
    return reachable_goals
  
  def get_descendants(self, src_node: Tuple, threshold=0.) -> List[Tuple]:
    row = self.hash2idx[src_node]
    adjacency = (self.transition_tensor > threshold).astype(bool)
    reachable_idx = bfs(adjacency, row)
    reachable_nodes = [self.idx2hash[idx] for idx in reachable_idx]
    print(f'[GoalSampler] Nodes reachable from {src_node}: {reachable_nodes}')
    return reachable_nodes

  def _get_descendants(
      self,
      src_node: Tuple,
      threshold: float = 0.5,
      n_steps: int = 10) -> List[Tuple]:
    def get_descendants_from_vector(x):
      idx = jnp.nonzero(x)[0]
      return set(idx.tolist()) if len(idx) > 0 else set()
    
    t0 = time.time()
    row = self.hash2idx[src_node]
    t1 = time.time()
    adjacency_matrix = (self.transition_tensor > threshold)
    t2 = time.time()
    adjacency_matrix = jnp.asarray(adjacency_matrix)
    t3 = time.time()
    descendants = get_descendants_from_vector(adjacency_matrix[row])
    t4 = time.time()

    for i in range(2, n_steps):
      reachables = jnp.linalg.matrix_power(adjacency_matrix, i)
      descendants |= get_descendants_from_vector(reachables[row])
    
    t5 = time.time()

    print(f'[GoalSamplerGetDescendants] Total time: {t5-t0}s.')
    print(f'[GoalSamplerGetDescendants] Took {t1-t0}s to get row.')
    print(f'[GoalSamplerGetDescendants] Took {t2-t1}s to get adjacency matrix.')
    print(f'[GoalSamplerGetDescendants] Took {t3-t2}s to convert A to jnp array.')
    print(f'[GoalSamplerGetDescendants] Took {t4-t3}s to get descendants for 1-step reachables.')
    print(f'[GoalSamplerGetDescendants] Took {t5-t4}s to loop over {n_steps} steps.')

    return [self.idx2hash[d] for d in descendants]
  

def bfs(adj_matrix: np.ndarray[bool], start_idx: int):
  """BFS to get reachable nodes from start_idx in an adjacency matrix."""

  num_nodes = len(adj_matrix)
  visited = [False] * num_nodes
  reachable_nodes = []

  queue = collections.deque()
  queue.append(start_idx)
  visited[start_idx] = True

  while queue:
      node = queue.popleft()
      reachable_nodes.append(node)

      for neighbor in range(num_nodes):
          if adj_matrix[node][neighbor] and not visited[neighbor]:
              queue.append(neighbor)
              visited[neighbor] = True

  return reachable_nodes
