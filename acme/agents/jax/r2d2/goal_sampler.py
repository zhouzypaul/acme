"""To pick expansion node in the graph and an abstract policy that reaches it."""

import time
import ipdb
import random
import numpy as np
import jax.numpy as jnp
import networkx as nx
import collections

from typing import List, Tuple, Dict

from acme.wrappers.oar_goal import OARG
from acme.utils.utils import scores2probabilities
from acme.agents.jax.r2d2.amdp import AMDP


class GoalSampler:
  def __init__(
      self,
      goal_dict, 
      count_dict,
      bonus_dict,
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
      method: str = 'amdp',
      ignore_non_rewarding_terminal_nodes: bool = False,
      rmax_factor: float = 2.,
      max_vi_iterations: int = 20,
      goal_space_size: int = 100,
      should_switch_goal: bool = False,
      success_rate_dict: Dict = None,
      discovered_goals: set = None):  # All discovered goals (including HER)
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
    self.bonus_dict = collections.defaultdict(float, bonus_dict)
    self.reward_dict = reward_dict
    self.discount_dict = discount_dict
    self.on_policy_edge_count_dict = on_policy_edge_count_dict
    self.transition_tensor = transition_tensor
    self.hash2idx = hash2idx
    self.idx2hash = idx2hash
    self.rmax_factor = rmax_factor
    self.max_vi_iterations = max_vi_iterations
    self._goal_space_size = goal_space_size
    self._should_switch_goal = should_switch_goal
    
    self._n_courier_errors = 0
    
    self._task_goal_hash = tuple(task_goal.goals)
    self._exploration_goal_hash = tuple(exploration_goal.goals)

    # This is to avoid picking death nodes as expansion nodes b/c 
    # the exploration policy can't be run from there anyway.
    self._ignore_non_rewarding_terminal_nodes = ignore_non_rewarding_terminal_nodes
    
    # Store success rates and discovered goals for goal selection
    self.success_rate_dict = success_rate_dict if success_rate_dict is not None else {}
    self.discovered_goals = discovered_goals if discovered_goals is not None else set()

    print(f'Created GoalSampler with GS size {goal_space_size} and method {method}.')

  def begin_episode(self, current_node: Tuple) -> Tuple:
    goal_dict = self.get_candidate_goals(current_node)
    if len(goal_dict) > 0:
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
        rmax_factor=self.rmax_factor,
        max_vi_iterations=self.max_vi_iterations,
        should_switch_goal=self._should_switch_goal,
      )
      print(f'[GoalSampler] Took {t1 - t0}s to select expansion node.')
      print(f'[GoalSampler] Took {time.time() - t1}s to create & solve AMDP.')
      goal_sequence = self._amdp.get_goal_sequence(
        start_node=current_node,
        goal_node=target_node
      )
      print(f'[GoalSampler] Goal Sequence: {goal_sequence}')
      return target_node

  def is_death_node(self, g):
    try:
      reward = self.reward_dict[g]
    except KeyError:
      print(f'[GoalSampler] KeyError for {g}, existing goals: {self.reward_dict.keys()}')
      raise KeyError
    try:
      discount = self.discount_dict[g]
    except KeyError:
      print(f'[GoalSampler] KeyError for {g}, existing goals: {self.discount_dict.keys()}')
      raise KeyError
    return reward <= 0 and discount == 0

  def get_candidate_goals(self, current_node: Tuple) -> Dict:
    """Get the possible goals to pursue at the current state."""
    # at_goal = lambda g: (timestep.observation.goals == g).all()
    at_goal = lambda g: all([g1 == g2 for g1, g2 in zip(current_node, g) if g2 >= 0])
    not_special_context = lambda g: g != self._exploration_goal_hash and g != self._task_goal_hash
    is_death = lambda g: self._ignore_non_rewarding_terminal_nodes and self.is_death_node(g)
    
    # Filter out never-achieved pre-learned goals, but only after some goals have been achieved
    # This allows initial exploration to sample from all pre-loaded goals
    # A goal is considered "achieved" if it's in discovered_goals OR has success_rate > 0
    any_goals_achieved = (len(self.discovered_goals) > 0) or any(rate > 0 for rate in self.success_rate_dict.values()) if self.success_rate_dict else False
    if any_goals_achieved:
      # Once we've achieved at least one goal, only include goals that have been discovered or achieved
      has_been_achieved = lambda g: (g in self.discovered_goals) or (self.success_rate_dict.get(g, 0.0) > 0)
    else:
      # During initial exploration, include all goals
      has_been_achieved = lambda g: True
    
    return {
      goal: oar for (goal, oar) in self.goal_dict.items()
        if not at_goal(goal) and not_special_context(goal) and not is_death(goal) and has_been_achieved(goal)
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
      if dist is None or len(dist[0]) <= 1:
        reachables = list(goal_dict.keys())
        probs = np.ones((len(reachables),)) / len(reachables)
        print('[GoalSampler] No reachable nodes, using uniform distribution.')
      else:
        reachables = dist[0]
        probs = dist[1]
      idx = np.random.choice(range(len(reachables)), p=probs)
      return reachables[idx]
    raise NotImplementedError(method)
  
  # TODO(ab): Support using the exploration value function.
  def _get_expansion_scores(self, reachable_goals, use_tabular_counts=False, success_rate_dict=None, exploration_rate=0.2):
    """Compute scores for goal selection.
    
    Favors goals with low but non-zero success rates (learnable goals).
    Includes exploration bonus to try never-achieved goals.
    
    Args:
      reachable_goals: List of candidate goals
      use_tabular_counts: Use visit counts if no success rates
      success_rate_dict: Maps goals to success rates
      exploration_rate: Probability mass for exploring never-achieved goals (default 0.2 = 20%)
    
    Scoring strategy:
    - Goals with 10-30% success: highest score (optimal learning zone)
    - Goals with 0% success: moderate score (exploration bonus)
    - Goals with >50% success: low score (already mastered)
    """
    if success_rate_dict is not None and len(success_rate_dict) > 0:
      scores = []
      for g in reachable_goals:
        success_rate = success_rate_dict.get(g, 0.0)
        
        # Compute score using inverted bell curve
        # Peak at optimal_rate (20%), drops off for 0% and high rates
        optimal_rate = 0.2  # 20% success rate is ideal for learning
        width = 0.3  # How quickly score drops off from optimal
        
        if success_rate == 0.0:
          # Never achieved - rely purely on intrinsic novelty bonus
          # The existing CFN/RND system handles exploration
          curriculum_score = 0.0
        else:
          # Gaussian curve centered at optimal learning rate
          curriculum_score = np.exp(-((success_rate - optimal_rate) ** 2) / (2 * width ** 2))
          curriculum_score = curriculum_score * 1.5  # Boost learnable goals
        
        # Get intrinsic novelty bonus from existing CFN/RND system
        novelty_bonus = self.bonus_dict.get(g, 0.0)
        
        # Combine: 70% curriculum (exploitation), 30% novelty (exploration)
        novelty_weight = 0.3
        combined_score = (1 - novelty_weight) * curriculum_score + novelty_weight * novelty_bonus
        
        scores.append(combined_score)
      
      return scores
    
    # Fallback to novelty-based scoring if no success rates available
    if use_tabular_counts:
      return [1. / np.sqrt(self.count_dict[g] + 1) for g in reachable_goals]
    return [self.bonus_dict[g] for g in reachable_goals]

  def _get_target_node_probability_dist(
    self,
    current_node: Tuple,
    default_behavior: str = 'none',
    sampling_type: str = 'sort_then_sample'
  ) -> Tuple[List[Tuple], np.ndarray]:
    assert default_behavior in ('task', 'exploration', 'none'), default_behavior
    assert sampling_type in ('argmax', 'sum_sample', 'sort_then_sample'), sampling_type
    
    # TODO(ab): handle case where we can be in multiple current nodes.
    t0 = time.time()
    reachable_goals = self.get_descendants(current_node)
    print(f'[GoalSampler] Took {time.time() - t0} to get descendants.')

    if reachable_goals:
      scores = self._get_expansion_scores(reachable_goals, success_rate_dict=self.success_rate_dict)
      scores = np.asarray(scores)
      if sampling_type == 'sum_sample':
        probs = scores2probabilities(scores)
      elif sampling_type == 'sort_then_sample':  # NOTE: Untested.
        # Sort by score, then sample based on scores from the top 10%.
        idx = np.argsort(scores)[::-1]  # descending order sort.
        # n_non_zero_probs = len(scores) // 10 if len(scores) > 50 else len(scores)
        n_non_zero_probs = min(self._goal_space_size, len(scores))
        non_zero_probs_idx  = idx[:n_non_zero_probs]
        probs = np.zeros_like(scores)
        probs[non_zero_probs_idx] = scores2probabilities(scores[non_zero_probs_idx])
      elif sampling_type == 'argmax':
        probs = np.zeros_like(scores)
        probs[np.argmax(scores)] = 1.
      else:
        raise NotImplementedError(sampling_type)
      
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
    if self._ignore_non_rewarding_terminal_nodes:
      reachable_nodes = [node for node in reachable_nodes if not self.is_death_node(node)]
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
