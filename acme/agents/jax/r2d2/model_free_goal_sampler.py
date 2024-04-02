"""Pick an expansion node for the goal-conditioned policy to pursue."""

import random
import numpy as np

from typing import List, Tuple, Set
from acme.utils.utils import scores2probabilities


class MFGoalSampler:
  def __init__(
    self,
    count_dict,  # maps hash -> count
    bonus_dict, # maps hash -> CFN bonus
    binary_reward_func,  # f: (s.goals, g.goals) -> {0, 1}
    goal_space_size: int = 10
  ):
    self.count_dict = count_dict
    self.bonus_dict = bonus_dict
    self.binary_reward_func = binary_reward_func
    self.goal_space_size = goal_space_size

  def begin_episode(self, current_node: Tuple) -> Tuple:
    goal_set = self.get_candidate_goals(current_node)
    if goal_set:
      target_node = self._select_expansion_node(current_node, goal_set, method='novelty')
      return target_node

  def get_candidate_goals(self, current_node: Tuple) -> Set[Tuple]:
    """Get the possible goals to pursue at the current state."""
    at_goal = lambda goal: self.binary_reward_func(np.asarray(current_node), np.asarray(goal))
    return {goal for goal in self.count_dict if not at_goal(goal)}
  
  def _select_expansion_node(
    self,
    current_node: Tuple,
    goal_set: set,
    method: str
  ) -> Tuple:
    if method == 'random':
      return random.choice(goal_set)

    if method == 'novelty':
      dist = self._get_target_node_probability_dist(current_node, goal_set)
      if dist is None or len(dist[0]) <= 1:
        reachables = goal_set
        probs = np.ones((len(goal_set),)) / len(goal_set)
        print('[GoalSampler] No reachable nodes, using uniform distribution.')
      else:
        reachables = dist[0]
        probs = dist[1]
      idx = np.random.choice(range(len(reachables)), p=probs)
      return reachables[idx]
    raise NotImplementedError(method)

  def _get_expansion_scores(self, reachable_goals, use_tabular_counts=False):
    if use_tabular_counts:
      return [1. / np.sqrt(self.count_dict[g] + 1) for g in reachable_goals]
    return [self.bonus_dict[g] for g in reachable_goals]
  
  def _get_target_node_probability_dist(
    self,
    current_node: Tuple,
    goal_set: Set[Tuple],
    sampling_type: str = 'sort_then_sample'
  ) -> Tuple[List[Tuple], np.ndarray]:
    assert sampling_type in ('argmax', 'sum_sample', 'sort_then_sample'), sampling_type
    
    reachable_goals = goal_set  # TODO(ab/mm): Incorporate descendants

    if reachable_goals:
      scores = self._get_expansion_scores(reachable_goals)
      scores = np.asarray(scores)
      if sampling_type == 'sum_sample':
        probs = scores2probabilities(scores)
      elif sampling_type == 'sort_then_sample':
        # Sort by score, then sample based on scores from the top 10%.
        idx = np.argsort(scores)[::-1]  # descending order sort.
        # n_non_zero_probs = len(scores) // 10 if len(scores) > 50 else len(scores)
        n_non_zero_probs = min(self.goal_space_size, len(scores))
        non_zero_probs_idx  = idx[:n_non_zero_probs]
        probs = np.zeros_like(scores)
        probs[non_zero_probs_idx] = scores2probabilities(scores[non_zero_probs_idx])
      elif sampling_type == 'argmax':
        probs = np.zeros_like(scores)
        probs[np.argmax(scores)] = 1.
      else:
        raise NotImplementedError(sampling_type)
      
      return reachable_goals, probs