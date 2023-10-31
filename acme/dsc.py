"""Skill chaining on a single trajectory."""

import collections
import numpy as np

from typing import Tuple, Dict, List


def compute_trajectory_probability(
    state: Tuple,
    subgoal: Tuple,
    goal: Tuple,
    node2node_probability_table: Dict,
    discount: float = 1.
  ) -> float:
  """Compute the probability of a trajectory given a subgoal."""
  log_prob1 = np.log(node2node_probability_table[state, subgoal])
  log_prob2 = np.log(node2node_probability_table[subgoal, goal])
  overall_log_prob = log_prob1 + (discount * log_prob2)
  # Convert log probability to probability
  overall_prob = np.exp(overall_log_prob)
  return overall_prob


def one_level_skill_chaining(
    trajectory: List[Tuple],
    value_table: Dict
  ) -> Tuple[Tuple, int, float]:
  assert len(trajectory) > 2, len(trajectory)

  start_state = trajectory[0]
  goal_state = trajectory[-1]
  baseline_score = value_table[start_state, goal_state]
  
  n_candidate_subgoals = len(trajectory) - 2
  subgoal_scores = [0.] * n_candidate_subgoals
  
  for i in range(n_candidate_subgoals):
    subgoal = trajectory[i+1]
    subgoal_scores[i] = compute_trajectory_probability(
      start_state,
      subgoal,
      goal_state,
      value_table)
  
  chosen_idx = np.argmax(subgoal_scores)
  chosen_subgoal = trajectory[chosen_idx]
  chosen_subgoal_score = subgoal_scores[chosen_idx]

  print(f'baseline score: {baseline_score} | ',
        f'chosen subgoal score: {chosen_subgoal_score} | ',
        f'chosen subgoal: {chosen_subgoal}')

  if max(subgoal_scores) > baseline_score:
    return chosen_subgoal, chosen_idx, chosen_subgoal_score

  return goal_state, -1, baseline_score


def recursive_skill_chaining(
    trajectory: List[Tuple],
    value_table: Dict,
) -> List[Tuple]:
  """Recursively chain skills on a trajectory."""
  start_state = trajectory[0]
  goal_state = trajectory[-1]
  baseline_score = value_table[start_state, goal_state]
  subgoals = collections.deque([])

  while len(trajectory) > 2:
    subgoal, subgoal_idx, subgoal_score = one_level_skill_chaining(
      trajectory,
      value_table)

    # No more subgoals to be found
    if subgoal == goal_state or subgoal == start_state:
      break
    
    baseline_score = value_table[start_state, goal_state]
    
    if subgoal_score > baseline_score:
      # We need the +1 to include the subgoal
      trajectory = trajectory[:subgoal_idx+1]
      subgoals.appendleft(subgoal)

  return list(subgoals)

  