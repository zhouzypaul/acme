"""Skill chaining on a single trajectory."""

import collections
import numpy as np
import jax.numpy as jnp

from typing import Tuple, Dict, List
from acme.wrappers.oar_goal import OARG
from acme.utils import utils


def construct_value_table(
    networks,
    params,
    rng_key,
    hash2oarg,
    obs_augment_fn,
  ) -> Dict:
  src_dest_pairs = []
  augmented_observations = []
  actions = []
  rewards = []
  goals = []

  for src in hash2oarg:
    for dest in hash2oarg:
      if src != dest:
        obs = hash2oarg[src]
        goal = hash2oarg[dest]
        oarg: OARG = obs_augment_fn(obs, goal)
        augmented_observations.append(oarg.observation)
        actions.append(oarg.action)
        rewards.append(oarg.reward)
        goals.append(oarg.goals)
        src_dest_pairs.append((src, dest))
    
  if augmented_observations:
    augmented_observations = jnp.asarray(
      augmented_observations)[jnp.newaxis, ...]
    augmented_observations = jnp.asarray(augmented_observations)
    actions = jnp.asarray(actions)[jnp.newaxis, ...]
    rewards = jnp.asarray(rewards)[jnp.newaxis, ...]
    goals = jnp.asarray(goals)[jnp.newaxis, ...]

  batch_oarg = OARG(augmented_observations, actions, rewards, goals)
  lstm_state = networks.init_recurrent_state(
    rng_key,
    batch_oarg.observation.shape[1]
  )

  return _construct_value_table(
    networks,
    params,
    rng_key,
    src_dest_pairs,
    batch_oarg,
    lstm_state
  )

def _construct_value_table(
    networks,
    params,
    rng_key,
    src_dest_pairs,
    batch_oarg,
    lstm_state
  ):
  value_table = {}

  values, _ = networks.unroll(
    params,
    rng_key,
    batch_oarg,
    lstm_state)
  values = values.max(axis=-1)[0]  # (1, B, |A|) -> (1, B) -> (B,)
  
  assert len(values) == len(src_dest_pairs)
  
  for (src, dest), value in zip(src_dest_pairs, values):
    if src not in value_table:
      value_table[src] = {}
    value_table[src][dest] = value.item()

  return value_table


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


def sample_random_subgoals(
    trajectory: List[Tuple],
    hash2oarg: Dict,
    hash2intrinsic: Dict,
    num_subgoals: int,
    method: str = 'weighted'
) -> Dict:
  """Given segment of trajectory between expansion node 
  and the most novel goal (only containing novel goals),
  sample subgoals in between.
  """
  assert method in ('weighted', 'uniform'), method

  num_subgoals = min(num_subgoals, len(trajectory))

  # Compute novelty scores
  scores = np.zeros(len(trajectory))
  
  for i, node in enumerate(trajectory):
    scores[i] = hash2intrinsic[node]

  probs = utils.scores2probabilities(scores) if method == 'weighted' else None

  selected_idx = np.random.choice(
    range(len(trajectory)),
    p=probs,
    size=num_subgoals,
    replace=False
  )

  subgoals = [trajectory[i] for i in selected_idx]

  return {k: hash2oarg[k] for k in subgoals}
