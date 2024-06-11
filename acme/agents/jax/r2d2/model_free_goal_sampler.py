"""Pick an expansion node for the goal-conditioned policy to pursue."""

import random
import numpy as np
import jax.numpy as jnp

from typing import List, Tuple, Set

from acme.wrappers.oar_goal import OARG
from acme.utils.utils import scores2probabilities


class MFGoalSampler:
  def __init__(
    self,
    proto_dict,  # maps hash -> proto np array
    count_dict,  # maps hash -> count
    bonus_dict, # maps hash -> CFN bonus
    binary_reward_func,  # f: (s.goals, g.goals) -> {0, 1}
    goal_space_size: int = 10,
    uvfa_params=None,
    uvfa_rng_key=None,
    uvfa_networks=None,
    use_uvfa_reachability: bool = False
  ):
    self.proto_dict = proto_dict
    self.count_dict = count_dict
    self.bonus_dict = bonus_dict
    self.binary_reward_func = binary_reward_func
    self.goal_space_size = goal_space_size
    self.use_uvfa_reachability = use_uvfa_reachability

    def get_recurrent_state(batch_size=None):
      return uvfa_networks.init_recurrent_state(uvfa_rng_key, batch_size)

    def _oarg2probabilities(batch_oarg):
      """Use the UVFA network to compute the edge probabilities."""
      
      lstm_state = get_recurrent_state(batch_oarg.observation.shape[1])
        
      values, _ = uvfa_networks.unroll(
        uvfa_params,
        uvfa_rng_key,
        batch_oarg,
        lstm_state
      )  # (1, B, |A|)
      values = values.max(axis=-1)[0]  # (1, B, |A|) -> (1, B) -> (B,)
      
      return values
    
    def _value_function(current_state: OARG, nodes: List[Tuple]) -> List[float]:
      """Query the value function for each node in the list."""
      def binary2img(proto_goal: np.ndarray) -> np.ndarray:
        assert proto_goal.dtype == bool, proto_goal.dtype
        goal_img_shape = (*current_state.observation.shape[:2], 1)
        flat_obs_shape = np.prod(goal_img_shape)  # 84 * 84
        goal_image = np.zeros(flat_obs_shape, dtype=obs.dtype)
        goal_image[np.where(proto_goal)] = 1.
        return goal_image.reshape(goal_img_shape)

      obs = current_state.observation[np.newaxis, ..., :3]  # (1, H, W, 3)
      goal_observations = np.asarray([binary2img(self.proto_dict[n]) for n in nodes])
      obs = np.repeat(obs, goal_observations.shape[0], axis=0)  # (B, H, W, 3)
      actions = np.repeat(current_state.action, goal_observations.shape[0], axis=0)
      rewards = np.repeat(current_state.reward, goal_observations.shape[0], axis=0)
      goals = np.asarray([current_state.goals for _ in range(goal_observations.shape[0])])
      
      combined_observation = np.concatenate([obs, goal_observations], axis=-1)
      batch_oarg = OARG(
        observation=jnp.asarray(combined_observation)[jnp.newaxis, ...],
        action=jnp.asarray(actions)[jnp.newaxis, ...],
        reward=jnp.asarray(rewards)[jnp.newaxis, ...],
        goals=jnp.asarray(goals)[jnp.newaxis, ...]
      )
      values = _oarg2probabilities(batch_oarg)
      return values

    self._value_function = _value_function

  def _construct_oarg(self, obs, action, reward, goal_features) -> OARG:
    """Convert the obs, action, etc from the GSM into an OARG object.

    Args:
        obs (list): obs image in list format
        action (int): action taken when this oarg was seen
        reward (float): gc reward taken when this oarg was seen
        goal_features (tuple): goal hash in tuple format
    """
    return OARG(
      observation=np.asarray(obs, dtype=np.float32),
      action=action,
      reward=reward,
      goals=np.asarray(goal_features, dtype=np.int16)
    )

  def begin_episode(self, current_oarg: Tuple) -> Tuple:
    current_oarg = self._construct_oarg(*current_oarg)
    current_node = tuple(current_oarg.goals)
    goal_dict = self.get_candidate_goals(current_node)
    if goal_dict:
      target_node = self._select_expansion_node(current_oarg, goal_dict, method='novelty')
      return target_node

  def get_candidate_goals(self, current_node: Tuple) -> dict:
    """Get the possible goals to pursue at the current state."""
    at_goal = lambda goal: self.binary_reward_func(np.asarray(current_node), np.asarray(goal))
    keys = list(self.proto_dict.keys())
    return {k: self.proto_dict[k] for k in keys if not at_goal(self.proto_dict[k])}  
  
  def _select_expansion_node(
    self,
    current_oarg: OARG,
    hash2proto: dict,
    method: str
  ) -> Tuple:
    if method == 'random':
      chosen = random.choice(hash2proto.values())
      return tuple(chosen)

    if method == 'novelty':
      hashes = list(hash2proto.keys())
      dist = self._get_target_node_probability_dist(current_oarg, hashes)
      if dist is None or len(dist[0]) <= 1:
        reachables = hashes
        probs = np.ones((len(hashes),)) / len(hashes)
        print('[GoalSampler] No reachable nodes, using uniform distribution.')
      else:
        reachables = dist[0]
        probs = dist[1]
      idx = np.random.choice(range(len(reachables)), p=probs)
      chosen_hash = reachables[idx]
      chosen = hash2proto[chosen_hash]
      return tuple(chosen)
    raise NotImplementedError(method)

  # TODO(ab/mm): Sanity check count_dict and bonus_dict to ensure that they are not consistently missing `g`.
  def _get_expansion_scores(self, reachable_goals, use_tabular_counts=False):
    if use_tabular_counts:
      return [1. / np.sqrt(self.count_dict.get(g, 0) + 1) for g in reachable_goals]
    return [self.bonus_dict.get(g, 1) for g in reachable_goals]

  def _get_reachability_scores(self, current_state: OARG, nodes: List[Tuple]):
    """Query the probability of reaching each node from the current state.

    Args:
      current_state (OARG): The current state of the agent.
      nodes (List): List of tuples where each tuple is the hash key into the proto_dict.

    Returns:
      values: List of probabilities of reaching each node.
    """
    values = self._value_function(current_state, nodes)
    return values.clip(0., 1.)

  def _get_target_node_probability_dist(
    self,
    current_oarg: OARG,
    goal_set: Set[Tuple],
    sampling_type: str = 'sort_then_sample',
    
  ) -> Tuple[List[Tuple], np.ndarray]:
    assert sampling_type in ('argmax', 'sum_sample', 'sort_then_sample'), sampling_type
    
    reachable_goals = goal_set  # TODO(ab/mm): Incorporate descendants

    if reachable_goals:
      scores = self._get_expansion_scores(reachable_goals)
      scores = np.asarray(scores)
      
      if self.use_uvfa_reachability:
        reachability_scores = self._get_reachability_scores(current_oarg, reachable_goals)
        reachability_scores = np.asarray(reachability_scores)
        scores = scores * reachability_scores

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