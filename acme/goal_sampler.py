import ipdb
import time
import copy
import dm_env
import random
import numpy as np
import networkx as nx

from typing import Tuple, List

from acme.jax import variable_utils
from acme.wrappers.oar_goal import OARG
from acme.agents.jax.r2d2.gsm import GoalSpaceManager
from acme.agents.jax.r2d2.amdp import AMDP


class GoalSampler:
  """Policy that outputs the goal for a lower level policy."""
  
  def __init__(self,
               gsm: GoalSpaceManager,
               task_goal_probability: float,
               task_goal: OARG,
               exploration_goal: OARG,
               exploration_goal_probability: float = 0.,
               method='amdp'):
    assert method in ('task', 'amdp', 'uniform', 'exploration'), method
    
    self._amdp = None
    self._goal_space_manager = gsm
    self._sampling_method = method
    self._task_goal = task_goal
    self._exploration_goal = exploration_goal
    self._task_goal_probability = task_goal_probability
    self._exploration_goal_probability = exploration_goal_probability
    
    self.value_dict = {}
    self.goal_dict = {}
    self.count_dict = {}
    self.reward_dict = {}
    self.discount_dict = {}
    self.on_policy_edge_count_dict = {}
    
    self._n_courier_errors = 0
    
    self._task_goal_hash = tuple(task_goal.goals)
    self._exploration_goal_hash = tuple(exploration_goal.goals)

    self.gsm_variable_client = variable_utils.VariableClient(
      gsm, key='gsm_variables', update_period=1500) if gsm else None

    if gsm:
      self.fetch_gsm_dictionaries()

  def fetch_gsm_dictionaries(self):
    self.value_dict, \
    self.goal_dict, \
    self.count_dict, \
    self.on_policy_edge_count_dict, \
    self.reward_dict, \
    self.discount_dict = self.gsm_variable_client.params
      
  def _update_gsm_dicts(self):
    """Update GSM dicts using try-except to account for GSM erroring b/c of threads."""
    if self._goal_space_manager is not None:
      try:
        self.fetch_gsm_dictionaries()
        print('[GoalSampler] Updated all GSM Dicts successfully.')
      except Exception as e:  # If error, keep the old stale copy of the dicts
        self._n_courier_errors += 1
        print(f'[GoalSampler] Warning: Courier error # {self._n_courier_errors}. Exception: {e}')

  def begin_episode(self, timestep: dm_env.TimeStep) -> Tuple:
    t0 = time.time()
    self._update_gsm_dicts()
    print(f'[GoalSampler] Took {time.time() - t0}s to update GSM dicts.')
    
    if self._sampling_method == 'amdp':
      goal_dict = self.get_candidate_goals(timestep)
      if len(goal_dict) > 1 and len(self.value_dict) > 1:
        target_node = self.select_expansion_node(
          timestep, goal_dict, method='novelty')
        print(f'[GoalSampler] Target Node = {target_node}')
        t0 = time.time()
        self._amdp = AMDP(
          value_dict=self.value_dict,
          reward_dict=self.reward_dict,
          discount_dict=self.discount_dict,
          target_node=target_node,
          count_dict=self.on_policy_edge_count_dict)
        print(f'[GoalSampler] Took {time.time() - t0}s to create & solve AMDP.')
        goal_sequence = self._amdp.get_goal_sequence(
          start_node=tuple(timestep.observation.goals), goal_node=target_node)
        print(f'[GoalSampler] Goal Sequence: {goal_sequence}')
        return target_node
    
  def get_candidate_goals(self, timestep: dm_env.TimeStep) -> dict:
    """Get the possible goals to pursue at the current state."""
    # at_goal = lambda g: (timestep.observation.goals == g).all()
    at_goal = lambda g: all([g1 == g2 for g1, g2 in zip(timestep.observation.goals, g) if g2 >= 0])
    not_special_context = lambda g: g != self._exploration_goal_hash and g != self._task_goal_hash
    return {
      goal: oar for (goal, oar) in self.goal_dict.items()
        if not at_goal(goal) and not_special_context(goal)
    }
    
  def __call__(self, timestep: dm_env.TimeStep) -> OARG:
    """Select a goal to pursue in the current episode."""

    if self._sampling_method == 'task' or \
      random.random() < self._task_goal_probability:
      return self._task_goal

    goal_hash = None
    goal_dict = self.get_candidate_goals(timestep)
    
    if len(goal_dict) == 0:
      return self._exploration_goal
    
    # Eps-greedy: 90% of the time, follow AMDP policy; 10% of the time, just explore.
    if self._sampling_method == 'exploration' or \
      random.random() < self._exploration_goal_probability:
      return self._exploration_goal
    
    if len(goal_dict) == 1:
      goal_hash = list(goal_dict.keys())[0]
        
    elif self._sampling_method == 'amdp' and self._amdp is not None:
      goal_hash = self._amdp_goal_sampling(timestep, goal_dict)
      
    elif self._sampling_method == 'uniform':
      goal_hash = random.choice(list(goal_dict.keys()))
      
    if goal_hash is not None and goal_hash in goal_dict:
      return OARG(
        np.asarray(goal_dict[goal_hash][0],
                  dtype=timestep.observation.observation.dtype),
        action=goal_dict[goal_hash][1],
        reward=goal_dict[goal_hash][2],
        goals=np.asarray(
          goal_hash, dtype=timestep.observation.goals.dtype
        ))
      
    # TODO(ab): when do we hit this case and why?
    return self._task_goal  # if random.random() < 0.5 else self._exploration_goal
    
  def _amdp_goal_sampling(self, timestep: dm_env.TimeStep, goal_dict: dict) -> Tuple:
    """Solve the abstract MDP to pick the hash of the goal to pursue at s_t."""
    
    if goal_dict:
      if len(goal_dict) == 1:
        return list(goal_dict.keys())[0]
      if len(self.value_dict) == 1:
        return list(self.value_dict.keys())[0]
      
      if self.value_dict:
        return self._amdp.policy(tuple(timestep.observation.goals))
    
  def _construct_oarg(self, obs, action, reward, goal_features):
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

  # TODO(ab): Plug in actual alg for picking expansion node
  def select_expansion_node(
    self,
    timestep: dm_env.TimeStep,
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
      dist = self.get_target_node_probability_dist(timestep)
      reachables = dist[0] if dist is not None else list(goal_dict.keys())
      probs = dist[1] if dist is not None else np.ones((len(reachables),)) / len(reachables)
      idx = np.random.choice(range(len(reachables)), p=probs)
      return reachables[idx]
    raise NotImplementedError(method)

  def get_target_node_probability_dist(
      self, timestep: dm_env.TimeStep,
      default_behavior: str = 'none') -> Tuple[List[Tuple], np.ndarray]:
    assert default_behavior in ('task', 'exploration', 'none'), default_behavior

    current_node = tuple(timestep.observation.goals)
    reachable_goals = self.get_descendants(current_node)
    if reachable_goals:
      scores = [1. / np.sqrt(self.count_dict[g] + 1) for g in reachable_goals]
      scores = np.asarray(scores)
      probs = scores / scores.sum()
      return reachable_goals, probs
    
    if default_behavior == 'exploration':
      return [self._exploration_goal_hash], np.array([1.], dtype=np.float32)
    if default_behavior == 'task':
      return [self._task_goal_hash], np.array([1.], dtype=np.float32)

  def _construct_skill_graph(self) -> nx.DiGraph:
    graph = nx.DiGraph()
    for src in self.value_dict:
      for dest in self.value_dict[src]:
        if self.value_dict[src][dest] > 0:  # TODO(ab): threshold
          graph.add_edge(src, dest)
    return graph

  def get_descendants(self, src_node: Tuple) -> List[Tuple]:
    graph = self._construct_skill_graph()
    if src_node not in graph.nodes:
      return []
    reachable_goals = nx.algorithms.dag.descendants(graph, src_node)
    reachable_goals = list(reachable_goals)
    if self._exploration_goal_hash in reachable_goals:
      reachable_goals.remove(self._exploration_goal_hash)
    if self._task_goal_hash in reachable_goals:
      reachable_goals.remove(self._task_goal_hash)
    return reachable_goals
