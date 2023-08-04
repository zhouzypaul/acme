import ipdb
import time
import copy
import dm_env
import random
import numpy as np

from typing import Tuple

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
               exploration_goal_probability: float = 0.1,
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
      
  def _update_gsm_dicts(self):
    """Update GSM dicts using try-except to account for GSM erroring b/c of threads."""
    if self._goal_space_manager is not None:
      try:
        # TODO(ab): get rid of the copying, rn it prevents them from going out of sync
        self.value_dict = copy.deepcopy(self._goal_space_manager.get_value_dict())
        self.goal_dict = copy.deepcopy(self._goal_space_manager.get_goal_dict())
        self.count_dict = copy.deepcopy(self._goal_space_manager.get_count_dict())
        self.on_policy_edge_count_dict = copy.deepcopy(
          self._goal_space_manager.get_on_policy_count_dict())
        extrinsics = self._goal_space_manager.get_extrinsic_reward_dicts()
        self.reward_dict = copy.deepcopy(extrinsics[0])
        self.discount_dict = copy.deepcopy(extrinsics[1])
        print('[GoalSampler] Updated all GSM Dicts successfully.')
      except Exception as e:  # If error, keep the old stale copy of the dicts
        self._n_courier_errors += 1
        print(f'[GoalSampler] Warning: Courier error # {self._n_courier_errors}. Exception: {e}')

  def begin_episode(self, timestep: dm_env.TimeStep):
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
    
  def get_candidate_goals(self, timestep: dm_env.TimeStep) -> dict:
    """Get the possible goals to pursue at the current state."""
    # at_goal = lambda g: (timestep.observation.goals == g).all()
    at_goal = lambda g: all([g1 == g2 for g1, g2 in zip(timestep.observation.goals, g) if g2 >= 0])
    return {goal: oar for (goal, oar) in self.goal_dict.items() if not at_goal(goal)}
    
  def __call__(self, timestep: dm_env.TimeStep) -> OARG:
    """Select a goal to pursue in the current episode."""
    if self._sampling_method == 'task' or \
      random.random() < self._task_goal_probability:
      return self._task_goal
    
    if self._sampling_method == 'exploration' or \
      random.random() < self._exploration_goal_probability:
      return self._exploration_goal
      
    goal_hash = None
    goal_dict = self.get_candidate_goals(timestep)
    
    if len(goal_dict) == 0:
      return self._task_goal
    
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
      
    return self._task_goal
    
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
    method: str) -> Tuple:
    if method == 'random':
      potential_goals = list(goal_dict.keys())
      return random.choice(potential_goals)
    if method == 'novelty':
      goals = list(goal_dict.keys())
      scores = [1. / np.sqrt(self.count_dict[g] + 1) for g in goals]
      scores = np.asarray(scores)
      probs = scores / scores.sum()
      idx = np.random.choice(range(len(goals)), p=probs)
      return goals[idx]
    raise NotImplementedError(method)
