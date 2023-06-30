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
               task_goal_features: Tuple):
    self._goal_space_manager = gsm
    self._task_goal_features = task_goal_features
    
  def __call__(
    self,
    timestep: dm_env.TimeStep,
    method='uniform'
  ) -> OARG:
    """Select a goal to pursue in the upcoming episode."""
    def get_candidate_goals():
      at_goal = lambda g: (timestep.observation.goals == g).all()
      goal_dict = self._goal_space_manager.get_goal_dict()
      return {goal: oar for (goal, oar) in goal_dict.items() if not at_goal(goal)}
    
    if method == 'uniform':
      goal_dict = get_candidate_goals()
      if goal_dict:
        sampled_goal_feats = random.choice(list(goal_dict.keys()))
        return OARG(
          np.asarray(goal_dict[sampled_goal_feats][0],
                     dtype=timestep.observation.observation.dtype),
          action=goal_dict[sampled_goal_feats][1],
          reward=goal_dict[sampled_goal_feats][2],
          goals=np.asarray(
            sampled_goal_feats, dtype=timestep.observation.goals.dtype)
        )
    if method == 'amdp':
      # TODO(ab): get rid of the copying,
      # have it now to prevent them from going out of sync
      value_dict = copy.deepcopy(self._goal_space_manager.get_value_dict())
      goal_dict = copy.deepcopy(get_candidate_goals())
      if value_dict:
        if len(value_dict) == 1:
          goal_features = list(value_dict.keys())[0]
        else:
          # TODO(ab): maintain reward_dict in GSM
          reward_dict = {node: 0. for node in value_dict}
          # TODO(ab): Plug in actual alg for picking expansion node
          target_node = self.select_expansion_node(timestep, goal_dict, method='random')
          print(f'[GoalSampler] TargetNode={target_node}')
          abstract_mdp = AMDP(
            value_dict=value_dict,
            reward_dict=reward_dict,
            target_node=target_node,
            count_dict=self._goal_space_manager.get_count_dict()
          )
          goal_features = abstract_mdp.policy(tuple(timestep.observation.goals))
        if goal_features in goal_dict:
          return OARG(
            np.asarray(goal_dict[goal_features][0],
                      dtype=timestep.observation.observation.dtype),
            action=goal_dict[goal_features][1],
            reward=goal_dict[goal_features][2],
            goals=np.asarray(
              goal_features, dtype=timestep.observation.goals.dtype
            ))
        elif goal_dict:  # if goal_dict is not empty, why doesn't it contain goal_features?
          print(f'[GoalSampler] Warning: {goal_features} not in goal_dict {goal_dict.keys()}')
    
    task_goal_img = np.zeros_like(timestep.observation.observation)
    task_goal_features = np.array(
      self._task_goal_features,
      dtype=timestep.observation.goals.dtype)
    return OARG(
      task_goal_img,
      action=timestep.observation.action,  # doesnt matter
      reward=timestep.observation.reward,  # doesnt matter
      goals=task_goal_features)
    
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

  def select_expansion_node(
    self,
    timestep: dm_env.TimeStep,
    goal_dict: dict,
    method: str) -> Tuple:
    if method == 'random':
      potential_goals = list(goal_dict.keys())
      return random.choice(potential_goals)
    raise NotImplementedError(method)
