import ipdb
import time
import dm_env
import random
import numpy as np

from acme.wrappers.oar_goal import OARG
from acme.agents.jax.r2d2.gsm import GoalSpaceManager


class GoalSampler:
  """Policy that outputs the goal for a lower level policy."""
  
  def __init__(self,
               gsm: GoalSpaceManager,
               task_goal_features=(6, 6)):
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

  def construct_abstract_mdp(self, skill_graph):
    pass
