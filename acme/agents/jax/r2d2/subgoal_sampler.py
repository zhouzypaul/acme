import time
import copy
import random
import numpy as np
import networkx as nx

from typing import Dict, Tuple, List, Set

from acme.wrappers.oar_goal import OARG


class SubgoalSampler:
  def __init__(
      self,
      abstract_policy: Dict,
      hash2goal: Dict,
      task_goal_probability: float,
      task_goal: OARG,
      exploration_goal_probability: float,
      exploration_goal: OARG,
      edges: Set[Tuple],
      expansion_node: Tuple,
      reward_dict: Dict,
      discount_dict: Dict,
      sampling_method: str = 'amdp',
      default_behavior: str = 'graph_search',
  ):
    """
    Args:
      abstract_policy (dict): map goal hash -> goal hash.
      hash2goal (dict): goal hash -> OARG.
      task_goal_probability (float): prob of pursuing the task goal.
      task_goal (OARG): tensor of all 0s representing the extrinsic context.
      exploration_goal_probability (float): prob of pursuing exploration context.
      exploration_goal (OARG): tensor of all 1s representing the intrinsic context.
      edges (set): set of edges in the graph.
      expansion_node (tuple): target node in the graph.
      reward_dict (dict): map node -> extrinsic reward.
      discount_dict (dict): map node -> extrinsic discount.
      sampling_method (str): AMDP/exploration/task.
      default_behavior (str): graph_search/task.
    """
    assert sampling_method in ('amdp', 'task'), sampling_method
    assert default_behavior in ('graph_search', 'task'), default_behavior

    self._hash2goal = copy.deepcopy(hash2goal)
    self._abstract_policy = abstract_policy
    self._task_goal_probability = task_goal_probability
    self._task_goal = task_goal
    self._exploration_goal_probability = exploration_goal_probability
    self._exploration_goal = exploration_goal
    self._sampling_method = sampling_method

    self._task_goal_hash = tuple(task_goal.goals)
    self._exploration_goal_hash = tuple(exploration_goal.goals)

    self._edges = edges
    self._skill_graph = None
    self._reward_dict = reward_dict
    self._discount_dict = discount_dict
    self._expansion_node = expansion_node
    self._default_behavior = default_behavior

    # If the expansion node is the task goal, then there will be no path to it in the graph.
    # So, if possible, switch it out for a sample goal state to target.
    if expansion_node == self._task_goal_hash:
      goal_node = self._sample_rewarding_node()
      if goal_node is not None:
        print(f'[SubgoalSampler] *** Switched target node from {expansion_node} -> {goal_node} ***')
        self._expansion_node = goal_node

  def __call__(self, current_node: Tuple) -> OARG:
    if self._sampling_method == 'task' or \
      random.random() < self._task_goal_probability:
        return self._task_goal
    
    if random.random() < self._exploration_goal_probability:
       return self._exploration_goal

    goal_hash = None
    goal_dict = self.get_candidate_goals(current_node)

    if len(goal_dict) == 1:
      goal_hash = list(goal_dict.keys())[0]

    elif current_node in self._abstract_policy:
      goal_hash = self._abstract_policy[current_node]

    if goal_hash is not None and goal_hash in goal_dict:
      return OARG(
        goal_dict[goal_hash][0],
        action=goal_dict[goal_hash][1],
        reward=goal_dict[goal_hash][2],
        goals=np.asarray(goal_hash, dtype=np.int16))
    
    if self._default_behavior == 'graph_search':
      new_policy = self._deterministic_policy(current_node, self._expansion_node)
      self._abstract_policy.update(new_policy)
      print(f'[SubgoalSampler] New path: {self.get_subgoal_sequence(current_node, self._expansion_node)}')
      if current_node in self._abstract_policy:
        goal_hash = self._abstract_policy[current_node]
        print(f'[SubgoalSampler] First step of deterministic policy: {goal_hash}')
        if goal_hash and goal_hash in goal_dict:
          return OARG(
            goal_dict[goal_hash][0],
            action=goal_dict[goal_hash][1],
            reward=goal_dict[goal_hash][2],
            goals=np.asarray(goal_hash, dtype=np.int16))

    print(f'[GoalSampler] Hitting the default case;',
          f'goal_dict size: {len(goal_dict)}, SamplingMethod: {self._sampling_method},',
          f'goal_hash: {goal_hash}, inside: {goal_hash in goal_dict}.')
    return self._exploration_goal if \
      random.random() < self._exploration_goal_probability else self._task_goal
  
  def get_candidate_goals(self, current_node: Tuple) -> Dict:
    """Get the possible goals to pursue at the current state."""
    at_goal = lambda g: all([g1 == g2 for g1, g2 in zip(current_node, g) if g2 >= 0])
    not_special_context = lambda g: g != self._exploration_goal_hash and g != self._task_goal_hash
    return {
      goal: oar for (goal, oar) in self._hash2goal.items()
        if not at_goal(goal) and not_special_context(goal)
    }
  
  def _deterministic_policy(self, start: Tuple, goal: Tuple) -> List[Tuple]:
    t0 = time.time()
    policy = {}
    if self._skill_graph is None:
      self._skill_graph = nx.DiGraph(list(self._edges))
      print(f'[SubgoalSampler] Graph creation time: {time.time() - t0}')
    if start in self._skill_graph and goal in self._skill_graph:
      node2path = nx.single_target_shortest_path(self._skill_graph, target=goal)
      print(f'[SubgoalSampler] Graph search time: {time.time() - t0}')
      t0 = time.time()
      for node, path in node2path.items():
        action = path[1] if len(path) > 1 else path[0]
        policy[node] = action
      print(f'[SubgoalSampler] Policy creation time: {time.time() - t0}')
    return policy
  
  def _sample_rewarding_node(self):
    """If the target node is the task goal, then switch it out for a rewarding node."""
    rewarding_nodes = [node for node, rew in self._reward_dict.items() if rew > 0]
    terminal_rewarding_nodes = [node for node in rewarding_nodes if self._discount_dict[node] == 0]
    if len(terminal_rewarding_nodes) > 0:
      return random.choice(terminal_rewarding_nodes)

  def get_subgoal_sequence(
      self,
      start_node: Tuple,
      goal_node: Tuple,
      max_len: int = 20
  ) -> List[Tuple]:
    """Get the sequence of subgoals from start -> goal."""
    i = 0
    current = start_node
    path = [start_node]
    if current in self._abstract_policy:
      while goal_node not in path and i < max_len:
        current = self._abstract_policy[current]
        path.append(current)
        i += 1
    return path
