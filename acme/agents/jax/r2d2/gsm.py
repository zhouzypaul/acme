import itertools
import threading
import collections
import dm_env

from typing import List, Tuple
from acme.wrappers.oar_goal import OARG


class GoalSpaceManager(object):
  """Worker that maintains the skill-graph."""
  def __init__(self) -> None:
    self._goals = set()
    # self._hash2counts = collections.defaultdict(int)
    # self._count_dict_lock = threading.Lock()
    
  def update(self, trajectory: List[Tuple]):
    """Update based on goals achieved by the different actors."""
    for goal in trajectory:
      self._goals.add(goal)

  def run(self):
    for iteration in itertools.count():
      if iteration % 10 == 0:
        print(len(self._goals))
