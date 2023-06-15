import itertools
import threading
import collections
import dm_env

from typing import Dict
from acme.wrappers.oar_goal import OARG


class GoalSpaceManager(object):
  """Worker that maintains the skill-graph."""
  def __init__(self) -> None:
    self._hash2obs = {}  # map goal hash to obs
    self._hash2counts = collections.defaultdict(int)
    self._count_dict_lock = threading.Lock()
  
  def get_count_dict(self):
    return self._hash2counts
  
  def get_goal_dict(self):
    return self._hash2obs
    
  def update(self, hash2obs: Dict, hash2count: Dict):
    """Update based on goals achieved by the different actors."""
    self._hash2obs.update(hash2obs)
    self._update_count_dict(hash2count)
    
  def _update_count_dict(self, hash2count: Dict):
    with self._count_dict_lock:
      for goal in hash2count:
        self._hash2counts[goal] += hash2count[goal]

  def run(self):
    for iteration in itertools.count():
      if iteration % 10 == 0:
        print(self._hash2counts)
