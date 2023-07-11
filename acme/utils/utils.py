import dm_env
import dataclasses

from typing import Tuple

from collections import defaultdict


def defaultify(d):
  """Convert a nested dictionary to a defaultdict."""
  dd = defaultdict(lambda: defaultdict(lambda: 1))
  for key1 in d:
    dd[key1] = defaultdict(lambda: 1, d[key1])
  return dd


@dataclasses.dataclass
class GoalBasedTransition:
  """Struct for keeping track of transitions during an episode."""
  ts: dm_env.TimeStep
  action: int
  reward: float  # extrinsic reward
  discount: float  # whether next_ts is a terminal state
  next_ts: dm_env.TimeStep
  pursued_goal: Tuple


if __name__ == '__main__':
  dictionary = {1: {1: 2, 2: 3}, 2: {1: 1}}
  print(defaultify(dictionary))
  print(defaultify(dictionary)[1][2])
  print(defaultify(dictionary)[1][3])
  print(defaultify(dictionary)[3])
