import os
import sys
import dm_env
import dataclasses
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Any
from collections import defaultdict

from acme.wrappers.oar_goal import OARG


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
  intrinsic_reward: float = 0.


def truncation(ts: dm_env.TimeStep) -> dm_env.TimeStep:
  discount = np.array(1, dtype=np.float32)
  step_type = dm_env.StepType.LAST
  
  return ts._replace(discount=discount, step_type=step_type)


def termination(ts: dm_env.TimeStep) -> dm_env.TimeStep:
  discount = np.array(0, dtype=np.float32)
  step_type = dm_env.StepType.LAST
  
  return ts._replace(discount=discount, step_type=step_type)

  
def continuation(ts: dm_env.TimeStep) -> dm_env.TimeStep:
  discount = np.array(1, dtype=np.float32)
  step_type = dm_env.StepType.MID
  
  return ts._replace(discount=discount, step_type=step_type)


def scores2probabilities(scores: np.ndarray) -> np.ndarray:
  min_score = np.min(scores)
  if min_score < 0:
    scores -= min_score  # Make the minimum score zero
  score_sum = scores.sum()
  if score_sum == 0:
    return np.ones_like(scores) / len(scores)
  probabilities = scores / score_sum
  if probabilities.sum() < 1:
    assert probabilities.sum() > 0.99, (probabilities, probabilities.sum())
    renormalization_factor = 1.0 / probabilities.sum()
    probabilities *= renormalization_factor
  return probabilities


def remove_duplicates_keep_last(lst):
  seen = set()
  # Start with an empty list where we'll store our elements
  # in reverse order (i.e., last occurrence first).
  last_occurrences = []
  # Iterate over the list in reverse order.
  for element in reversed(lst):
    # If the element hasn't been seen yet, add it to our
    # last_occurrences list and mark it as seen.
    if element not in seen:
      last_occurrences.append(element)
      seen.add(element)
  # Reverse to get the correct order back and return.
  return last_occurrences[::-1]


def create_log_dir(experiment_name):
  path = os.path.join(os.getcwd(), experiment_name)
  try:
      os.makedirs(path, exist_ok=True)
  except OSError:
      pass
  else:
      print("Successfully created the directory %s " % path)
  return path


def debug_visualize_trajectory(
    trajectory: List[GoalBasedTransition], folder_name: str):
  
  def dump_oarg(filename: str, oarg: OARG, pursued_goal: Tuple):
    plt.subplot(121)
    plt.imshow((oarg.observation[:,:,:3] * 255).astype(np.uint8))
    plt.title(f'R={oarg.reward} G={tuple(oarg.goals)}')
    plt.subplot(122)
    plt.imshow((oarg.observation[:,:,3:] * 255).astype(np.uint8))
    plt.title(f'R={oarg.reward} G={pursued_goal}')
    plt.savefig(f'plots/{folder_name}/{filename}.png')

  for i, transition in enumerate(trajectory):
    obs = transition.next_ts.observation
    dump_oarg(f'state_{i}', obs, transition.pursued_goal)

def get_size_mb(obj: Any) -> float:
  """Recursively calculate the size of an object and its contents in megabytes."""
  seen = set()  # To keep track of already counted objects

  def inner(obj):
    obj_id = id(obj)
    if obj_id in seen:
      return 0
    seen.add(obj_id)
    
    if isinstance(obj, np.ndarray):
      return obj.nbytes
    
    size = sys.getsizeof(obj)
    
    if isinstance(obj, (str, bytes, int, float)):
      pass  # These are simple objects, no need to look inside them
    elif isinstance(obj, (tuple, list, set, frozenset)):
      size += sum(inner(item) for item in obj)
    elif isinstance(obj, dict):
      size += sum(inner(key) + inner(value) for key, value in obj.items())
    elif hasattr(obj, '__dict__'):
      size += inner(obj.__dict__)
    
    return size

  return inner(obj) / (1024 * 1024)  # Convert bytes to MB


def print_data_structure_sizes(**kwargs):
  """Print the sizes of multiple data structures."""
  total_size = 0
  for name, obj in kwargs.items():
      size = get_size_mb(obj)
      total_size += size
      print(f"Size of {name}: {size:.2f} MB")
  print(f"Total size of all structures: {total_size:.2f} MB")


if __name__ == '__main__':
  dictionary = {1: {1: 2, 2: 3}, 2: {1: 1}}
  print(defaultify(dictionary))
  print(defaultify(dictionary)[1][2])
  print(defaultify(dictionary)[1][3])
  print(defaultify(dictionary)[3])
