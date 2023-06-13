import itertools


class GoalSpaceManager(object):
  """Worker that maintains the skill-graph."""
  def __init__(self) -> None:
    pass

  def run(self):
    for iteration in itertools.count():
      print(f'GSM iteration {iteration}')

