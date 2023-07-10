from collections import defaultdict


def defaultify(d):
  """Convert a nested dictionary to a defaultdict."""
  dd = defaultdict(lambda: defaultdict(lambda: 1))
  for key1 in d:
    dd[key1] = defaultdict(lambda: 1, d[key1])
  return dd


if __name__ == '__main__':
  dictionary = {1: {1: 2, 2: 3}, 2: {1: 1}}
  print(defaultify(dictionary))
  print(defaultify(dictionary)[1][2])
  print(defaultify(dictionary)[1][3])
  print(defaultify(dictionary)[3])
