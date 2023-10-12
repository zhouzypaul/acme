import math
import matplotlib.pyplot as plt


def get_quantity_from_hash_to_counts(hash2counts, quantity):
  x = []; y = []; z = []
  for obs_hash, count in hash2counts.items():
    x.append(obs_hash[0])
    y.append(obs_hash[1])
    if quantity == 'count':
      z.append(count)
    else:
      z.append(1. / math.sqrt(count))
  return x, y, z


def get_quantity_from_hash_to_bonus(hash2bonus, quantity):
  x = []; y = []; z = []
  for obs_hash, bonus in hash2bonus.items():
    x.append(obs_hash[0])
    y.append(obs_hash[1])
    if quantity == 'bonus':
      z.append(bonus)
    else:
      # z.append(1. / math.sqrt(bonus))
      z.append(1. / (bonus ** 2 + 1e-10))
  return x, y, z


def plot_spatial_count_or_bonus(true_count_info, approx_bonus_info, save_path, quantity='bonus'):
  assert quantity in ('bonus', 'count'), quantity

  x, y, z = get_quantity_from_hash_to_counts(true_count_info, quantity)
  
  plt.figure(figsize=(24, 12))
  plt.subplot(121)
  plt.scatter(x, y, c=z, s=400, marker='s')
  plt.colorbar()
  plt.title(f'Ground truth {quantity}')

  x, y, z = get_quantity_from_hash_to_bonus(approx_bonus_info, quantity)

  plt.subplot(122)
  plt.scatter(x, y, c=z, s=400, marker='s')
  plt.colorbar()
  plt.title(f'Approx {quantity}')
  plt.savefig(save_path)
  plt.close()

def plot_true_vs_approx_bonus(true_count_info, approx_bonus_info, save_path):

  true_bonuses = []
  approx_bonuses = []

  true_count_info_keys = list(true_count_info.keys())
  for k in true_count_info_keys:
    if k in approx_bonus_info:
      true_bonuses.append(1 / math.sqrt(true_count_info[k]))
      approx_bonuses.append(approx_bonus_info[k])

  plt.figure(figsize=(12, 12))
  plt.scatter(true_bonuses, approx_bonuses, alpha=0.3)
  plt.title("True Vs Approx Bonus")
  max_y = max(max(true_bonuses), max(approx_bonuses))
  plt.xlim((0, min(1, max_y) + 0.1))
  plt.ylim((0, max_y + 0.1))
  plt.xlabel('True Bonus')
  plt.ylabel('Approx Bonus')
  plt.savefig(save_path)
  plt.close()


def split_by_agent_dir(hash2quantity):
  split_directionary = {}
  for hash, quantity in hash2quantity.items():
    direction = hash[2]
    if direction not in split_directionary:
      split_directionary[direction] = {}
    else:
      position = hash[0], hash[1]
      split_directionary[direction][position] = quantity
  return split_directionary


def plot_spatial_values_direction_split(hash2value):
  direction_to_pos_to_val = split_by_agent_dir(hash2value)
  for i, (direction, pos2val) in enumerate(direction_to_pos_to_val.items()):
    xs, ys, values = get_quantity_from_hash_to_counts(pos2val, 'count')
    plt.subplot(2, 2, i + 1)
    plt.scatter(xs, ys, c=values, s=40, marker='s')
    plt.colorbar()
    plt.title(f'Direction: {direction}')


def plot_spatial_values(hash2value, save_path, split_by_direction: bool,
                        use_discrete_colorbar: bool = False):
  
  plt.figure(figsize=(12, 12))
  cmap = 'tab10' if use_discrete_colorbar else 'viridis'
  
  if split_by_direction:
    plot_spatial_values_direction_split(hash2value)
  else:
    xs, ys, values = get_quantity_from_hash_to_counts(hash2value, 'count')
    plt.scatter(xs, ys, c=values, s=40, marker='s', cmap=cmap)
    plt.colorbar()

  plt.savefig(save_path)
  plt.close()


def plot_histogram(values, save_path,
                   title="", xlabel="", ylabel="",
                   bins=100):
  plt.figure(figsize=(12, 12))
  plt.hist(values, bins=bins)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.savefig(save_path)
  plt.close()
