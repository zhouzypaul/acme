import math
import numpy as np
import collections
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


def plot_spatial_values(hash2value, save_path, split_by_direction: bool):
  
  plt.figure(figsize=(12, 12))
  
  if split_by_direction:
    plot_spatial_values_direction_split(hash2value)
  else:
    xs, ys, values = get_quantity_from_hash_to_counts(hash2value, 'count')
    plt.scatter(xs, ys, c=values, s=40, marker='s')
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


def compute_bonus_prediction_error(true_count_info, approx_bonus_info):
  true_count_info_keys = list(true_count_info.keys())
  total_error = 0
  num_points = 0
  for k in true_count_info_keys:
    if k in approx_bonus_info:
      true_count = true_count_info[k]
      true_bonus = 1 / math.sqrt(true_count)
      approx_bonus = approx_bonus_info[k]
      mse = (true_bonus - approx_bonus) ** 2
      total_error += mse
      num_points += 1
  return total_error / num_points if num_points > 0 else -1


def plot_quantity_over_iteration(
    quantity_over_iteration,
    save_path,
    quantity_name: str = 'MSE'):
  plt.figure(figsize=(12, 12))
  plt.plot(quantity_over_iteration, linewidth=4)
  plt.title(f"{quantity_name} over iteration")
  plt.xlabel("Iteration")
  plt.ylabel(f"{quantity_name}")
  plt.grid()
  plt.savefig(save_path)
  plt.close()


def split_hash_to_bonus_by_hash(hash2bonus):
  """Split hash2bonus into hash_bit_to_mean_bonus,
  which clusters hash2bonus based on bonuses observed when different bits of hash are on."""
  split_directionary = collections.defaultdict(list)

  # This dictionary maps each index in the hash to the value that it activates on.
  # A value of -1 denotes that the index is not used.
  # Missing indices are assumed to correspond to the door and is high at val=1.
  idx_high_vals = collections.defaultdict(int, {
    0: -1, 
    1: -1,
    2: 1,
    -1: 1,
    -2: 1
  })

  for hash, bonus in hash2bonus.items():
    for i in range(len(hash)):

      # Ignore the hash bits that are -1 in the index dictionary.
      if idx_high_vals[i] == -1:
        continue

      if hash[i] == idx_high_vals[i]:
        split_directionary[i].append(bonus)

  for i in split_directionary:
    split_directionary[i] = (np.mean(split_directionary[i]),
                             np.var(split_directionary[i]),
                             len(split_directionary[i]))

  return split_directionary


def plot_average_bonus_for_each_hash_bit(hash2bonus, save_path):
  """Plot the average bonus for each hash bit."""
  hash_bit_to_mean_bonus = split_hash_to_bonus_by_hash(hash2bonus)
  hash_bit_to_mean_bonus = sorted(hash_bit_to_mean_bonus.items(), key=lambda x: x[0])
  hash_bits = [x[0] for x in hash_bit_to_mean_bonus]
  mean_bonuses = [x[1][0] for x in hash_bit_to_mean_bonus]
  std_bonuses = [x[1][1] for x in hash_bit_to_mean_bonus]
  num_occurences = [x[1][2] for x in hash_bit_to_mean_bonus]
  plt.figure(figsize=(14, 14))
  plt.subplot(121)
  plt.bar(hash_bits, mean_bonuses, yerr=std_bonuses)
  plt.title("Average bonus for each hash bit")
  plt.xlabel("Hash bit")
  plt.ylabel("Average bonus")
  plt.grid()
  plt.subplot(122)
  plt.bar(hash_bits, num_occurences)
  plt.title("Number of occurences for each hash bit")
  plt.xlabel("Hash bit")
  plt.ylabel("Number of occurences")
  plt.grid()
  plt.savefig(save_path)
  plt.close()
