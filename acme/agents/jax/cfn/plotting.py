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
  plt.scatter(true_bonuses, approx_bonuses)
  plt.title("True Vs Approx Bonus")
  plt.savefig(save_path)
  plt.close()
