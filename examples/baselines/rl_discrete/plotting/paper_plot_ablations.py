import os
import numpy as np
import matplotlib.pyplot as plt
from plotting_utils import get_summary_data
from comparison_plotting_utils import gather_csv_files_from_base_dir

# Base directory (adjust if necessary)
BASE_DIR = os.path.expanduser('~/git-repos/acme/examples/baselines/rl_discrete/local_testing/sweeps/kc')

# Define the two experiment directories.
directories = {
    'IM-DSG': os.path.join(BASE_DIR, 'dsg_kc_thesis1'),
    'No planning': os.path.join(BASE_DIR, 'dsg_without_planning_kc1')
}

avg_returns = {}
error_bars = {}

# Loop over the two directories and gather returns.
for label, dir_path in directories.items():
    id_to_csv = gather_csv_files_from_base_dir(base_dir=dir_path, log_type='actor')
    
    final_returns = []
    for acme_id, csv_path in id_to_csv.items():
        try:
            frames, returns = get_summary_data(csv_path, xkey='actor_steps', ykey='episode_return')
            if len(returns) > 0:
                final_returns.extend(
                    [float(x) for x in returns.values if not isinstance(x, str) or x.replace('.', '', 1).isdigit()]
                )
        except Exception as e:
            print(f"Error processing {acme_id}: {e}")
    
    if final_returns:
        avg_returns[label] = np.mean(final_returns)
        error_bars[label] = np.std(final_returns) / np.sqrt(50)
    else:
        avg_returns[label] = np.nan
        error_bars[label] = np.nan
    print(f"{label}: Average = {avg_returns[label]}, Error = {error_bars[label]}")

# Prepare data for the bar plot.
labels = list(avg_returns.keys())
values = [avg_returns[lbl] for lbl in labels]
errors = [error_bars[lbl] for lbl in labels]

# Increase the figure size.
plt.figure(figsize=(10, 8))

# Create a bar plot with white fill, black edges, and include error bars.
bars = plt.bar(labels, values, yerr=errors, facecolor='white', edgecolor='black',
               error_kw=dict(ecolor='black', capsize=5))

# Apply different hatch patterns to each bar.
patterns = ['//', 'xx']
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

plt.ylabel("Avg Undiscounted Episodic Return")
plt.title("MiniGrid-KeyCorridorS5R3-v0")
plt.ylim(0, max(values)*1.2)

plt.savefig('kc_planning_ablation.png')
plt.close()
