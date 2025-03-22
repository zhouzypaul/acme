import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set("talk", "white")

# ---------------
# Load and process returns data for IM-DSG and HER (for actor_steps >= threshold)
# ---------------
base_dir = "local_testing/sweeps/pinball"
algorithms = {
    "IM-DSG": {
        "dirs": [
            "dsg_large_pinball_expansioncfnrf1_copy",
            "dsg_large_pinball_expansioncfnrf1_copy2",
            "dsg_large_pinball_expansioncfnrf1_copy3",
        ],
        "subfolder": "dsg_big_pinball_bigger_and_longer_sweep1_3__seed_1__optiontimeout_500__goalspacesize_10"
    },
    "HER": {
        "parent_dir": "dsg_large_pinball_pure_her_no_planning_full_state_space_as_goal_space1",
        "subfolders": [
            "dsg_big_pinball_pure_her6_copy3",
            "dsg_big_pinball_pure_her6_copy2"
        ]
    }
}

# Compute returns for frames >= 50M (Eval Returns)
algorithm_returns = {}
for algo, config in algorithms.items():
    returns_group = []
    if "dirs" in config:
        subfolder = config.get("subfolder", "")
        for d in config["dirs"]:
            exp_path = os.path.join(base_dir, d)
            base_path = os.path.join(exp_path, subfolder) if subfolder else exp_path
            csv_path = os.path.join(base_path, "logs", "actor", "logs.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(base_path, "logs", "logs.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["actor_steps"] = pd.to_numeric(df["actor_steps"], errors="coerce")
                df["episode_return"] = pd.to_numeric(df["episode_return"], errors="coerce")
                filtered_df = df[(df["actor_steps"] >= 50000000) & (df["episode_return"].notna())]
                returns_group.extend(filtered_df["episode_return"].tolist())
            else:
                print(f"CSV file not found: {csv_path}")
    elif "parent_dir" in config:
        parent_path = os.path.join(base_dir, config["parent_dir"])
        for sub in config["subfolders"]:
            exp_path = os.path.join(parent_path, sub)
            base_path = exp_path  # Logs are directly under each subfolder.
            csv_path = os.path.join(base_path, "logs", "actor", "logs.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(base_path, "logs", "logs.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["actor_steps"] = pd.to_numeric(df["actor_steps"], errors="coerce")
                df["episode_return"] = pd.to_numeric(df["episode_return"], errors="coerce")
                filtered_df = df[(df["actor_steps"] >= 45000000) & (df["episode_return"].notna())]
                returns_group.extend(filtered_df["episode_return"].tolist())
            else:
                print(f"CSV file not found: {csv_path}")
    else:
        print(f"No valid configuration for {algo}")
    
    algorithm_returns[algo] = returns_group

# Compute average and standard error for returns (>=50M)
algos_to_plot = ["IM-DSG", "HER"]
avg_values = []
stderr_values = []
for algo in algos_to_plot:
    returns = algorithm_returns.get(algo, [])
    if returns:
        avg = np.mean(returns)
        std_val = np.std(returns, ddof=1)
        stderr = std_val / np.sqrt(len(returns))
        avg_values.append(avg)
        stderr_values.append(stderr)
    else:
        avg_values.append(0)
        stderr_values.append(0)

# ---------------
# Compute returns for frames < 50M ("Train Returns")
# ---------------
algorithm_returns_early = {}
for algo, config in algorithms.items():
    returns_group = []
    if "dirs" in config:
        subfolder = config.get("subfolder", "")
        for d in config["dirs"]:
            exp_path = os.path.join(base_dir, d)
            base_path = os.path.join(exp_path, subfolder) if subfolder else exp_path
            csv_path = os.path.join(base_path, "logs", "actor", "logs.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(base_path, "logs", "logs.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["actor_steps"] = pd.to_numeric(df["actor_steps"], errors="coerce")
                df["episode_return"] = pd.to_numeric(df["episode_return"], errors="coerce")
                filtered_df = df[(df["actor_steps"] < 50000000) & (df["episode_return"].notna())]
                returns_group.extend(filtered_df["episode_return"].tolist())
            else:
                print(f"CSV file not found: {csv_path}")
    elif "parent_dir" in config:
        parent_path = os.path.join(base_dir, config["parent_dir"])
        for sub in config["subfolders"]:
            exp_path = os.path.join(parent_path, sub)
            base_path = exp_path
            csv_path = os.path.join(base_path, "logs", "actor", "logs.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(base_path, "logs", "logs.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["actor_steps"] = pd.to_numeric(df["actor_steps"], errors="coerce")
                df["episode_return"] = pd.to_numeric(df["episode_return"], errors="coerce")
                filtered_df = df[(df["actor_steps"] < 45000000) & (df["episode_return"].notna())]
                returns_group.extend(filtered_df["episode_return"].tolist())
            else:
                print(f"CSV file not found: {csv_path}")
    else:
        print(f"No valid configuration for {algo}")
    
    algorithm_returns_early[algo] = returns_group

# Compute average and standard error for early returns (<50M)
early_avg_values = []
early_stderr_values = []
for algo in algos_to_plot:
    returns = algorithm_returns_early.get(algo, [])
    if returns:
        avg = np.mean(returns)
        std_val = np.std(returns, ddof=1)
        stderr = std_val / np.sqrt(len(returns))
        early_avg_values.append(avg)
        early_stderr_values.append(stderr)
    else:
        early_avg_values.append(0)
        early_stderr_values.append(0)

# ---------------
# Load and compute coverage data (fractional, 0-1)
# ---------------
def compute_free_space_coverage(points, grid_resolution=100):
    """
    Given a list of (x, y) tuples (each measured to 2 decimal places),
    compute the fraction of free space visited.
    
    Assumptions:
      - The free-space is ~59.5% of the unit square.
      - The unit square is discretized into grid_resolution x grid_resolution cells.
    """
    total_cells = grid_resolution * grid_resolution
    estimated_free_cells = 0.595 * total_cells
    visited_cells = set()
    for x, y in points:
        # Coordinates already map to 0-99.
        i = min(int(x), grid_resolution - 1)
        j = min(int(y), grid_resolution - 1)
        visited_cells.add((i, j))
    return len(visited_cells) / estimated_free_cells

with open('imdsg_pinball_hash2obs_graph.pkl', 'rb') as f:
    dsg_points = list(pickle.load(f).keys())
with open('pure_her_pinball_graph_hash2obs.pkl', 'rb') as f:
    her_points = list(pickle.load(f).keys())

dsg_coverage = compute_free_space_coverage(dsg_points)
her_coverage = compute_free_space_coverage(her_points)
coverage_values = [dsg_coverage, her_coverage]

# Placeholder error for coverage bars (e.g., 0.01 and 0.03)
coverage_errors = [0.01, 0.03]

# ---------------
# Create a combined grouped bar plot with dual y-axes.
# We have 3 groups:
#   Group 0: Eval Returns (≥50M)
#   Group 1: Train Returns (<50M)
#   Group 2: Coverage (fraction, 0-1)
#
# In each group, we have 2 bars (IM-DSG and HER).
# The return groups use ax1 (left y-axis) and the coverage group uses ax2 (right y-axis).
# ---------------
n_groups = 3
group_labels = ["Eval Returns", "Train Returns", "Coverage"]

# Define a dictionary of hatch styles by algorithm
hatch_styles = {
    "IM-DSG": "///",
    "HER": "xxx"
}

# Bar width
bar_width = 0.35
# Group centers along x-axis
group_centers = np.arange(n_groups)

# x positions for each algorithm within each group:
x_positions = {grp: [group_centers[grp] - bar_width/2, group_centers[grp] + bar_width/2]
               for grp in range(n_groups)}

fig, ax1 = plt.subplots(figsize=(10,6))
ax2 = ax1.twinx()

# Plot Eval Returns on group 0 using ax1
for i, algo in enumerate(algos_to_plot):
    ax1.bar(x_positions[0][i], avg_values[i], bar_width, yerr=stderr_values[i],
            capsize=5, color='white', edgecolor='black', hatch=hatch_styles[algo],
            label=algo)  # Only add label once per algorithm.
    
# Plot Train Returns on group 1 using ax1
for i, algo in enumerate(algos_to_plot):
    ax1.bar(x_positions[1][i], early_avg_values[i], bar_width, yerr=early_stderr_values[i],
            capsize=5, color='white', edgecolor='black', hatch=hatch_styles[algo],
            label="" )  # No label here, already provided.

# Plot Coverage on group 2 using ax2
for i, algo in enumerate(algos_to_plot):
    ax2.bar(x_positions[2][i],
            coverage_values[i],
            bar_width,
            yerr=coverage_errors[i],
            capsize=5,
            color='white',
            edgecolor='black',
            hatch=hatch_styles[algo],
            label="")

# Set x-ticks at the group centers and label them
ax1.set_xticks(group_centers)
ax1.set_xticklabels(group_labels)
ax1.set_title("IM-DSG & HER: Returns and Coverage Comparison")
ax1.set_ylabel("Avg Undiscounted Episodic Return")
ax2.set_ylabel("Fraction of Goal Space Covered")

# Set y-limits
ax1.set_ylim(0, max(max(avg_values), max(early_avg_values)) + 0.1)
ax2.set_ylim(0, max(max(avg_values), max(early_avg_values)) + 0.1)

# Create a legend using the bars from group 0 (which have labels)
ax1.legend(loc='best')

plt.tight_layout()
plt.savefig('pinball_eval_barplot_combined.png')
plt.close()
