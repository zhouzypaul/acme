
import os
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import MagicMock
import collections
import seaborn as sns

sns.set_theme(style="white", context="poster")

# Mocking for robust unpickling
class RobustUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        mock_modules = [
            'launchpad', 'tensorflow', 'gym', 'dm_env', 
            'dm_tree', 'tree', 'jax', 'haiku', 'rlax', 'optax', 'chex', 'acme'
        ]
        if any(module.startswith(m) for m in mock_modules):
            return MagicMock
        try:
            return super().find_class(module, name)
        except (ImportError, ModuleNotFoundError):
            return MagicMock

def load_counts(file_path):
    try:
        with open(file_path, 'rb') as f:
            unpickler = RobustUnpickler(f)
            data = unpickler.load()
            
        # Try Index 9 (New Counter)
        if len(data) > 9 and isinstance(data[9], dict) and data[9]:
            return list(data[9].values())
            
        # Fallback to Index 7 (Set of Infos)
        if len(data) > 7 and isinstance(data[7], dict):
            return [len(v) for v in data[7].values()]
            
        return [0] # Return 0 if nothing found
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def get_data(pattern):
    base_dir = '/mnt/nfs/home/ademello/research/acme/examples/baselines/rl_discrete/local_testing/dev/montezuma/factored_dev'
    search_path = os.path.join(base_dir, pattern, 'plots', 'plotting_vars.pkl')
    files = glob.glob(search_path)
    print(f"Found {len(files)} files for pattern '{pattern}'")
    
    all_counts = []
    for f in files:
        counts = load_counts(f)
        all_counts.extend(counts)
        print(f"  Loaded {len(counts)} classifiers from {os.path.basename(os.path.dirname(os.path.dirname(f)))}")
        
    return all_counts

def plot_comparison():
    print("Loading Factored Data...")
    factored_counts = get_data('monte_factored_*')
    
    print("\nLoading Standard Data...")
    standard_counts = get_data('monte_standard_*')
    
    # Calculate stats
    f_mean = np.mean(factored_counts) if factored_counts else 0
    f_sem = np.std(factored_counts) / np.sqrt(len(factored_counts)) if factored_counts else 0
    
    s_mean = np.mean(standard_counts) if standard_counts else 0
    s_sem = np.std(standard_counts) / np.sqrt(len(standard_counts)) if standard_counts else 0
    
    print(f"\nResults:")
    print(f"Abstract Subgoals: Mean={f_mean:.2f} (N={len(factored_counts)})")
    print(f"Standard Subgoals: Mean={s_mean:.2f} (N={len(standard_counts)})")
    
    # Plotting
    plt.figure(figsize=(12, 9))
    
    labels = ['Abstract Subgoals', 'Standard Subgoals']
    means = [f_mean, s_mean]
    errors = [f_sem, s_sem]
    
    plt.bar(labels, means, yerr=errors, capsize=10, color=['blue', 'purple'], alpha=0.8)
    
    plt.ylabel('Num Unique States')
    # plt.title('Comparison of Unique States per Classifier') # Removed as per request
    
    save_path = '/mnt/nfs/home/ademello/research/acme/comparison_plot.png'
    plt.savefig(save_path)
    print(f"\nComparison plot saved to {save_path}")
    
    # Plotting just Abstract Subgoals
    plt.close()
    plt.figure(figsize=(8, 8))
    plt.bar(['Abstract Subgoals'], [f_mean], yerr=[f_sem], capsize=10, color='blue', alpha=0.8)
    plt.ylabel('Num Unique States')
    plt.tight_layout()
    
    factored_save_path = '/mnt/nfs/home/ademello/research/acme/factored_unique_states.png'
    plt.savefig(factored_save_path)
    print(f"Abstract Subgoals plot saved to {factored_save_path}")

if __name__ == "__main__":
    plot_comparison()
