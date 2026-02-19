
import pickle
import sys
import numpy as np

file_path = '/mnt/nfs/home/ademello/research/acme/examples/baselines/rl_discrete/local_testing/dev/montezuma/factored_dev/monte_factored_seed42/plots/plotting_vars.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print("Type of data:", type(data))
        
        if isinstance(data, tuple):
            print(f"Tuple length: {len(data)}")
            for i, item in enumerate(data):
                print(f"Item {i}: Type: {type(item)}")
                if isinstance(item, (list, dict, tuple)):
                     print(f"  Length: {len(item)}")

except Exception as e:
    print(f"Error reading pickle file: {e}")
