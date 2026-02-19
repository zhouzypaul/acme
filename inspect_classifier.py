
import pickle
import sys
import numpy as np

file_path = '/mnt/nfs/home/ademello/research/acme/classifiers/classifiers0_calcThresh_low_subsampled1/classifier_101.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print("Salient Patches Keys:", data['salient_patches'].keys())
        first_key = list(data['salient_patches'].keys())[0]
        print("First Patch Key:", first_key)
        print("First Patch Value:", data['salient_patches'][first_key])
except Exception as e:
    print(f"Error reading pickle file: {e}")
