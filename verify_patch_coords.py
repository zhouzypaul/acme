
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

file_path = '/mnt/nfs/home/ademello/research/acme/classifiers/classifiers0_calcThresh_low_subsampled1/classifier_101.pkl'

def verify_coords():
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    img = data['prototype_image']
    patches = data['salient_patches']
    
    # Try to match patches
    # Key format hypothesis: (r, c, h, w) or (x, y, w, h)
    # We established key[2] is 21 (width?) and key[3] is 2 (height?) for value shape (2, 21).
    # So key is likely (x, y, w, h) or (r, c, w, h).
    
    for key, patch in patches.items():
        # key = (k0, k1, k2, k3)
        # patch shape = (h, w) = (2, 21)
        # k2=21, k3=2 -> so k2=w, k3=h
        
        k0, k1, w, h = key
        # Try Interpretation 1: k0=row, k1=col
        # slice: img[k0:k0+h, k1:k1+w]
        sub_img_1 = img[k0:k0+h, k1:k1+w]
        
        # Try Interpretation 2: k0=col, k1=row
        # slice: img[k1:k1+h, k0:k0+w]
        sub_img_2 = img[k1:k1+h, k0:k0+w]
        
        match_1 = False
        match_2 = False
        
        if sub_img_1.shape == patch.shape:
            if np.allclose(sub_img_1, patch):
                match_1 = True
                
        if sub_img_2.shape == patch.shape:
             if np.allclose(sub_img_2, patch):
                match_2 = True
                
        print(f"Key: {key}, Patch Shape: {patch.shape}")
        print(f"Interpretation 1 (r, c, w, h): Match={match_1}")
        print(f"Interpretation 2 (c, r, w, h): Match={match_2}")
        
        if match_1:
            print("Confirmed: (row, col, width, height)")
            return "rcwh"
        if match_2:
            print("Confirmed: (col, row, width, height)")
            return "crwh"
            
verify_coords()
