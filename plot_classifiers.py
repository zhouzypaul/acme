import os
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_classifier(file_path, output_dir, save_id=None):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        classifier_id = data.get('classifier_id', 'unknown')
        if classifier_id == 'unknown':
            # Fallback if id not in dict, try to get from filename
            basename = os.path.basename(file_path)
            # expected format classifier_ID.pkl
            try:
                classifier_id = int(basename.split('_')[1].split('.')[0])
            except:
                pass
        else:
             classifier_id = int(classifier_id)

        img = data['prototype_image']
        patches_data = data['salient_patches']
        
        # Create figure without frame
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(img, aspect='auto', cmap='gray')
        plt.close(fig)
        
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        
        # Add label - REMOVED per request
        # ax.text(0.05, 0.95, f"{classifier_id}", transform=ax.transAxes, 
        #         color='white', fontsize=12, verticalalignment='top', 
        #         bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        
        for key, patch_content in patches_data.items():
            # key is (col, row, width, height) as confirmed
            col, row, width, height = key
            
            # Create a Rectangle patch
            rect = patches.Rectangle((col, row), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
        # Remove axis labels and ticks
        ax.set_axis_off()
        
        # Use save_id if provided, otherwise classifier_id
        final_id = save_id if save_id is not None else classifier_id
        output_path = os.path.join(output_dir, f'classifier_{final_id}.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # print(f"Saved plot for classifier {classifier_id} at {output_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    base_dir = '/mnt/nfs/home/ademello/research/acme/classifiers/classifiers0_calcThresh_low_subsampled1'
    output_dir = os.path.join(base_dir, 'plots')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(base_dir, 'id_mapping.pkl'), 'rb') as f:
        id_map = pickle.load(f)
    
    # id_map is {mapped_id: original_id}
    print(f"Loaded {len(id_map)} IDs from mapping.")
    
    count = 0
    for mapped_index, original_id in id_map.items():
        file_path = os.path.join(base_dir, f'classifier_{original_id}.pkl')
        if os.path.exists(file_path):
            plot_classifier(file_path, output_dir, save_id=mapped_index)
            count += 1
            if count % 50 == 0:
                print(f"Processed {count} files...")
        else:
            print(f"Warning: File not found for ID {original_id}")
            
    print("Done.")

if __name__ == "__main__":
    main()
