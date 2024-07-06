import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple, Optional, Callable

from acme.salient_event import patch_lib
from acme.salient_event import patch_utils


def create_classifier(
    salient_patches: Dict[Tuple, np.ndarray],
    prototype_image: np.ndarray,
    prototype_info_vector: Tuple,
    classifier_id: Optional[int] = None,
    redundancy_checking_method: str = 'histogram',
    base_plotting_dir: str = ""
) -> Dict:
    plotting_dir = ""
    if base_plotting_dir != "":
        plotting_dir = os.path.join(base_plotting_dir, f'redundancy/classifier_{classifier_id}')
        os.makedirs(os.path.join(base_plotting_dir, "redundancy"), exist_ok=True)
        os.makedirs(plotting_dir, exist_ok=True)

    return {
        "salient_patches": salient_patches,
        "prototype_image": prototype_image,
        "prototype_info_vector": prototype_info_vector,
        "classifier_id": classifier_id,
        "redundancy_checking_method": redundancy_checking_method,
        "plotting_dir": plotting_dir
    }

def classify(classifier: Dict, obs: np.ndarray) -> bool:
    assert classifier["classifier_id"] is not None, 'Assign ID before use.'
    if len(classifier["salient_patches"]) == 0:
        return False
    
    # salient patches are in BGR so we need to convert obs to BGR as well.
    obs = patch_lib.np2cv(obs)

    # TODO(ab): do we need to subtract the background from the obs?
    
    for bb, salient_patch in classifier["salient_patches"].items():
        obs_patch = patch_lib.extract_patch(obs, bb)
        if not patch_lib.is_similar(obs_patch, salient_patch):
            return False
    return True

def assign_id(classifier: Dict, classifier_id: int) -> Dict:
    return {**classifier, "classifier_id": classifier_id}

def equals(classifier1: Dict, classifier2: Dict) -> bool:
    if classifier1["redundancy_checking_method"] == 'template':
        return equals_template_based(classifier1, classifier2)
    return equals_histogram_based(classifier1, classifier2)

def equals_template_based(classifier1: Dict, classifier2: Dict) -> bool:
    if len(classifier1["salient_patches"]) != len(classifier2["salient_patches"]):
        return False
    for bb, patch1 in classifier1["salient_patches"].items():
        if bb not in classifier2["salient_patches"]:
            return False
        if not patch_lib.is_similar(patch1, classifier2["salient_patches"][bb]):
            return False
    print(f'Found a match for classifier {classifier1["classifier_id"]} and {classifier2["classifier_id"]}')
    return True

def equals_histogram_based(classifier1: Dict, classifier2: Dict) -> bool:
    if len(classifier1["salient_patches"]) != len(classifier2["salient_patches"]):
        return False
    
    matches = {}

    for bb1, patch1 in classifier1["salient_patches"].items():
        for bb2, patch2 in classifier2["salient_patches"].items():
            if patch_lib.are_bounding_boxes_similar(bb1, patch1, bb2, patch2):
                matches[bb1] = bb2
                break

    eq = len(matches) == len(classifier1["salient_patches"])

    if eq:
        print(f'Found a match for classifier {classifier1["classifier_id"]} and {classifier2["classifier_id"]}')
        print(f'Matches: {matches}')

        oid2patches1 = {i: box for i, box in enumerate(classifier1["salient_patches"].keys())}
        oid2patches2 = {i: box for i, box in enumerate(classifier2["salient_patches"].keys())}
        
        if classifier1["plotting_dir"] != "":
            plot_data = {
                'classifier1_id': classifier1["classifier_id"],
                'classifier2_id': classifier2["classifier_id"],
                'classifier1_oid2patches': oid2patches1,
                'classifier2_oid2patches': oid2patches2,
                'classifier1_image': classifier1["prototype_image"],
                'classifier2_image': classifier2["prototype_image"],
                'plot_path': os.path.join(classifier1["plotting_dir"], f'match_{classifier1["classifier_id"]}.png')
            }
            # plot_matches(plot_data)

    return eq

def plot_matches(plot_data: Dict):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(
        patch_utils.draw_bounding_boxes(
            plot_data['classifier1_image'].copy(),
            plot_data['classifier1_oid2patches'])
    )
    axs[0].set_title(f'Classifier {plot_data["classifier1_id"]}')
    axs[0].axis('off')
    axs[1].imshow(
        patch_utils.draw_bounding_boxes(
            plot_data['classifier2_image'].copy(),
            plot_data['classifier2_oid2patches'])
    )
    axs[1].set_title(f'Classifier {plot_data["classifier2_id"]}')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(plot_data['plot_path'])
    plt.close()
