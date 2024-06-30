"""Given a trajectory, the most novel obs in that traj and a novelty func,
return a classifier that only compares to the most salient factors in the most
novel obs."""


import os
import ipdb
import random
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple, Callable, Optional
import uuid

import acme.salient_event.patch_lib as patch_lib
import acme.salient_event.patch_utils as patch_utils


class SalientEventClassifier:
  def __init__(
      self,
      salient_patches: Dict[patch_lib.BoundingBox, np.ndarray],
      prototype_image: np.ndarray,
      classifier_id: Optional[int] = None,
    ):
    self.salient_patches = salient_patches
    self.classifier_id = classifier_id
    self.prototype_image = prototype_image  # RGB image (not BGR)

  def __call__(self, obs: np.ndarray) -> bool:
    assert self.classifier_id is not None, 'Assign ID before use.'
    if len(self.salient_patches) == 0:
      return False
    
    # salient patches are in BGR so we need to convert obs to BGR as well.
    obs = patch_lib.np2cv(obs)

    # TODO(ab): do we need to subtract the background from the obs?
    
    for bb in self.salient_patches:
      salient_patch = self.salient_patches[bb]
      obs_patch = patch_lib.extract_patch(obs, bb)
      if not patch_lib.is_similar(obs_patch, salient_patch):
        return False
    return True
  
  def assign_id(self, classifier_id: int):
    self.classifier_id = classifier_id
  
  def equals(self, other: 'SalientEventClassifier') -> bool:
    if len(self.salient_patches) != len(other.salient_patches):
      return False
    for bb in self.salient_patches:
      if bb not in other.salient_patches:
        return False
      if not patch_lib.is_similar(
        self.salient_patches[bb], other.salient_patches[bb]):
        return False
    return True



class SalientEventClassifierGenerator:
  def __init__(
      self,
      obs_traj: List[np.ndarray],
      hash_traj: List[Tuple],
      full_obs_traj: List[np.ndarray],
      most_novel_obs: np.ndarray,
      most_novel_hash: Tuple,
      novelty_func: Callable,
      save_dir: str,
      plot_intermediate_results: bool = True
  ):
    self._input_hash_traj = hash_traj
    self._most_novel_obs = patch_lib.np2cv(most_novel_obs.copy())
    self._input_obs_traj = [patch_lib.np2cv(obs.copy()) for obs in obs_traj]
    self._most_novel_hash = most_novel_hash

    self._novelty_fn = novelty_func

    self._save_dir = save_dir
    self._plot_intermediate_results = plot_intermediate_results

    self._debug_info = {
      'most_novel_obs': most_novel_obs.copy()
    }

  def generate_salient_patches(self, ref_img: np.ndarray, ref_hash: Tuple) -> Tuple[Dict, Dict]:
    # Step 1. Find the reference image
    ref_img = patch_lib.np2cv(ref_img)
    
    # Step 2. Find the bounding boxes in the reference image.
    ref_img_bboxes, most_novel_img_bboxes = self._find_bboxes(
      ref_img,
      self._input_obs_traj,
    )

    # Step 4. Find the most salient/important patches in the most novel image.
    oid2bboxes = self._find_salient_patches(
      ref_img,
      ref_img_bboxes,
      self._most_novel_obs,
      most_novel_img_bboxes
    )

    # Step 5. Return a dict that maps bboxes to patches.
    salient_patches = {}

    for oid in oid2bboxes:
      bbox = oid2bboxes[oid]
      if bbox != (0, 0, 1, 1):
        patch = patch_lib.extract_patch(self._most_novel_obs, bbox)
        print(f'Found bbox {bbox} shape: {patch.shape}')
        salient_patches[bbox] = patch

    # Step 6. Filter out bboxes that are too small.
    # Filter out bounding boxes that are too small
    # Each bbox is stored as tuple (x, y, width, height)
    n_grid_tiles_per_dim = 13
    tile_size = ref_img.shape[0] // n_grid_tiles_per_dim
    bbox_size_threshold = 0.25 * tile_size
    cond = lambda bbox: bbox[2] > bbox_size_threshold and bbox[3] > bbox_size_threshold
    salient_patches = {bbox: patch for bbox, patch in salient_patches.items() if cond(bbox)}
    most_novel_img_bboxes = {oid: bbox for oid, bbox in most_novel_img_bboxes.items() if cond(bbox)}
    ref_img_bboxes = {oid: bbox for oid, bbox in ref_img_bboxes.items() if cond(bbox)}

    self._debug_info['ref_img_with_bboxes'] = patch_lib.draw_bounding_boxes(
      ref_img.copy(), ref_img_bboxes)
    self._debug_info['most_novel_img_with_bboxes'] = patch_lib.draw_bounding_boxes(
      self._most_novel_obs.copy(), most_novel_img_bboxes)
    
    if self._plot_intermediate_results:
      self.visualize()

    return salient_patches, most_novel_img_bboxes, ref_img_bboxes

  def _find_salient_patches(
    self,
    ref_img: np.ndarray,
    ref_img_bboxes: Dict[int, Tuple[int, int, int, int]],
    most_novel_img: np.ndarray,
    most_novel_img_bboxes: Dict[int, Tuple[int, int, int, int]],
  ) -> Dict[Tuple[int, int, int, int], np.ndarray]:
    oid2counterfactuals = patch_lib.generate_counterfactuals(
      most_novel_img,
      ref_img,
      ref_img_bboxes,
      most_novel_img_bboxes,
      save_dir=self._save_dir)
    print(f'Generated {len(oid2counterfactuals)} counterfactuals')
    salient_patches, debug_info = patch_lib.find_relevant_bboxes(
      self._most_novel_obs,
      ref_img_bboxes,
      most_novel_img_bboxes,
      oid2counterfactuals,
      self._novelty_fn,
      debug_info=self._debug_info)
    
    self._debug_info.update(debug_info)

    return salient_patches

    
  def _find_bboxes(
    self,
    ref_img: np.ndarray,
    obs_traj: List[np.ndarray]
  ):
    background_img = patch_lib.learn_background(obs_traj)
    ref_foreground_img = patch_lib.subtract_background([ref_img], background_img)[0]
    novel_img_fg = patch_lib.subtract_background([self._most_novel_obs], background_img)[0]
    
    ref_img_bboxes = patch_lib.find_objects(ref_foreground_img)
    new_bboxes = patch_lib.track_objects_in_img(
      ref_foreground_img, ref_img_bboxes, novel_img_fg)
    
    self._debug_info['background_img'] = background_img
    self._debug_info['ref_foreground_img'] = ref_foreground_img
    self._debug_info['novel_foreground_img'] = novel_img_fg
      
    return ref_img_bboxes, new_bboxes
  
  def visualize(self):
    """visualize the self._debug_info."""
    is_title_key = lambda key: 'novelty' in key or 'relevant' in key
    image_keys = [key for key in self._debug_info if not is_title_key(key)]
    
    num_images = len(image_keys)
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    
    for idx, key in enumerate(image_keys):
      row = idx // num_cols
      col = idx % num_cols
      axs[row, col].imshow(self._debug_info[key])  # Assuming the images are in a compatible format
      axs[row, col].set_title(key)
      axs[row, col].axis('off')
      
      # If the key is a counterfactual image, add the corresponding float values as title/annotation
      if 'counterfactual' in key:
        oid, i = key.split('_')[1], key.split('_')[-1]
        novelty = self._debug_info.get(f'oid_{oid}_counterfactual_{i}_novelty', 'N/A')
        novelty_drop = self._debug_info.get(f'oid_{oid}_counterfactual_{i}_novelty_drop', 'N/A')
        relevant = self._debug_info.get(f'oid_{oid}_counterfactual_{i}_relevant', 'N/A')
        
        axs[row, col].set_title(
          f"{key} Novelty: {novelty:.2f}\nNovelty Drop: {novelty_drop:.2f}\nRelevance: {relevant}")

    plt.tight_layout()
    plt.savefig(os.path.join(self._save_dir, f'debug_info_{uuid.uuid4()}.png'))
    plt.close()
