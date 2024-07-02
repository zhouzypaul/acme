"""Given a trajectory, the most novel obs in that traj and a novelty func,
return a classifier that only compares to the most salient factors in the most
novel obs."""


import os
import ipdb
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple, Callable, Optional

import acme.salient_event.patch_lib as patch_lib
import acme.salient_event.patch_utils as patch_utils


class SalientEventClassifier:
  def __init__(
      self,
      salient_patches: Dict[patch_lib.BoundingBox, np.ndarray],
      prototype_image: np.ndarray,
      classifier_id: Optional[int] = None,
      redundancy_checking_method: str = 'histogram',
      base_plotting_dir: str = ""
    ):
    self.salient_patches = salient_patches
    self.classifier_id = classifier_id
    self.prototype_image = prototype_image  # RGB image (not BGR)
    self.redundancy_checking_method = redundancy_checking_method
    self.plotting_dir = ""

    if base_plotting_dir != "":
      self.plotting_dir = os.path.join(base_plotting_dir, f'redundancy/classifier_{classifier_id}')
      os.makedirs(os.path.join(base_plotting_dir, "redundancy"), exist_ok=True)
      os.makedirs(self.plotting_dir, exist_ok=True)

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
    if self.redundancy_checking_method == 'template':
      return self.equals_template_based(other)
    return self.equals_histogram_based(other)

  def equals_template_based(self, other: 'SalientEventClassifier') -> bool:
    if len(self.salient_patches) != len(other.salient_patches):
      return False
    for bb in self.salient_patches:
      if bb not in other.salient_patches:
        return False
      if not patch_lib.is_similar(
        self.salient_patches[bb], other.salient_patches[bb]):
        return False
    print(f'Found a match for classifier {self.classifier_id} and {other.classifier_id}')
    return True

  def equals_histogram_based(self, other: 'SalientEventClassifier') -> bool:

    if len(self.salient_patches) != len(other.salient_patches):
      return False
    
    matches = {}

    for bb1 in self.salient_patches:
      for bb2 in other.salient_patches:
      
        if patch_lib.are_bounding_boxes_similar(
          bb1,
          self.salient_patches[bb1],
          bb2,
          other.salient_patches[bb2]
        ):
          matches[bb1] = bb2
          break

    eq = len(matches) == len(self.salient_patches)

    if eq:
      print(f'Found a match for classifier {self.classifier_id} and {other.classifier_id}')
      print(f'Matches: {matches}')
      
      # Visualize the salient patches from both classifiers.
      if self.plotting_dir != "":
        oid2patches = {i: box for i, box in enumerate(self.salient_patches.keys())}
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(patch_utils.draw_bounding_boxes(self.prototype_image.copy(), oid2patches))
        axs[0].set_title(f'Classifier {self.classifier_id}')
        axs[0].axis('off')
        oid2patches = {i: box for i, box in enumerate(other.salient_patches.keys())}
        axs[1].imshow(patch_utils.draw_bounding_boxes(other.prototype_image.copy(), oid2patches))
        axs[1].set_title(f'Classifier {other.classifier_id}')
        axs[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plotting_dir, f'match_{self.classifier_id}.png'))
        plt.close()

    return eq


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
    self._input_obs_traj = [patch_lib.np2cv(obs.copy()) for obs in full_obs_traj]
    self._most_novel_hash = most_novel_hash

    self._novelty_fn = novelty_func

    self._save_dir = save_dir
    self._reverse_mode_save_dir = save_dir + '_reverse'
    self._plot_intermediate_results = plot_intermediate_results

    self._debug_info = {
      'most_novel_obs': most_novel_obs.copy()
    }

    os.makedirs(self._save_dir, exist_ok=True)
    os.makedirs(self._reverse_mode_save_dir, exist_ok=True)

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
      most_novel_img_bboxes,
      save_dir=self._save_dir
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

  def generate_appearance_salient_patches(self, ref_img: np.ndarray, ref_hash: Tuple) -> Tuple[Dict, Dict]:
    
    # Step 1. Find the reference image
    ref_img = patch_lib.np2cv(ref_img)
    
    # Step 2. Find the bounding boxes in the novel image.
    novel_img_bboxes, ref_img_bboxes = self._find_bboxes_switched(
      ref_img,
      self._most_novel_obs,
      self._input_obs_traj,
    )

    # Step 3. find the most salient/important patches in the reference image.
    oid2bboxes = self._find_salient_patches_switched(
      ref_img,
      ref_img_bboxes,
      self._most_novel_obs,
      novel_img_bboxes,
      save_dir=self._reverse_mode_save_dir
    )

    # Step 4. Return a dict that maps bboxes to patches.
    salient_patches = {}

    for oid in oid2bboxes:
      bbox = oid2bboxes[oid]
      if bbox[2] > 2 and bbox[3] > 2:  # this is filtering small bboxes.
        patch = patch_lib.extract_patch(ref_img, bbox)
        print(f'[Appearance] Found bbox {bbox} shape: {patch.shape}')
        salient_patches[bbox] = patch

    self._debug_info['appearance_ref_img_with_bboxes'] = patch_lib.draw_bounding_boxes(
      ref_img.copy(), ref_img_bboxes)
    self._debug_info['appearance_novel_img_with_bboxes'] = patch_lib.draw_bounding_boxes(
      self._most_novel_obs.copy(), novel_img_bboxes)

    if self._plot_intermediate_results:
      self.visualize(reverse_mode=True)

    return salient_patches, novel_img_bboxes, ref_img_bboxes

  def _find_salient_patches(
    self,
    ref_img: np.ndarray,
    ref_img_bboxes: Dict[int, Tuple[int, int, int, int]],
    most_novel_img: np.ndarray,
    most_novel_img_bboxes: Dict[int, Tuple[int, int, int, int]],
    save_dir: str = ""
  ) -> Dict[Tuple[int, int, int, int], np.ndarray]:
    oid2counterfactuals = patch_lib.generate_counterfactuals(
      most_novel_img,
      ref_img,
      ref_img_bboxes,
      most_novel_img_bboxes,
      save_dir=save_dir)
    print(f'Generated {len(oid2counterfactuals)} counterfactuals')
    salient_patches, debug_info = patch_lib.find_relevant_bboxes(
      most_novel_img,
      ref_img_bboxes,
      most_novel_img_bboxes,
      oid2counterfactuals,
      self._novelty_fn,
      debug_info=self._debug_info)
    
    self._debug_info.update(debug_info)

    return salient_patches

  def _compute_background(
    self,
    obs_traj: List[np.ndarray],
    method: str,
  ):
    if method == 'background_model':
      return patch_lib.learn_background(obs_traj)
    elif method == 'change_mask':
      return patch_lib.compute_change_mask(obs_traj)
    raise NotImplementedError(f'Unknown method: {method}')

  def _remove_background(
    self,
    image,
    background_img,
    method: str,
  ):
    if method == 'background_model':
      return patch_lib.subtract_background([image], background_img)[0]
    elif method == 'change_mask':
      return patch_lib.apply_mask_to_image(image, background_img)
    raise NotImplementedError(f'Unknown method: {method}')

  def _find_bboxes(
    self,
    ref_img: np.ndarray,
    obs_traj: List[np.ndarray],
    background_method: str = 'change_mask'
  ):
    background_img = self._compute_background(obs_traj, method=background_method)
    ref_foreground_img = self._remove_background(ref_img, background_img, background_method)
    novel_img_fg = self._remove_background(self._most_novel_obs, background_img, background_method)
    
    ref_img_bboxes = patch_lib.find_objects(ref_foreground_img)
    new_bboxes = patch_lib.track_objects_in_img(
      ref_foreground_img, ref_img_bboxes, novel_img_fg)
    
    self._debug_info['background_img'] = background_img
    self._debug_info['ref_foreground_img'] = ref_foreground_img
    self._debug_info['novel_foreground_img'] = novel_img_fg
      
    return ref_img_bboxes, new_bboxes

  def _find_bboxes_switched(
    self,
    ref_img: np.ndarray,
    novel_img: np.ndarray,
    obs_traj: List[np.ndarray],
    background_method: str = 'change_mask'
  ):
    background_img = self._compute_background(obs_traj, method=background_method)
    ref_foreground_img = self._remove_background(ref_img, background_img, background_method)
    novel_img_fg = self._remove_background(self._most_novel_obs, background_img, background_method)
    
    novel_img_bboxes = patch_lib.find_objects(novel_img_fg)
    bboxes_in_ref_img = patch_lib.track_objects_in_img(
      novel_img_fg, novel_img_bboxes, ref_foreground_img)
    
    self._debug_info['appearance_background_img'] = background_img
    self._debug_info['appearance_ref_foreground_img'] = ref_foreground_img
    self._debug_info['appearance_novel_foreground_img'] = novel_img_fg
      
    return novel_img_bboxes, bboxes_in_ref_img
  
  def _find_salient_patches_switched(
    self,
    ref_img: np.ndarray,
    ref_img_bboxes: Dict[int, Tuple[int, int, int, int]],
    most_novel_img: np.ndarray,
    most_novel_img_bboxes: Dict[int, Tuple[int, int, int, int]],
    save_dir: str = ""
  ) -> Dict[Tuple[int, int, int, int], np.ndarray]:
    oid2counterfactuals = patch_lib.generate_app_counterfactuals(
      most_novel_img,
      ref_img,
      ref_img_bboxes,
      most_novel_img_bboxes,
      save_dir=save_dir
    )
    print(f'[Appearance] Generated {len(oid2counterfactuals)} counterfactuals')
    salient_patches, debug_info = patch_lib.find_relevant_bboxes(
      ref_img,
      most_novel_img_bboxes,
      ref_img_bboxes,
      oid2counterfactuals,
      self._novelty_fn,
      debug_info=self._debug_info,
      reverse_mode=True)
    
    self._debug_info.update(debug_info)

    return salient_patches
  
  def visualize(self, reverse_mode: bool = False):
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
        oid, i = key.split('_')[3], key.split('_')[-1]
        novelty = self._debug_info.get(f'reverse_{reverse_mode}_oid_{oid}_counterfactual_{i}_novelty', 'N/A')
        novelty_drop = self._debug_info.get(f'reverse_{reverse_mode}_oid_{oid}_counterfactual_{i}_novelty_drop', 'N/A')
        relevant = self._debug_info.get(f'reverse_{reverse_mode}_oid_{oid}_counterfactual_{i}_relevant', 'N/A')

        if novelty != 'N/A':
          novelty = round(novelty, 3)

        if novelty_drop != 'N/A':
          novelty_drop = round(novelty_drop, 3)
        
        axs[row, col].set_title(
          f"{key} Novelty: {novelty}\nNovelty Drop: {novelty_drop}\nRelevance: {relevant}")

    plt.tight_layout()
    dirpath = self._reverse_mode_save_dir if reverse_mode else self._save_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    plt.savefig(os.path.join(dirpath, f'debug_info_{timestamp}.png'))
    plt.close()
