import os
import cv2
import time
import collections
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple
import uuid

from acme.salient_event.patch_utils import draw_bounding_boxes

# Dictionary that maps object_id to (x, y, width, height)
BoundingBox = Dict[int, Tuple[int, int, int, int]]


def learn_background(images):
    """Iterate over all the images in the trajectory and learn a background."""
    t0 = time.time()
    detector = cv2.createBackgroundSubtractorKNN(history=1000, detectShadows=False)
    
    unique_images = []
    seen_hashes = set()

    for image in images:
        image_hash = hash(image.tobytes())
        if image_hash not in seen_hashes:
            seen_hashes.add(image_hash)
            unique_images.append(image)
    
    for image in unique_images:
        detector.apply(image)

    print(f'[SalientEventLibrary] Learned background model from {len(unique_images)}',
          f'unique images and {len(images)} total images (dt={time.time() - t0}).')
        
    return detector.getBackgroundImage()


def subtract_background(images, background):
    """Apply the learned background model to the images in the trajectory."""
    return [cv2.subtract(image, background) for image in images]


def np2cv(obs):
    return cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)


def opencv_to_imageio(images):
    return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]


def find_objects(image):
    # Create bounding boxes for each object in the image
    assert image.dtype == np.uint8
    low = 30 if image.dtype == np.uint8 else 0.1
    high = 255 if image.dtype == np.uint8 else 1.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, low, high, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_SIMPLE
    bboxes = {i: cv2.boundingRect(c) for i, c in enumerate(contours)}
    print(f'[SalientEventLibrary] Found {len(bboxes)} objects using thresholds {low} and {high}')
    return bboxes


def extract_patches(image, bboxes) -> dict:
    """Given the bounding boxes, extract the patches from the image."""
    return {oid: image[y:y+h, x:x+w] for oid, (x, y, w, h) in bboxes.items()}


def rotate_patch(patch) -> list:
    """Rotate the patch 4 times and return the list of patches."""
    return [patch,
            cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(patch, cv2.ROTATE_180),
            cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)]


def augment_with_rotations(id2patch):
    return {oid: rotate_patch(patch) for oid, patch in id2patch.items()}


def find_patch_with_highest_match(new_image, patches, threshold) -> int:
    """Given the new image and the patches, find the patch with the highest match."""
    max_val = -1
    best_patch = None
    patch_point = None
    for patch in patches:
        res = cv2.matchTemplate(new_image, patch, cv2.TM_CCOEFF_NORMED)
        if res is not None:
            _, val, _, point = cv2.minMaxLoc(res)
            if val > max_val and val > threshold:
                max_val = val
                best_patch = patch
                patch_point = point
    return (best_patch, patch_point) if best_patch is not None else None


def extract_patch(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]


def is_similar(patch1, patch2, threshold=0.8):
    res = cv2.matchTemplate(patch1, patch2, cv2.TM_CCOEFF_NORMED)
    _, val, _, _ = cv2.minMaxLoc(res)
    return val > threshold


def extract_objects(new_image, id2patches, threshold=0.4):
    new_bboxes = {}
    for oid in id2patches:
        patches = id2patches[oid]
        patch_and_point = find_patch_with_highest_match(new_image, patches, threshold)
        if patch_and_point:
            patch, point = patch_and_point
            patch_width, patch_height = patch.shape[:2]
            # new_bboxes[oid] = (point[0], point[1], patch_height, patch_width)
            new_bboxes[oid] = (point[0], point[1], patch_width, patch_height)
    return new_bboxes


def track_objects_in_img(ref_img, ref_img_bboxes, img):
    id2patch = extract_patches(ref_img, ref_img_bboxes)
    id2patches = augment_with_rotations(id2patch)
    new_bboxes = extract_objects(img, id2patches)
    return new_bboxes


def bb_disappearance_counterfactual(
    old_bbox: BoundingBox,
    old_obs: np.ndarray,
    new_obs: np.ndarray,
    margin: int = 1,
    save_dir: str = ""
) -> np.ndarray:
    counterfactual = new_obs.copy()
    x, y, w, h = old_bbox
    counterfactual[y-margin:y+h+margin, x-margin:x+w+margin] = old_obs[
    y-margin:y+h+margin, x-margin:x+w+margin]

    if save_dir:
        plt.subplot(1, 3, 1)
        plt.imshow(draw_bounding_boxes(old_obs.copy(), {0: old_bbox}))
        plt.title('Original Image')
        plt.subplot(1, 3, 2)
        plt.imshow(draw_bounding_boxes(new_obs.copy(), {0: old_bbox}))
        plt.title('New Image')
        plt.subplot(1, 3, 3)
        plt.imshow(draw_bounding_boxes(counterfactual.copy(), {0: old_bbox}))
        plt.title('Counterfactual Image')
        plt.savefig(f'{save_dir}/disappearance_counterfactual_{uuid.uuid4()}.png')
        plt.close()

    return counterfactual


def bb_motion_counterfactual(
    old_bbox: BoundingBox,
    new_bbox: BoundingBox,
    old_obs: np.ndarray,
    new_obs: np.ndarray,
    margin: int = 1,
    save_dir: str = ""
) -> np.ndarray:
    counterfactual = new_obs.copy()
    x, y, w, h = old_bbox
    xnew, ynew, wnew, hnew = new_bbox
    counterfactual[y-margin:y+h+margin, x-margin:x+w+margin] = old_obs[
    y-margin:y+h+margin, x-margin:x+w+margin]
    counterfactual[ynew-margin:ynew+hnew+margin, xnew-margin:xnew+wnew+margin] = old_obs[
    ynew-margin:ynew+hnew+margin, xnew-margin:xnew+wnew+margin]

    if save_dir:
        plt.subplot(1, 3, 1)
        plt.imshow(draw_bounding_boxes(old_obs.copy(), {0: old_bbox}))
        plt.title('Original Image')
        plt.subplot(1, 3, 2)
        plt.imshow(draw_bounding_boxes(new_obs.copy(), {1: new_bbox}))
        plt.title('New Image')
        plt.subplot(1, 3, 3)
        plt.imshow(draw_bounding_boxes(counterfactual.copy(), {0: old_bbox, 1: new_bbox}))
        plt.title('Counterfactual Image')
        plt.savefig(f'{save_dir}/motion_counterfactual_{uuid.uuid4()}.png')
        plt.close()

    return counterfactual


def generate_counterfactuals(
    most_novel_obs: np.ndarray,
    reference_image: np.ndarray,
    reference_img_bboxes: List[BoundingBox],
    most_novel_img_bboxes: List[BoundingBox],
    margin: int = 1,
    save_dir: str = ""
) -> Dict[int, List[np.ndarray]]:
    oid2counterfactuals = collections.defaultdict(list)
    for object_id in reference_img_bboxes:
        if object_id not in most_novel_img_bboxes:
            print(f'Object {object_id} disappeared')
            counterfactual = bb_disappearance_counterfactual(
                old_bbox=reference_img_bboxes[object_id],
                old_obs=reference_image,
                new_obs=most_novel_obs,
                margin=margin,
                save_dir=save_dir)
            oid2counterfactuals[object_id].append(counterfactual)

        elif reference_img_bboxes[object_id] != most_novel_img_bboxes[object_id]:
            print(f'Object {object_id} moved')
            counterfactual = bb_motion_counterfactual(
                old_bbox=reference_img_bboxes[object_id],
                new_bbox=most_novel_img_bboxes[object_id],
                old_obs=reference_image,
                new_obs=most_novel_obs,
                margin=margin,
                save_dir=save_dir)
            oid2counterfactuals[object_id].append(counterfactual)

    return oid2counterfactuals


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err


def get_most_simiar_obs(obs, hash2obs, distance_fn=mse):
    """Find the most similar obs in hash2obs using MSE."""
    min_val = +np.inf
    most_similar_obs = None
    most_similar_hash = None
    for h, template in hash2obs.items():
        err = distance_fn(obs, template)
        if err < min_val:
            min_val = err
            most_similar_obs = template
            most_similar_hash = h
    print(f'Most similar hash: {most_similar_hash} with mse: {min_val}')
    return most_similar_obs, most_similar_hash


def find_relevant_bboxes(
    novel_obs: np.ndarray,
    old_img_bboxes: BoundingBox,
    novel_img_bboxes: BoundingBox,
    oid2counterfactuals: Dict[int, List[np.ndarray]],
    novelty_fn: callable,  # maps hash to [0, 1]
    drop_threshold: float = 0.1,
    debug_info: dict = {},
):
    """Iterate over the oids, delete the ones that don't lead to a drop in novelty."""
    relevant_bboxes = {}
    salient_novelty = novelty_fn(novel_obs)
    os.makedirs('results/counterfactuals', exist_ok=True)
    for oid in oid2counterfactuals:
        counterfactuals = oid2counterfactuals[oid]
        for i, counterfactual in enumerate(counterfactuals):
            # Counterfactual means that oid was reverted to boring state and 
            # everything else in the image was allowed to change.
            new_novelty = novelty_fn(counterfactual)
            novelty_drop = salient_novelty - new_novelty
            print(f'Undoing the change in {oid} led to a novelty drop of {novelty_drop} ',
                  f'from {salient_novelty} to {new_novelty} ')
            
            if novelty_drop > drop_threshold:
                relevant_bboxes[oid] = novel_img_bboxes[oid] if oid in \
                      novel_img_bboxes else old_img_bboxes[oid]
            
            if debug_info:
                debug_info[f'oid_{oid}_counterfactual_{i}'] = counterfactual
                debug_info[f'oid_{oid}_counterfactual_{i}_novelty'] = new_novelty.item()
                debug_info[f'oid_{oid}_counterfactual_{i}_novelty_drop'] = novelty_drop.item()
                debug_info[f'oid_{oid}_counterfactual_{i}_relevant'] = (novelty_drop > drop_threshold).item()

    return relevant_bboxes, debug_info
