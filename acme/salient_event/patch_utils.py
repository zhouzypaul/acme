import os
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from typing import List


COLOR_CHANNEL = 2


def traj2images(traj):
    """Converts a trajectory to a list of images.

    Args:
      traj (List[dict]): Trajectory

    Returns:
      List[np.ndarray]: List of images
    """
    return [x[1] for x in traj]


def color2gray(image: np.ndarray):
    """Converts an RGB image to grayscale.

    Args:
      image (np.ndarray): RGB image

    Returns:
      np.ndarray: Grayscale image
    """
    if image.shape[COLOR_CHANNEL] == 1:
        return image.squeeze(COLOR_CHANNEL)
    assert image.shape[COLOR_CHANNEL] == 3, image.shape
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def gray2color(image: np.ndarray, shift=(0, 0, 0)):
    """Converts an RGB image to grayscale.

    Args:
      image (np.ndarray): RGB image

    Returns:
      np.ndarray: Grayscale image
    """
    if image.shape[COLOR_CHANNEL] == 3:
        return image
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for i in range(len(shift)):
        image[:, :, i] = image[:, :, i] + shift[i]
    image = np.clip(image, 0, 255)

    assert image.shape[COLOR_CHANNEL] == 3, image.shape
    return image


def cv2np(obs):
    return cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)


def save_images_to_disk(images: List[np.ndarray], out_dir: str):
    """Saves a list of images to disk.

    Args:
      images (List[np.ndarray]): List of images
      out_dir (str): Output directory
    """
    os.makedirs(out_dir, exist_ok=True)
    for i, image in enumerate(images):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=COLOR_CHANNEL)
        cv2.imwrite(f"{out_dir}/{i}.png", image)


def save_images_as_a_gif(images: List[np.ndarray], out_file: str, duration=0.02):
    """Saves a list of images as a gif.

    Args:
      images (List[np.ndarray]): List of images
      out_file (str): Output file
    """
    with imageio.get_writer(out_file, mode="I", duration=duration) as writer:
        for image in images:
            writer.append_data(image)


def create_gif_with_bounding_boxes(image_tuples, save_path):
    """
    Draws bounding boxes on images and creates a GIF from these images.

    :param image_tuples: List of tuples, where each tuple contains:
                          - An image as a NumPy array of shape (H, W)
                          - A bounding box as a tuple (x, y, w, h)
    :param save_path: Path where the GIF should be saved.
    """
    imgs = []

    for image, (x, y, w, h) in image_tuples:
        # Convert grayscale image to BGR for color drawing
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw the bounding box
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box

        # Ensure that the image is in RGB format
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Add to list
        imgs.append(Image.fromarray(image_rgb))

    # Create GIF
    imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=50, loop=0)


def draw_bounding_boxes(image, bboxes, color=(255, 255, 255)):
    """Draws bounding boxes on an image.

    Args:
      image (np.ndarray): Image
      bboxes_and_ids (List[Tuple[int, int, int, int, int]]): List of bounding boxes and ids

    Returns:
      np.ndarray: Image with bounding boxes
    """
    for object_id, bbox in bboxes.items():
        c = color if isinstance(color, tuple) else color[object_id]
        if bbox is None:
            continue
        (x, y, w, h) = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), c, 1)
        cv2.putText(
            image,
            str(object_id),
            (x, y - 5),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.4,  # font scale
            c,  # color
            1,  # thickness
        )
    return image


def draw_image_index(image, idx, total, pos=(2, 15)):
    cv2.putText(
        image,
        f"{idx}/{total}",
        pos,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        0.4,  # font scale
        (255, 255, 255),  # color
        1,  # thickness
    )
    return image


def draw_debug_string(image, debug, pos=(2, 76)):
    cv2.putText(
        image,
        str(debug),
        pos,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        0.4,  # font scale
        (255, 255, 255),  # color
        1,  # thickness
    )
    return image


def save_img(img, path):
    img = img.copy()
    plt.imshow(img)
    plt.savefig(path)
    plt.close()


def save_img_with_bboxes(img, bboxes, path):
    img = img.copy()
    color = (255, 255, 255) if img.dtype == np.uint8 else (1, 1, 1)
    img = draw_bounding_boxes(img, bboxes, color=color)
    save_img(img, path)
