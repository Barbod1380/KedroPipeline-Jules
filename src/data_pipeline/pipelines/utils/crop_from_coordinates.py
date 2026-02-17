import cv2
import numpy as np
from typing import Tuple
from ultralytics import YOLO


def crop_obb(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Crops and warps a perspective region of an image based on the given coordinates.

    Parameters:
        image (np.ndarray): The input image to be cropped.
        coords (np.ndarray): A numpy array containing four (x, y) coordinates representing the corners of the region to crop.

    Returns:
        np.ndarray: The cropped and perspective-warped region of the image. If the height of the result is greater than its width,
                    it is rotated by 90 degrees counterclockwise for correct orientation.
    """
    src_pts = coords

    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # Sort points to have a consistent order
    src_pts_sorted = src_pts[np.lexsort((src_pts[:, 1], src_pts[:, 0]))]
    two_left = src_pts_sorted[:2]
    two_right = src_pts_sorted[2:]
    two_left = two_left[np.argsort(two_left[:, 1])]
    two_right = two_right[np.argsort(two_right[:, 1])]

    ordered_pts = np.array(
        [two_left[0], two_right[0], two_right[1], two_left[1]], dtype=np.float32
    )
    (tl, tr, br, bl) = ordered_pts
    width_top = distance(tl, tr)
    width_bottom = distance(bl, br)
    max_width = int(max(width_top, width_bottom))

    height_left = distance(tl, bl)
    height_right = distance(tr, br)
    max_height = int(max(height_left, height_right))

    dst_pts = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def crop_hbb(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Crops a rectangular region from an image based on the given bounding box coordinates.

    Parameters:
        image (np.ndarray): The input image to be cropped.
        coords (np.ndarray): A numpy array containing four (x, y) coordinates representing the corners of the bounding box.

    Returns:
        np.ndarray: The cropped region of the image defined by the bounding box.
    """
    x_min = int(np.min(coords[:, 0]))
    x_max = int(np.max(coords[:, 0]))
    y_min = int(np.min(coords[:, 1]))
    y_max = int(np.max(coords[:, 1]))

    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image


def crop_from_coordinates(
    image: np.ndarray,
    coords: np.ndarray,
    mode: str = "obb",
    save_path: str | None = None,
    name: str = "",
) -> np.ndarray:
    """
    Crops and warps a perspective region from an image using provided coordinates.

    Parameters:
        image (np.ndarray): The input image to be cropped. Should be a valid image array.
        coords (np.ndarray): A numpy array containing four (x, y) coordinates representing the corners of the region to crop.
        save_path (str): If passed, saves the cropped image to a file.

    Returns:
        np.ndarray: The cropped and perspective-warped region of the image. Returns None if the input image is None.
    """
    if image is None:
        print("Input image is None")
        return None

    if mode == "obb":
        cropped_image = crop_obb(image, coords)
    elif mode == "hbb":
        cropped_image = crop_hbb(image, coords)
    else:
        raise ValueError("Invalid mode for crop from coordinates. Use 'obb' or 'hbb'.")

    if save_path:
        cv2.imwrite(f"{save_path}/{name}_coordinates_cropped.png", cropped_image)

    return cropped_image



def predict_classification(
    image: np.ndarray,
    cls_model: YOLO,
    conf_thresh: float = 0,
    default_label: str = "clear",
) -> Tuple[float, str]:
    
    cls_result = cls_model.predict(image, device="cuda", verbose=False)[0]

    # Validate and extract results
    if not hasattr(cls_result, "probs") or not hasattr(cls_result, "names"):
        raise ValueError(
            "Classification model must return YOLO Results object with probs and names"
        )

    top_idx = cls_result.probs.top1
    confidence = cls_result.probs.top1conf
    
    if confidence > conf_thresh:
        label = cls_result.names[top_idx]
    else:
        label = default_label
    
    return confidence, label


def pretty_oof_preprocess(image: np.ndarray, save_path: str = None) -> np.ndarray:
    """
    Preprocess an image using a series of transformations to enhance its features.

    This function applies grayscale conversion, normalization, Sobel gradients,
    adaptive thresholding, and rescaling to preprocess the input image.

    Args:
        image (numpy.ndarray): The input image in BGR format (datatype: numpy.ndarray).
        save_path (bool, optional): Whether to save the processed image to a file. Defaults to False (datatype: bool).

    Returns:
        numpy.ndarray: The processed image, rescaled to the 0-255 range (datatype: numpy.ndarray).

    Steps:
    1. Convert the input image to grayscale.
    2. Normalize the grayscale image to the range [0, 255].
    3. Compute Sobel gradients in the x and y directions.
    4. Combine the gradients to calculate the gradient magnitude.
    5. Generate a mean image by blending the gradient magnitude and the original grayscale image.
    6. Compute a histogram of pixel intensities and adjust pixel values based on the most frequent interval.
    7. Clamp positive pixel values in the adjusted image to zero.
    8. Normalize the adjusted image to the range [0, 255].
    9. Apply adaptive Gaussian thresholding to the normalized image.
    10. Rescale the binary image to the 0-255 range.
    11. Optionally save the processed image as "pretty_oof_preprocess.png".

    Note:
    - The input image must be in BGR format as expected by OpenCV.
    - The processed image is returned as an 8-bit single-channel image.
    """
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    gradient_magnitude = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    gradient_magnitude = (1 - gradient_magnitude / gradient_magnitude.max()) * 255
    ALPHA = 0.21
    mean_image = ALPHA * gradient_magnitude + (1 - ALPHA) * image

    mean_image = cv2.normalize(
        mean_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    bin_indices = mean_image // 16
    counts = np.bincount(bin_indices.ravel(), minlength=16)
    max_bin = np.argmax(counts[:16])  # Consider only bins 0-15
    adjustment = max_bin * 16

    adjusted_image = np.minimum(mean_image, adjustment)

    # adjusted_image_normalized = cv2.normalize(adjusted_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # _, otsu_thresh = cv2.threshold(adjusted_image_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    adjusted_image_normalized = cv2.normalize(
        adjusted_image, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(
        adjusted_image_normalized,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        149,  # Block size 27 very important
        21,  # Constant subtracted from the mean 11
    )

    # Rescale to 0-255 using OpenCV
    rescaled_image = cv2.normalize(binary_image, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    if save_path:
        cv2.imwrite(f"{save_path}/pretty_oof_preprocess.png", rescaled_image)

    return rescaled_image



def is_top_left(shape: Tuple[int, int], coords: np.ndarray) -> bool:
    """
    Determines if the center of the given coordinates is in the top-left quadrant of the image.

    Parameters:
        shape (Tuple[int, int]): The shape of the image as (height, width).
        coords (np.ndarray): A numpy array containing coordinates (x, y) to check.

    Returns:
        bool: True if the center of coords is in the top-left quadrant, False otherwise.
    """
    coords_center = np.mean(coords, axis=0)
    image_center = (shape[1] / 2, shape[0] / 2)
    if coords_center[0] < image_center[0] and coords_center[1] < image_center[1]:
        return True
    return False