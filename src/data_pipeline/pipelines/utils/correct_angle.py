import os
import cv2
import logging
import numpy as np
from enum import Enum
from ultralytics import YOLO


class RotationAngle(Enum):
    ANGLE_0 = "0"
    ANGLE_90 = "90"
    ANGLE_180 = "180"
    ANGLE_270 = "270"



def predict_angle_to_rotate(
    image: np.ndarray,
    model: YOLO,
    default_value: str = "0",
    device: str = "cuda",
    verbose: bool = False,
) -> str:
    """Predicts the rotation angle needed for proper image orientation.

    Args:
        image: Input image as numpy array in BGR format
        model: Pretrained YOLO classification model
        default_value: Default angle to return if prediction fails
        device: Device to run inference on ('cpu' or 'cuda')
        verbose: Whether to show prediction details

    Returns:
        Predicted angle as string ('0', '90', '180', or '270')

    Raises:
        ValueError: For invalid input image
        RuntimeError: For prediction failures
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a valid numpy array")

    try:
        padded_image = pad_to_square(image)
        results = model.predict(padded_image, device=device, verbose=verbose)
        if not results:
            logging.warning("No YOLO results returned")
            return default_value

        if hasattr(results[0], "probs"):
            return results[0].names[results[0].probs.top1]

        logging.warning("No probabilities in YOLO results")
        return default_value
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise RuntimeError("Angle prediction failed") from e



def rotate_image(
    image: np.ndarray, angle: str, save_path: str | None = None
) -> np.ndarray:
    """Rotates image according to specified angle.

    Args:
        image: Input image to rotate
        angle: Rotation angle ('0', '90', '180', or '270')
        save_path: Optional path to save rotated image

    Returns:
        Rotated image as numpy array

    Raises:
        ValueError: For invalid inputs
        IOError: If image saving fails
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a valid numpy array")

    valid_angles = {a.value for a in RotationAngle}
    if angle not in valid_angles:
        raise ValueError(f"Invalid angle. Must be one of {valid_angles}")

    try:
        if angle == RotationAngle.ANGLE_0.value:
            rotated = image
        elif angle == RotationAngle.ANGLE_90.value:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == RotationAngle.ANGLE_180.value:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == RotationAngle.ANGLE_270.value:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_path = os.path.join(save_path, f"rotated_image.png")
            if not cv2.imwrite(save_path, rotated):
                raise IOError(f"Failed to save image to {save_path}")

        return rotated
    except cv2.error as e:
        raise ValueError(f"Image processing error: {str(e)}") from e


def pad_to_square(image: np.ndarray) -> np.ndarray:
    """
    Pads the given image with white borders so that the output image is square.

    If the image is taller than wide, pads the left and right sides.
    If the image is wider than tall, pads the top and bottom.

    Parameters:
        image (np.ndarray): The input image in BGR format.

    Returns:
        np.ndarray: The square image with added white padding.
    """
    height, width = image.shape[:2]

    # If already square, return the original image.
    if height == width:
        return image

    # Determine the required padding based on image dimensions.
    if height > width:
        diff: int = height - width
        left_pad: int = diff // 2
        right_pad: int = diff - left_pad
        square_image: np.ndarray = cv2.copyMakeBorder(
            image, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
    else:
        diff: int = width - height
        top_pad: int = diff // 2
        bottom_pad: int = diff - top_pad
        square_image: np.ndarray = cv2.copyMakeBorder(
            image, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

    return square_image