import os
import cv2
import numpy as np



def denoise_normalize(image: np.ndarray, save_path: str = None) -> np.ndarray:
    """
    Normalize an image to [0, 255] using cv2.normalize().

    Args:
        image (np.ndarray): Input image (BGR, grayscale, or any dtype).

    Returns:
        np.ndarray: Normalized uint8 image [0, 255].

    Raises:
        ValueError: If input is not a NumPy array.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Normalize to 0-255 range using cv2
    normalized = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    denoised = cv2.fastNlMeansDenoising(normalized, h=10, searchWindowSize=7)
    denoised_normalized = cv2.normalize(
        denoised, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    if save_path:
        cv2.imwrite(f"{save_path}/blurred_image.png", denoised_normalized)
    return denoised_normalized



def apply_adaptive_mean_threshold(
    image: np.ndarray,
    save_path: str | None = None,
) -> np.ndarray:
    """
    Apply adaptive mean thresholding to a grayscale image.

    Adaptive mean thresholding calculates the threshold for each pixel as the
    mean of the neighborhood area minus a constant C. This method is effective
    for images with varying lighting conditions.

    Args:
        image (np.ndarray): Input grayscale image as numpy array.
                           Should be 2D for grayscale or 3D (will be converted).
        save_path (Optional[str], optional): Path to save the thresholded image.
                                           Defaults to None.
        timing (bool, optional): Whether to print execution timing information.
                               Defaults to False.

    Returns:
        np.ndarray: Binary thresholded image (0 and 255 values).

    Raises:
        ValueError: If input image is not a valid numpy array or has invalid dimensions.
        TypeError: If image data type is not supported.

    Note:
        - Block size: 21x21 pixels (must be odd)
        - C parameter: 2 (constant subtracted from the mean)
        - Method: ADAPTIVE_THRESH_MEAN_C
    """

    # Input validation
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array")

    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be 2D (grayscale) or 3D (color)")

    # Convert to grayscale if needed
    if image.ndim == 3:
        if image.shape[2] == 3:  # RGB
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:  # RGBA
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError("Color image must have 3 (RGB) or 4 (RGBA) channels")
    else:
        gray_image = image.copy()

    # Ensure image is in proper format for OpenCV
    if gray_image.dtype != np.uint8:
        if gray_image.max() <= 1.0:
            gray_image = (gray_image * 255).astype(np.uint8)
        else:
            gray_image = gray_image.astype(np.uint8)

    try:
        # Apply adaptive mean thresholding with specified parameters
        cv2.normalize(gray_image, gray_image, 0, 255, cv2.NORM_MINMAX)
        # block_size=21, C=2
        image_mean = np.mean(gray_image)
        if image_mean < 80:
            blockSize = 45
            C = 8
        elif image_mean < 160:
            blockSize = 23
            C = 13
        elif image_mean < 185:
            blockSize = 25
            C = 10
        else:
            blockSize = 25
            C = 16
        binary_image = cv2.adaptiveThreshold(
            gray_image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=blockSize,
            C=C,
        )

    except Exception as e:
        raise RuntimeError(f"Error during adaptive mean thresholding: {str(e)}")

    # Save image if path provided
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_path = os.path.join(save_path, "adaptive_mean_receiver.png")
            if not cv2.imwrite(save_path, binary_image):
                raise IOError(f"Failed to save image to {save_path}")
        except Exception as e:
            print(f"Warning: Could not save image to {save_path}. Error: {str(e)}")

    # Print timing information

    return binary_image