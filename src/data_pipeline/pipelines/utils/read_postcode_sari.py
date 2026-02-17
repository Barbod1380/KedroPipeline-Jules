import cv2
import numpy as np
from typing import Optional, Union, Tuple
from ultralytics import YOLO
from .CONSTS_sari.CONSTS_sari import (
    POSTCODE_CORRECTION,
    REGION_BASKET_MAP,
    VALID_REGIONS,
)
import pandas as pd


def get_region_if_index_exists(
    df: pd.DataFrame, index_value: int, target_col: str
) -> int:
    """
    Retrieves an integer value from a specified column in a DataFrame based on an index.

    It first attempts to find the `index_value` directly in the DataFrame's index.
    If not found, it then tries to find `index_value // 10` (integer division) in the index.
    If neither index is found, it returns 0.

    Args:
        df (pd.DataFrame): The DataFrame to search within.
        index_value (int): The primary index value to look for.
        target_col (str): The name of the column from which to retrieve the value.

    Returns:
        int: The integer value from the `target_col` at the found index, or 0 if
             no matching index is found after both checks.
    """
    if index_value in df.index:
        return int(df.loc[index_value, target_col])
    elif index_value // 10 in df.index:
        return int(df.loc[index_value // 10, target_col])
    elif index_value // 100 in df.index:
        return int(df.loc[index_value // 100, target_col])
    elif index_value // 1000 in df.index:
        return int(df.loc[index_value // 1000, target_col])
    else:
        return 0
    

def correct_postcode(postcode: str) -> int:
    """
    Corrects a given postcode using a predefined dictionary of corrections.

    The function takes a string representation of a postcode, converts it to an integer,
    and checks if it matches any of the values in the `POSTCODE_CORRECTION_13` dictionary.
    If a match is found, it returns the corresponding corrected postcode (key).
    If no match is found, it defaults to returning the first three digits of the original postcode.

    Args:
        postcode (str): The postcode as a string.

    Returns:
        int: The corrected postcode if found in `POSTCODE_CORRECTION_13`, or
        the first three digits of the original postcode as an integer otherwise.
    """

    postcode_int = int(postcode)
    exception_region = get_region_if_index_exists(
        POSTCODE_CORRECTION, postcode_int, "region"
    )
    if exception_region != 0:
        return exception_region

    postcode_region = int(postcode[:3])
    if postcode_region in REGION_BASKET_MAP.keys():
        postcode_region = REGION_BASKET_MAP[postcode_region]

    return postcode_region


def is_digit_repetition_valid(number_str: str, n: int = 8) -> bool:
    """
    Checks if the most repeated digit in the string occurs more than n times.

    Args:
        number_str (str): The string representation of the number to check.
        n (int): The maximum allowed repetitions for any digit.

    Returns:
        bool: False if any digit repeats more than n times, True otherwise.
    """
    from collections import Counter

    digit_counts = Counter(number_str)
    most_common = digit_counts.most_common(1)
    if most_common and most_common[0][1] > n:
        return False
    return True


def is_number_of_unique_digits_valid(number_str: str, n: int = 3) -> bool:
    """Validates if a numeric string contains more than the specified number of unique digits.

    This function checks whether the input string representation of a number
    has a sufficient diversity of digits by counting distinct characters
    and comparing against a threshold value.

    Args:
        number_str: String representation of the number to validate.
            Should contain only numeric characters (0-9).
        n: Threshold for unique digit count. Defaults to 2.
            The function returns True if unique digits exceed this value.

    Returns:
        bool: True if the number of unique digits in `number_str` is greater than `n`,
            False otherwise.
    """
    if len(set(number_str)) > n:
        return True
    return False


def read_postcode_sari(
    image: np.ndarray, digit_model: YOLO, save_path: Optional[str] = None
) -> int:
    """
    Predicts a 3-digit postcode from an image using a YOLO model.

    Workflow:
    1. Runs the YOLO detection on the provided image with:
       - device='cuda'
       - agnostic_nms=True
       - conf=0.3
    2. If 'save_path' is provided, saves an annotated image (showing YOLO predictions)
       to '{save_path}/read_postcode.jpg'.
    3. Extracts the first and last columns from res[0].obb.data (2D CUDA tensor).
    4. If there are fewer than 3 rows in this tensor, returns -1.
    5. Sorts by the first column and takes the first 3 rows from the last column.
    6. Concatenates those 3 digits into a single integer.
    7. If the resulting integer is not in the range [131..139], returns -1;
       otherwise returns that integer.

    Parameters
    ----------
    image : np.ndarray
        The input image as a NumPy array.
    digit_model : YOLO model
        The YOLO model used for digit detection.
    save_path : str, optional
        Directory path to save the YOLO-annotated image
        (read_postcode.jpg). If None, no image is saved.

    Returns
    -------
    int
        A 3-digit postcode in [131..139], or -1 if unsuccessful.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")

    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    # 1. Run YOLO detection
    try:
        res = digit_model.predict(
            source=image,
            conf=0.25,
            iou=0.5,
            device="cuda",
            agnostic_nms=True,
            verbose=False,
        )
    except Exception as e:
        print(f"Error during YOLO postcode model prediction: {e}")
        return -1

    # Check if we have any detections
    if not res or len(res) == 0 or len(res[0].obb.data) == 0:
        return -1, 'No-Detection', None

    # 3. Extract first and last columns (assuming res[0].obb.data is [N x ...])
    columns = res[0].obb.data[:, [0, -1]]

    # 5. Sort by the first column and select the first 6 rows from the last column
    sorted_columns = columns[columns[:, 0].argsort()][:, -1]
    # 6. Concatenate those three digits into a single integer
    digits_str = str(int("".join(str(int(d.item())) for d in sorted_columns)))

    # some valid postcodes has 9 digits
    if len(digits_str) != 10:
        return -1, 'postcode-length-constraint', digits_str

    if not is_digit_repetition_valid(digits_str):
        return -1, 'digit-repetition-constraint', digits_str
    
    if not is_number_of_unique_digits_valid(digits_str):
        return -1, 'unique-value-constraint', digits_str

    first_sev = digits_str[:7]
    postcode = correct_postcode(first_sev)

    # 2. Save models prediction
    if save_path:
        try:
            annotated_image = res[0].plot()  # Annotate the image with bounding boxes
            cv2.imwrite(f"{save_path}/read_postcode_{postcode}.png", annotated_image)
        except Exception as e:
            print(f"Warning: Failed to save annotated image: {e}")

    # 7. Check that the postcode is in valid range, else return -1
    if postcode in VALID_REGIONS:
        print("WE ARE IN HERE AND POSTCODE IS VALID")
        return postcode, 'valid', digits_str
    else:
        print("WE ARE IN HERE AND POSTCODE IS NOT VALID")
        return -1, 'not-valid-region', digits_str
