import os
import torch
import numpy as np
import Levenshtein

import cv2
from ultralytics import YOLO
from dataclasses import dataclass
from paddleocr import TextRecognition
from typing import Any, Set, List, Tuple
from ..utils.crop_from_coordinates import crop_from_coordinates
from .CONSTS.CONSTS import CORRECT_WORDS_SET, VALID_SPELL_CORRECTION_DICT


@dataclass
class Prediction:
    """
    Data class representing a single prediction.

    Attributes:
        bbox (Tuple[float, float, float, float]): The bounding box as (x_min, y_min, x_max, y_max).
        center (Tuple[float, float]): The center coordinates of the bounding box.
        text (str): The corrected text label associated with the prediction.
        raw_text (str): The raw text label before spelling correction.
    """

    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    text: str
    raw_text: str


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


def process_prediction(result: Any, cls_model: YOLO) -> Any:
    """
    Process a YOLO prediction by cropping detected regions, classifying the crops,
    and saving the predicted labels in the result object.

    The function expects the following attributes in the `result` object:
        - orig_img: np.ndarray containing the original image.
        - names: dict mapping class indices to label strings.
        - boxes: an object that contains detection information including:
            - xyxyxyxy: torch.Tensor of shape (N, 4, 2) representing detection coordinates.

    The classification model (cls_model) must be callable with a numpy array image and return
    either an integer class index or a dict containing a key "class_idx".

    Parameters
    ----------
    result : Any
        Prediction result object from the YOLO model.
    cls_model : YOLO
        A YOLO-CLS model that takes a 3-channel image and returns a predicted class index.

    Returns
    -------
    Any
        The updated prediction result object with a new attribute `boxes.cls_labels`
        that holds a list of predicted labels (strings) for each detection.
    """
    # Ensure the original image is available.
    if not hasattr(result, "orig_img"):
        raise AttributeError("Result object must have an 'orig_img' attribute.")
    orig_img: np.ndarray = result.orig_img
    # print(f"result: {result}")
    # Ensure the boxes attribute and required detection coordinates exist.
    if not hasattr(result, "boxes"):
        raise AttributeError(
            "Result object must have an 'boxes' attribute with detection coordinates."
        )
    boxes = result.boxes

    if not hasattr(boxes, "xyxy"):
        raise AttributeError(
            "boxes object must have a 'xyxy' attribute containing detection coordinates."
        )

    coords_tensor: torch.Tensor = boxes.xyxy.reshape(
        (-1, 2, 2)
    )  # Expected shape: (N, 2, 2)
    
    num_detections: int = coords_tensor.shape[0]
    cls_labels = []
    cls_labels_raw = []
    cls_conf = []

    for i in range(num_detections):
        word_digit = result.names[int(boxes.cls[i])]
        if word_digit == "حروف":
            # Extract coordinates for the i-th detection.
            coords = coords_tensor[i]
            # Convert tensor to a numpy array.
            coords_np = (
                coords.cpu().numpy() if hasattr(coords, "cpu") else np.array(coords)
            )

            # Crop the detected region from the original image.
            cropped_img: np.ndarray = crop_from_coordinates(
                orig_img, coords_np, mode="hbb"
            )

            conf, label_raw = predict_classification_real_ocr(cropped_img, cls_model)
            label_corrected = correct_spelling(label_raw, CORRECT_WORDS_SET)
        elif word_digit == "عدد":
            conf, label_raw, label_corrected = 1, "عدد", "عدد"
        else:
            raise ValueError("Invalid class label.")

        cls_labels.append(label_corrected)
        cls_labels_raw.append(label_raw)
        cls_conf.append(conf)

    # Save predicted labels in the result object.
    boxes.cls_labels = cls_labels
    boxes.cls_labels_raw = cls_labels_raw
    boxes.cls_conf = cls_conf
    return result


def extract_predictions(results: Any) -> List[Prediction]:
    """
    Extracts predictions from the YOLO11n results object.

    The results object is assumed to have:
        - an 'boxes' attribute with:
            - 'xyxy': a tensor of shape (N, 4) where each row is [x_min, y_min, x_max, y_max]
            - 'data': a tensor of shape (N, 7) where the last column is the predicted class index.
        - a 'names' attribute mapping class indices to text labels.

    Args:
        results (Any): The results object from the model.

    Returns:
        List[Prediction]: A list of predictions with computed centers and associated text.
    """
    predictions: List[Prediction] = []

    # Extract bounding boxes (xyxy) and convert to CPU numpy array if necessary
    xyxy_tensor = results.boxes.xyxy
    xyxy = (
        xyxy_tensor.cpu().numpy()
        if hasattr(xyxy_tensor, "cpu")
        else xyxy_tensor.numpy()
    )

    for i, box in enumerate(xyxy):
        x_min, y_min, x_max, y_max = box
        # Compute center of the bounding box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center = (center_x, center_y)
        # Get predicted class index and map it to text using the names dictionary
        # cls = int(data[i, -1])
        text = results.boxes.cls_labels[i]
        text_raw = results.boxes.cls_labels_raw[i]
        conf = results.boxes.cls_conf[i]
        if text == "others":
            predictions.append(
                Prediction(
                    bbox=(x_min, y_min, x_max, y_max), 
                    center=center, 
                    text="کوفت",
                    raw_text="کوفت"
                )
            )
        elif conf > 0.8:
            predictions.append(
                Prediction(
                    bbox=(x_min, y_min, x_max, y_max), 
                    center=center, 
                    text=text,
                    raw_text=text_raw
                )
            )
    return predictions


def sort_predictions(predictions: List[Prediction]) -> Tuple[str, str]:
    """
    Sorts predictions by their spatial location and merges their texts.

    The sorting algorithm works as follows:
      1. Sort all predictions by their center's y-coordinate in descending order (top-most first).
      2. Use the top-most prediction as the reference for the current line.
      3. Group all predictions whose center y-coordinate is greater than or equal to
         the reference's y_min value.
      4. Within the group (line), sort predictions from right to left based on center's x-coordinate.
      5. Merge the texts in the line and remove these predictions from the list.
      6. Repeat until no predictions remain.
      7. Merge all lines into a single text string.

    Args:
        predictions (List[Prediction]): List of predictions extracted from the results.

    Returns:
        str: The merged text from the sorted predictions.
    """
    # Sort by center's y (highest first)
    predictions_sorted = sorted(predictions, key=lambda p: p.center[1], reverse=True)
    merged_lines: List[str] = []

    while predictions_sorted:
        # Use the top-most prediction as the reference for this line
        reference = predictions_sorted[0]
        ref_y_min = reference.bbox[1]

        # Group all predictions with center y >= reference's y_min
        line_group = [p for p in predictions_sorted if p.center[1] >= ref_y_min]
        # Sort the grouped predictions from right to left (largest center x first)
        line_group_sorted = sorted(line_group, key=lambda p: p.center[0], reverse=True)

        line_group_sorted = sorted(line_group, key=lambda p: p.center[0], reverse=True)

        # Merge texts from the current line
        line_group_sorted_corrected = correct_line_group(line_group_sorted)
        line_text_corrected = " ".join(p.text for p in line_group_sorted_corrected)
        line_text_raw = " ".join(p.raw_text for p in line_group_sorted)
        merged_lines.append(line_text_corrected)

        # Remove grouped predictions from the sorted list
        predictions_sorted = [p for p in predictions_sorted if p not in line_group]

    # Merge all lines into a single text string (top to bottom)
    corrected_result = " ".join(merged_lines[::-1])
    
    # Build raw version (without line-level corrections)
    merged_lines_raw = []
    predictions_sorted = sorted(predictions, key=lambda p: p.center[1], reverse=True)
    
    while predictions_sorted:
        reference = predictions_sorted[0]
        ref_y_min = reference.bbox[1]
        line_group = [p for p in predictions_sorted if p.center[1] >= ref_y_min]
        line_group_sorted = sorted(line_group, key=lambda p: p.center[0], reverse=True)
        line_text_raw = " ".join(p.raw_text for p in line_group_sorted)
        merged_lines_raw.append(line_text_raw)
        predictions_sorted = [p for p in predictions_sorted if p not in line_group]
    
    raw_result = " ".join(merged_lines_raw[::-1])
    
    return corrected_result, raw_result


def correct_line_group(line_group_sorted: List[Prediction]) -> List[Prediction]:
    for i in range(len(line_group_sorted) - 1):
        curr_p_text = line_group_sorted[i].text
        next_p_text = line_group_sorted[i + 1].text
        if len(curr_p_text) > 2 and len(next_p_text) > 2:
            word_candidate = curr_p_text + " " + next_p_text
            corrected_word = correct_spelling(word_candidate, CORRECT_WORDS_SET)
            if corrected_word != word_candidate:
                line_group_sorted[i].text = corrected_word
                line_group_sorted[i + 1].text = ""
                i += 1
    return line_group_sorted


def sort_results_by_location(results: Any) -> Tuple[str, str]:
    """
    Processes a YOLO11n results object to sort predictions by their spatial location
    and merge them into one text string.

    Steps:
      1. Extract predictions from the results using the 'boxes' attribute.
      2. Sort predictions into lines based on the center coordinates.
      3. Within each line, sort predictions from right to left.
      4. Merge the sorted lines into a single text string.

    Args:
        results (Any): The YOLO11n results object containing attributes 'boxes' and 'names'.

    Returns:
        str: The merged text from the sorted predictions.
    """
    predictions = extract_predictions(results)
    return sort_predictions(predictions)  # Already returns tuple now   


def read_text(
    image: np.ndarray, word_digit_model: YOLO, cls_model: YOLO, save_path: str = None
) -> Tuple[str, str]:
    """
    Reads and extracts text from an image using a YOLO-based word-digit detection model and a classification model.

    This function performs the following steps:
      1. Validates that the input image is a NumPy array.
      2. Uses the word_digit_model to predict word and digit locations in the image.
      3. Optionally saves a visualization of the first detection result if a save_path is provided.
      4. Processes the raw detection predictions using the classification model (cls_model).
      5. Sorts the processed predictions by their spatial locations.
      6. Prints and returns the concatenated result string representing the detected text.

    Parameters:
        image (np.ndarray): The input image as a NumPy array.
        word_digit_model (YOLO): The YOLO model used for detecting words and digits in the image.
        cls_model (YOLO): The YOLO classification model used to process the detection predictions.
        save_path (str, optional): Directory path where the detection visualization image will be saved.
                                   If provided, saves the first detection result as 'word_digit_detection.jpg'.
                                   Defaults to None.

    Returns:
        str: A string representing the sorted text results extracted from the image.

    Raises:
        ValueError: If the input image is not a NumPy array.
        AttributeError: If the detection result does not support the 'save' method when a save_path is provided.
    """
    # Ensure the input image is valid
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")

    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    word_digit_res = word_digit_model.predict(
        image, device="cuda", iou=0.1, verbose=False
    )

    if save_path:
        if hasattr(word_digit_res[0], "save") and callable(word_digit_res[0].save):
            word_digit_res[0].save(f"{save_path}/word_digit_detection.png")
        else:
            raise AttributeError(
                "Result does not have a 'save' method for visualization."
            )
    word_digit_res = process_prediction(word_digit_res[0], cls_model)
    result_str_corrected, result_str_raw = sort_results_by_location(word_digit_res)
    
    if save_path:
        file_path = os.path.join(save_path, "read_text.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"{result_str_raw}\n")
            file.write(f"{result_str_corrected}\n")

    return result_str_corrected, result_str_raw


def predict_classification_real_ocr(
    image: np.ndarray, ocr_engine: TextRecognition
) -> Tuple[float, str]:
    """
    Performs text recognition on a single image using a pre-initialized model.

    Parameters
    ----------
    image : np.ndarray
        Input image array containing a single line of text.
    ocr_engine : PaddleOCR
        The initialized PaddleOCR instance.

    Returns
    -------
    Tuple[float, str]
        A tuple of (confidence, predicted_text).
        Returns (0.0, "") if recognition fails.
    """
    if image is None or not isinstance(image, np.ndarray):
        return 0.0, ""

    # The ocr method returns a list of results, even for one image
    # For recognition-only, the structure is [[(text, confidence)]]

    result = ocr_engine.predict(image)

    # Extract the text and confidence score
    if result and result[0] and result[0]["rec_text"]:
        predicted_text = result[0]["rec_text"]
        confidence_score = result[0]["rec_score"]
        return confidence_score, str(predicted_text)
    else:
        # To match your original function's fallback:
        return 1.0, "کوفت"
        # Or a more standard empty result:
        # return 0.0, ""


def has_valid_changes(
    source_word: str, candidate_word: str, valid_changes_dict: dict
) -> bool:
    """
    Check if the changes from source_word to candidate_word are valid based on a dictionary of valid changes.

    Args:
        source_word (str): The original word.
        candidate_word (str): The candidate corrected word.
        valid_changes_dict (dict): A dictionary where keys are characters in the source word
                                   and values are sets of valid replacement characters.

    Returns:
        bool: True if all character changes are valid, False otherwise.
    """
    if len(source_word) != len(candidate_word):
        return False

    changes_cnt = 0

    for s_char, c_char in zip(source_word, candidate_word):
        if s_char != c_char:
            if (
                s_char not in valid_changes_dict
                or c_char not in valid_changes_dict[s_char]
            ):
                return False
            else:
                changes_cnt += 1

            if changes_cnt > 1:
                return False
    return True


def correct_spelling(word: str, correct_words_set: Set[str]) -> str:
    """
    Corrects the spelling of a word based on the Levenshtein distance
    to a set of correct words. This version is optimized to avoid sorting
    the entire list of distances.

    Args:
        word (str): The word to correct.
        correct_words_set (Set[str]): A set of correct words to compare against.

    Returns:
        str: The corrected word or the original word if no unambiguous correction is found.
    """
    original_word = word
    stripped_word = word.replace(" ", "")
    if len(stripped_word) < 6 or not stripped_word:
        return original_word

    # Fast path: check for an exact match first.
    if stripped_word in correct_words_set:
        return stripped_word

    min_distance = float("inf")
    best_match = stripped_word
    ambiguous = False

    # Optimization: Filter candidates by length.
    # A Levenshtein distance of 1 is only possible if lengths differ by at most 1.
    min_len = len(stripped_word) - 1
    max_len = len(stripped_word) + 1
    candidate_words = [w for w in correct_words_set if min_len <= len(w) <= max_len]

    for candidate_word in candidate_words:
        if len(candidate_word) == len(stripped_word):
            if not has_valid_changes(
                stripped_word, candidate_word, VALID_SPELL_CORRECTION_DICT
            ):
                candidate_words.remove(candidate_word)

    # Find the best match without sorting all distances
    for true_word in candidate_words:
        dist = Levenshtein.distance(stripped_word, true_word)

        if dist < min_distance:
            min_distance = dist
            best_match = true_word
            ambiguous = False
            # If we find a perfect match, we can stop early.
            if dist == 0:
                return true_word
        elif dist == min_distance:
            ambiguous = True

    # Return the correction only if it's an unambiguous single-edit-distance match.
    if min_distance == 1 and not ambiguous:
        return best_match

    # In all other cases, return the original word.
    return original_word