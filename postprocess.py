#!/usr/bin/env python3
"""
Image Data Processing Pipeline

This comprehensive script processes image metadata through a complete pipeline:

Pipeline Stages:
    1. CSV Generation: Scans folder structure to extract metadata from text
       files (postcodes, addresses, regions) and organizes into CSV format.

    2. Pattern Matching: Applies rule-based pattern matching to addresses
       using an Excel file containing matching patterns and 3-digit codes.

    3. Document Classification: Uses YOLO classification model to predict
       document types from preprocessed images.

    4. Data Consolidation: Merges new data with existing historical records,
       removes duplicates, and filters out incomplete entries.

Expected Folder Structure:
    root_path/
        ├── 02_intermediate/          # Contains image subfolders
        │   ├── image_001/
        │   ├── image_002/
        │   └── ...
        ├── 03_postcode_regions/      # Postcode information
        │   ├── image_001/
        │   │   └── info.txt          # Line 1: area, Line 2: full, Line 3: stat
        │   └── ...
        ├── 03_address_region/        # Address region data
        │   ├── image_001/
        │   │   └── region.txt        # Line 1: region name
        │   └── ...
        └── 03_read_address/          # Raw and preprocessed addresses
            ├── image_001/
            │   └── address.txt       # Line 1: raw, Line 2: preprocessed
            └── ...

Dependencies:
    - pandas
    - openpyxl
    - ultralytics (YOLO)
    - torch (PyTorch)
    - tqdm (optional, for progress bars)

Installation:
    pip install pandas openpyxl ultralytics torch tqdm --break-system-packages

Author: Data Processing Pipeline Team
Version: 1.0
"""

import os
import csv
import re
import argparse
from pathlib import Path
import pandas as pd
import torch
from ultralytics import YOLO
from src.data_pipeline.pipelines.utils.CONSTS.CONSTS import RULES, REPLACEMENT_DICT, ABBREVIATION_DICT, KEYWORDS_TO_CUT_BEFORE, KEYWORDS_TO_CUT_AFTER

# Optional progress bar library for better user experience
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Image file suffix and valid extensions for YOLO classification
IMAGE_SUFFIX = "_receiver_box_prep"
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".bmp", ".webp")


# =============================================================================
# UTILITY FUNCTIONS - File System Operations
# =============================================================================

def find_txt_file(folder_path):
    """
    Find the single .txt file in a specified folder.

    This function searches for exactly one text file within the given folder.
    It's designed for folders that should contain only one .txt file with
    specific information (like postcode data or address information).

    Args:
        folder_path (str): Path to the folder to search in.

    Returns:
        str or None: Full path to the .txt file if exactly one is found,
                     None otherwise (folder doesn't exist, no .txt files,
                     or multiple .txt files present).

    Example:
        >>> find_txt_file("/data/image_001/")
        "/data/image_001/postcode_info.txt"
    """
    # Check if the folder exists before attempting to list files
    if not os.path.exists(folder_path):
        return None

    # Get all .txt files in the folder
    txt_files = [
        f for f in os.listdir(folder_path)
        if f.endswith('.txt')
    ]

    # Return path only if exactly one .txt file exists
    if len(txt_files) == 1:
        return os.path.join(folder_path, txt_files[0])
    return None


def get_all_image_names(root_path):
    """
    Retrieve all image names (folder names) from the 02_intermediate folder.

    This function scans the intermediate data folder and returns a sorted list
    of all subdirectory names. Each subdirectory represents one image/document
    that has been processed through earlier pipeline stages.

    Args:
        root_path (str): Path to the root data folder containing the
                        '02_intermediate' subfolder.

    Returns:
        list: Sorted list of folder names (strings) representing image IDs.
              Returns empty list if the intermediate folder doesn't exist.

    Example:
        >>> get_all_image_names("/data/")
        ['image_001', 'image_002', 'image_003', ...]
    """
    intermediate_path = os.path.join(root_path, '02_intermediate')

    # Validate that the intermediate folder exists
    if not os.path.exists(intermediate_path):
        print(f"Error: {intermediate_path} does not exist!")
        return []

    # Get all subdirectories (each represents an image)
    image_names = [
        name for name in os.listdir(intermediate_path)
        if os.path.isdir(os.path.join(intermediate_path, name))
    ]

    # Return sorted list for consistent processing order
    return sorted(image_names)


# =============================================================================
# DATA READING FUNCTIONS - Extract Information from Text Files
# =============================================================================

def read_postcode_info(root_path, image_name):
    """
    Read postcode information from the 03_postcode_regions folder.

    Postcodes are stored in text files with three lines:
        Line 1: Postcode area (region/district)
        Line 2: Full postcode
        Line 3: Postcode status/statistics

    Args:
        root_path (str): Root data folder path.
        image_name (str): Name of the image folder to read from.

    Returns:
        tuple: (postcode_area, full_postcode, postcode_stat)
               Each element is a string. Empty strings returned if file
               not found or errors occur.

    Example:
        >>> read_postcode_info("/data/", "image_001")
        ('Tehran-District5', '1234567890', 'valid')
    """
    folder_path = os.path.join(
        root_path, '03_postcode_regions', image_name
    )
    txt_file = find_txt_file(folder_path)

    # Return empty tuple if no text file found
    if txt_file is None:
        return ('', '', '')

    try:
        # Read file with UTF-8 encoding (with BOM support)
        with open(txt_file, 'r', encoding='utf-8-sig') as f:
            lines = f.read().splitlines()

        # Extract each line, providing empty string as default
        postcode_area = lines[0] if len(lines) > 0 else ''
        full_postcode = lines[1] if len(lines) > 1 else ''
        postcode_stat = lines[2] if len(lines) > 2 else ''

        return (postcode_area, full_postcode, postcode_stat)

    except Exception as e:
        print(f"Error reading postcode for {image_name}: {e}")
        return ('', '', '')


def read_address_region(root_path, image_name):
    """
    Read address region information from the 03_address_region folder.

    The region represents a geographic area classification for the address.
    This is typically stored as a single line in a text file.

    Args:
        root_path (str): Root data folder path.
        image_name (str): Name of the image folder to read from.

    Returns:
        str: The region name, or empty string if not found or error occurs.

    Example:
        >>> read_address_region("/data/", "image_001")
        'Central District'
    """
    folder_path = os.path.join(
        root_path, '03_address_region', image_name
    )
    txt_file = find_txt_file(folder_path)

    # Return empty string if no text file found
    if txt_file is None:
        return ''

    try:
        # Read file with UTF-8 encoding (with BOM support)
        with open(txt_file, 'r', encoding='utf-8-sig') as f:
            lines = f.read().splitlines()

        # Return first line or empty string
        return lines[0] if len(lines) > 0 else ''

    except Exception as e:
        print(f"Error reading address region for {image_name}: {e}")
        return ''


def read_address_info(root_path, image_name):
    """
    Read raw and preprocessed address data from the 03_read_address folder.

    Addresses are stored in two forms:
        Line 1: Raw address (original text extracted from image)
        Line 2: Preprocessed address (cleaned, normalized text)

    The preprocessed version is used for pattern matching in later stages.

    Args:
        root_path (str): Root data folder path.
        image_name (str): Name of the image folder to read from.

    Returns:
        tuple: (raw_address, preprocessed_address)
               Both elements are strings. Empty strings returned if file
               not found or errors occur.

    Example:
        >>> read_address_info("/data/", "image_001")
        ('Original Street Name 123', 'normalized_street_name_123')
    """
    folder_path = os.path.join(root_path, '03_read_address', image_name)
    txt_file = find_txt_file(folder_path)

    # Return empty tuple if no text file found
    if txt_file is None:
        return ('', '')

    try:
        # Read file with UTF-8 encoding (with BOM support)
        with open(txt_file, 'r', encoding='utf-8-sig') as f:
            lines = f.read().splitlines()

        # Extract both address forms
        raw_address = lines[0] if len(lines) > 0 else ''
        preprocessed_address = lines[1] if len(lines) > 1 else ''

        return (raw_address, preprocessed_address)

    except Exception as e:
        print(f"Error reading address info for {image_name}: {e}")
        return ('', '')


# =============================================================================
# PATTERN MATCHING FUNCTIONS - Address Rule Processing
# =============================================================================

def load_rules(excel_file_path):
    """
    Load and preprocess pattern matching rules from an Excel file.

    The Excel file contains address matching patterns with positive (+) and
    negative (-) conditions. Each pattern is associated with a 3-digit code.

    Excel Format:
        Column 1: Pattern (e.g., "+street_name-district_5")
        Column 2: 3-digit code (e.g., 123)

    The function parses each pattern into structured conditions that can be
    efficiently evaluated against address strings.

    Args:
        excel_file_path (str): Path to the patterns.xlsx file.

    Returns:
        tuple: (rules, dataframe)
            rules (list): List of tuples, each containing:
                - conditions: List of (operator, phrase) tuples
                  operator is " + " (must contain) or " - " (must not contain)
                  phrase is the text to match
                - number: The associated 3-digit code (int)
            dataframe: The original pandas DataFrame (for further processing)
                      Returns None if loading fails.

    Example:
        >>> rules, df = load_rules("patterns.xlsx")
        >>> rules[0]
        ([(' + ', ' main_street '), (' - ', ' district_2 ')], 101)
    """
    try:
        df = pd.read_excel(excel_file_path, header=None, names=['pattern', 'number'])
        def remove_dots_regex(value):
            if pd.isna(value):
                return value
            return re.sub(r'\.', '', str(value))
        
        df = df.map(remove_dots_regex)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []
    
    rules = []
    for _, row in df.iterrows():
        pattern_str = str(row['pattern']).strip()
        number = int(row['number'])
        
        tokens = re.split('([+-])', pattern_str)
        tokens = [' '+token.strip()+' ' for token in tokens if token.strip()]
        if not tokens:
            continue
        conditions = []
        conditions.append((' + ', tokens[0]))
        
        i = 1
        while i < len(tokens):
            operator = tokens[i]
            phrase = tokens[i+1] if i+1 < len(tokens) else ''
            if operator in (' + ', ' - ') and phrase:
                conditions.append((operator, phrase))
            i += 2
        rules.append((conditions, number))
    return rules,df


def process_phrases_to_dict_to_handle_underlines(df):
    """
    Create a phrase replacement dictionary from Excel patterns.

    This function processes the Excel DataFrame to find all phrases containing
    underscores and creates a mapping from space-separated versions to
    underscore versions. This is crucial for matching multi-word phrases.

    For example:
        "main street" -> "main_street"
        "central district" -> "central_district"

    This ensures that addresses with spaces can still match patterns that
    use underscores as word separators.

    Args:
        df (pandas.DataFrame): DataFrame loaded from patterns Excel file.

    Returns:
        dict: Mapping from {phrase_with_spaces: phrase_with_underscores}

    Example:
        >>> df = pd.DataFrame({'pattern': ['+main_street-old_town']})
        >>> process_phrases_to_dict_to_handle_underlines(df)
        {'main street': 'main_street', 'old town': 'old_town'}
    """
    replacement_dict = {}

    # Scan all cells in the DataFrame
    for column in df.columns:
        for cell in df[column]:
            # Only process non-empty string cells
            if pd.notna(cell) and isinstance(cell, str):
                # Split by operators to get individual phrases
                split_phrases = re.split(r'[+-]', cell)

                for phrase in split_phrases:
                    phrase = phrase.strip()

                    # Only process phrases containing underscores
                    if phrase and '_' in phrase:
                        # Create space-separated version as key
                        space_version = phrase.replace('_', ' ')
                        replacement_dict[space_version] = phrase

    return replacement_dict


def replace_phrases(input_string, replacement_dict):
    """
    Replace space-separated phrases with underscore versions in a string.

    This function performs longest-match-first replacement to handle
    overlapping phrases correctly. For example, if we have both
    "main street" and "street", we want to match "main street" first.

    Args:
        input_string (str): The address string to process.
        replacement_dict (dict): Mapping from space phrases to underscore
                                phrases (from process_phrases_to_dict...).

    Returns:
        str: String with all replacements applied.

    Example:
        >>> replacements = {'main street': 'main_street'}
        >>> replace_phrases('the main street is', replacements)
        'the main_street is'
    """
    # Sort replacements by length (longest first) to avoid partial matches
    sorted_replacements = sorted(
        replacement_dict.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    # Apply each replacement
    for space_phrase, underscore_phrase in sorted_replacements:
        input_string = input_string.replace(space_phrase, underscore_phrase)

    return input_string


def match_pattern_get_all_rules(input_string, rules, replacements):
    """
    Match an address string against all rules and return matching patterns.

    This is the core pattern matching function. It preprocesses the address
    string, applies phrase replacements, and evaluates all rules to find
    matches. Multiple rules can match a single address.

    Preprocessing steps:
        1. Add surrounding spaces for boundary matching
        2. Remove text before location keywords (city, province, etc.)
        3. Remove text after certain separators (building codes, etc.)
        4. Replace abbreviated terms with full forms
        5. Apply underscore phrase replacements
        6. Evaluate positive (+) and negative (-) conditions

    Args:
        input_string (str): The preprocessed address to match.
        rules (list): List of rule tuples from load_rules().
        replacements (dict): Phrase replacement dictionary.

    Returns:
        str: Comma-separated string of all matching rule conditions in
             readable format (e.g., "+street_a-district_b,+avenue_c").
             Returns empty string if no matches found.

    Example:
        >>> match_pattern_get_all_rules(
        ...     "address on main street in district 5",
        ...     rules,
        ...     replacements
        ... )
        '+main_street-district_2,+main_street-district_4'
    """
    if not rules:
        rules = RULES

    if not replacements:
        replacements = REPLACEMENT_DICT

    # Add spaces around the remaining string
    input_string = " " + input_string.strip() + " "

    # if " فرستنده " in input_string:
    #     return -1

    for abbr in ABBREVIATION_DICT:
        input_string = input_string.replace(abbr, ABBREVIATION_DICT[abbr])

    for keyword in KEYWORDS_TO_CUT_BEFORE:
        index = input_string.find(keyword)
        if index != -1:
            input_string = input_string[index:]

    for keyword in KEYWORDS_TO_CUT_AFTER:
        index = input_string.find(keyword)
        if index != -1:
            input_string = input_string[: index + 1]

    for remove_str in [" شهید ", " سید "]:
        input_string = input_string.replace(remove_str, " ")

    # Apply phrase replacements
    input_string = replace_phrases(input_string, replacements)

    # Evaluate all rules against the processed string
    matched_conditions = []
    for conditions, number in rules:
        match = True
        for operator, phrase in conditions:
            if operator == " + ":
                if phrase not in input_string:
                    match = False
                    break
            elif operator == " - ":
                if phrase in input_string:
                    match = False
                    break

        # If all conditions satisfied, add to matched list
        if match:
            matched_conditions.append(conditions)

    # Return empty string if no matches
    if not matched_conditions:
        return ''

    # Format all matched conditions in readable format
    result_parts = []
    for conditions in matched_conditions:
        # Format each rule as: +phrase1 -phrase2 +phrase3
        rule_str = ' '.join([
            f"{op.strip()}{phrase.strip()}"
            for op, phrase in conditions
        ])
        result_parts.append(rule_str)

    # Join multiple matching rules with commas
    return ','.join(result_parts)


def escape_for_excel(value):
    """
    Escape values that might be interpreted as formulas by Excel.

    Excel interprets strings starting with +, -, =, or @ as formulas,
    which can cause errors or security warnings. This function adds a
    single quote prefix to prevent formula interpretation.

    Args:
        value: The value to escape (can be any type).

    Returns:
        str or original type: Escaped string if needed, otherwise original
                             value unchanged.

    Example:
        >>> escape_for_excel("+123")
        "'+123"
        >>> escape_for_excel("normal text")
        "normal text"
        >>> escape_for_excel(42)
        42
    """
    # Only process string values
    if not value or not isinstance(value, str):
        return value

    # Add single quote prefix if starts with formula character
    if value.startswith(('+', '-', '=', '@')):
        return "'" + value

    return value


# =============================================================================
# YOLO CLASSIFICATION FUNCTIONS - Document Type Prediction
# =============================================================================

def find_target_image(folder_path, suffix=IMAGE_SUFFIX,
                      exts=VALID_IMAGE_EXTENSIONS):
    """
    Find the preprocessed image file in a folder for YOLO classification.

    This function searches for an image file whose name (without extension)
    ends with the specified suffix. This is typically the preprocessed image
    that's ready for document type classification.

    Args:
        folder_path (str): Path to the folder containing images.
        suffix (str): Filename suffix to match (default: "_receiver_box_prep").
        exts (tuple): Valid image file extensions to consider.

    Returns:
        str or None: Full path to the matching image file, or None if not
                    found.

    Example:
        >>> find_target_image("/data/image_001/")
        "/data/image_001/doc_receiver_box_prep.jpg"
    """
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)

        # Skip if not a file
        if not os.path.isfile(fpath):
            continue

        # Split filename into name and extension
        name, ext = os.path.splitext(fname)

        # Check if extension is valid (case-insensitive)
        if ext.lower() not in exts:
            continue

        # Check if filename ends with target suffix
        if name.endswith(suffix):
            return fpath

    return None


def predict_image(model, image_path):
    """
    Run YOLO classification model on an image and extract prediction.

    This function handles multiple YOLO API versions and output formats to
    ensure robust prediction extraction. It attempts several methods to
    retrieve the predicted class and confidence score.

    The function is designed to be resilient to API changes in the
    Ultralytics library by trying multiple extraction approaches.

    Args:
        model: Loaded YOLO model object (from ultralytics.YOLO).
        image_path (str): Path to the image file to classify.

    Returns:
        tuple: (class_name, confidence)
            class_name (str): Predicted class name, or None if prediction
                             failed.
            confidence (float): Confidence score (0-1), or None if not
                               available.

    Prediction Strategies:
        1. Classification via r.probs (Probs object with top1/top1conf)
        2. Classification via probability array (tensor/numpy conversion)
        3. Detection-style output via r.boxes (takes highest confidence)
        4. Fallback to first available class name

    Example:
        >>> model = YOLO("model.pt")
        >>> predict_image(model, "/data/image.jpg")
        ('invoice', 0.9234)
    """
    # Run inference on the image
    results = model(image_path)

    # Results is list-like, take first element
    if len(results) == 0:
        return None, None
    r = results[0]

    # Helper: Get class names mapping from model or result
    names = None
    if hasattr(model, "names") and model.names is not None:
        names = model.names
    elif hasattr(r, "names") and r.names is not None:
        names = r.names

    # Strategy 1: Classification via r.probs (modern Ultralytics API)
    probs = getattr(r, "probs", None)
    if probs is not None:
        # Check if it's Ultralytics' Probs class with top1/top1conf
        if hasattr(probs, 'top1') and hasattr(probs, 'top1conf'):
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)

            # Resolve class ID to name
            if isinstance(names, dict):
                class_name = names.get(class_id, str(class_id))
            elif isinstance(names, (list, tuple)):
                class_name = (
                    names[class_id] if class_id < len(names)
                    else str(class_id)
                )
            else:
                class_name = str(class_id)
            return class_name, confidence

        # Strategy 2: Fallback for older API - convert to numpy array
        try:
            # Access underlying data if it's a Probs object
            if hasattr(probs, 'data'):
                probs_raw = probs.data
            else:
                probs_raw = probs

            # Convert PyTorch tensor to numpy if needed
            if hasattr(probs_raw, "cpu"):
                probs_arr = probs_raw.cpu().numpy()
            else:
                import numpy as np
                probs_arr = np.array(probs_raw)

            if probs_arr is not None and len(probs_arr) > 0:
                # Handle batch dimension if present (N, C)
                if hasattr(probs_arr, 'ndim') and probs_arr.ndim == 2:
                    probs_arr = probs_arr[0]

                # Get class with highest probability
                class_id = int(probs_arr.argmax())
                confidence = float(probs_arr[class_id])

                # Resolve class ID to name
                if isinstance(names, dict):
                    class_name = names.get(class_id, str(class_id))
                elif isinstance(names, (list, tuple)):
                    class_name = (
                        names[class_id] if class_id < len(names)
                        else str(class_id)
                    )
                else:
                    class_name = str(class_id)
                return class_name, confidence

        except (AttributeError, ImportError, IndexError):
            # Continue to next strategy if this fails
            pass

    # Strategy 3: Detection-style output via r.boxes
    # (some models output bounding boxes even for classification)
    boxes = getattr(r, "boxes", None)
    if boxes is not None:
        # Extract class and confidence attributes
        cls_attr = getattr(boxes, "cls", None)
        conf_attr = (
            getattr(boxes, "conf", None) or
            getattr(boxes, "confidence", None)
        )

        try:
            if cls_attr is not None:
                # Convert to numpy arrays
                if hasattr(cls_attr, "cpu"):
                    cls_arr = cls_attr.cpu().numpy()
                else:
                    cls_arr = cls_attr

                if hasattr(conf_attr, "cpu"):
                    conf_arr = conf_attr.cpu().numpy()
                else:
                    conf_arr = (
                        conf_attr if conf_attr is not None
                        else None
                    )

                # Choose box with highest confidence
                if conf_arr is not None and len(conf_arr) > 0:
                    idx = int(conf_arr.argmax())
                    class_id = int(cls_arr[idx])
                    confidence = float(conf_arr[idx])
                else:
                    # Fallback: take first box
                    class_id = int(cls_arr[0])
                    confidence = None

                # Resolve class ID to name
                if isinstance(names, dict):
                    class_name = names.get(class_id, str(class_id))
                elif isinstance(names, (list, tuple)):
                    class_name = (
                        names[class_id] if class_id < len(names)
                        else str(class_id)
                    )
                else:
                    class_name = str(class_id)
                return class_name, confidence

        except (AttributeError, IndexError, TypeError):
            # Continue to fallback if this fails
            pass

    # Strategy 4: Final fallback - return first available class name
    try:
        if names is not None:
            if isinstance(names, dict):
                # Choose the first mapping value
                first_name = next(iter(names.values()))
                return first_name, None
            elif isinstance(names, (list, tuple)) and len(names) > 0:
                return names[0], None
    except (StopIteration, TypeError):
        pass

    # All strategies failed
    return None, None


# =============================================================================
# PIPELINE STAGE FUNCTIONS
# =============================================================================

def stage_1_generate_csv_with_patterns(root_path, output_csv,
                                       patterns_excel_path=None):
    """
    Pipeline Stage 1: Generate CSV from folder structure with pattern matching.

    This stage scans the folder hierarchy, extracts metadata from text files,
    and applies rule-based pattern matching to addresses. The result is a
    comprehensive CSV containing all extracted information.

    Process:
        1. Load pattern matching rules from Excel (if provided)
        2. Get list of all image folders from 02_intermediate
        3. For each image folder:
           - Read postcode information (area, full code, status)
           - Read address region
           - Read raw and preprocessed addresses
           - Apply pattern matching to preprocessed address
        4. Write all data to CSV with proper encoding

    Args:
        root_path (str): Path to root folder containing data structure.
        output_csv (str): Path where the CSV file will be created.
        patterns_excel_path (str, optional): Path to patterns.xlsx file.
                                            If None, matched rules column
                                            will be empty.

    Output CSV Columns:
        - image_name: Folder/image identifier
        - postcode_area: Postcode region
        - full_postcode: Complete postcode
        - postcode_stat: Postcode status
        - address_region: Geographic region
        - raw_address: Original address text
        - preprocessed_address: Cleaned address text
        - matched_rules: Comma-separated matching patterns

    Returns:
        None: Function writes directly to CSV file.

    Example:
        >>> stage_1_generate_csv_with_patterns(
        ...     "/data/",
        ...     "output.csv",
        ...     "patterns.xlsx"
        ... )
        Loading pattern rules from: patterns.xlsx
        Loaded 150 rules and 45 phrase replacements
        Found 1000 images
        Processing 100/1000 images...
        ...
        Successfully created CSV file: output.csv
    """
    print("\n" + "=" * 70)
    print("STAGE 1: Generating CSV from folder structure")
    print("=" * 70)

    # Load pattern matching rules if provided
    rules = []
    replacements = {}
    if patterns_excel_path:
        print(f"Loading pattern rules from: {patterns_excel_path}")
        rules, df = load_rules(patterns_excel_path)
        if df is not None:
            replacements = process_phrases_to_dict_to_handle_underlines(df)
            print(f"Loaded {len(rules)} rules and "
                  f"{len(replacements)} phrase replacements")
        else:
            print("Warning: Failed to load rules, matched rules column "
                  "will be empty")
    else:
        print("No patterns file provided, skipping pattern matching")

    # Get all image names from intermediate folder
    image_names = get_all_image_names(root_path)

    if not image_names:
        print("Error: No images found in 02_intermediate folder!")
        return

    print(f"Found {len(image_names)} images to process")

    # Prepare CSV data structure
    csv_data = []
    headers = [
        'image_name',           # Image identifier
        'postcode_area',        # Postcode region
        'full_postcode',        # Complete postcode
        'postcode_stat',        # Postcode status
        'address_region',       # Geographic region
        'raw_address',          # Original address
        'preprocessed_address', # Cleaned address
        'matched_rules'         # Pattern matches
    ]

    # Process each image
    for idx, image_name in enumerate(image_names):
        # Progress indicator every 100 images
        if (idx + 1) % 100 == 0:
            print(f"Processing {idx + 1}/{len(image_names)} images...")

        # Read postcode information
        postcode_area, full_postcode, postcode_stat = \
            read_postcode_info(root_path, image_name)

        # Read address region
        address_region = read_address_region(root_path, image_name)

        # Read address information (raw and preprocessed)
        raw_address, preprocessed_address = \
            read_address_info(root_path, image_name)

        # Apply pattern matching to preprocessed address
        matched_rules = ''
        if rules and replacements and preprocessed_address:
            matched_rules = match_pattern_get_all_rules(
                preprocessed_address, rules, replacements
            )

        # Build CSV row
        row = [
            image_name,
            postcode_area,
            full_postcode,
            postcode_stat,
            address_region,
            raw_address,
            preprocessed_address,
            escape_for_excel(matched_rules)  # Prevent Excel formula injection
        ]
        csv_data.append(row)

    # Write to CSV file with UTF-8 BOM encoding for Excel compatibility
    try:
        with open(output_csv, 'w', newline='',
                  encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(csv_data)

        print(f"\n✓ Successfully created CSV file: {output_csv}")
        print(f"✓ Total rows written: {len(csv_data)}")

    except Exception as e:
        print(f"\n✗ Error writing CSV file: {e}")


def stage_2_add_yolo_predictions(root_path, csv_path, model_path,
                                 output_csv, use_cuda=False,
                                 image_suffix=IMAGE_SUFFIX):
    """
    Pipeline Stage 2: Add YOLO document type predictions to CSV.

    This stage loads a YOLO classification model and predicts document types
    for images found in the folder structure. Predictions are added as new
    columns to the existing CSV.

    Process:
        1. Load existing CSV from Stage 1
        2. Validate required columns exist
        3. Load YOLO classification model
        4. For each image folder:
           - Find preprocessed image (with specified suffix)
           - Run YOLO prediction
           - Extract class name and confidence
           - Update corresponding CSV row
        5. Write updated CSV with predictions

    Args:
        root_path (str): Path to root folder (contains 02_intermediate).
        csv_path (str): Path to CSV from Stage 1.
        model_path (str): Path to YOLO .pt model file.
        output_csv (str): Path for output CSV with predictions.
        use_cuda (bool): Whether to use GPU acceleration if available.
        image_suffix (str): Suffix of preprocessed image files to process.

    New CSV Columns Added:
        - Document Type: Predicted document class name
        - Document Type Confidence: Prediction confidence (0-1)

    Returns:
        None: Function writes directly to CSV file.

    Raises:
        FileNotFoundError: If root_path, csv_path, or model_path don't exist.
        KeyError: If CSV doesn't contain 'image_name' column.

    Example:
        >>> stage_2_add_yolo_predictions(
        ...     "/data/",
        ...     "stage1_output.csv",
        ...     "model.pt",
        ...     "stage2_output.csv",
        ...     use_cuda=True
        ... )
        Loading CSV from: stage1_output.csv
        Loading YOLO model from: model.pt
        Using GPU for inference
        Model classes: {0: 'invoice', 1: 'receipt', 2: 'contract'}
        Processing 1000 folders...
        ✓ Successfully updated CSV with predictions
    """
    print("\n" + "=" * 70)
    print("STAGE 2: Adding YOLO document type predictions")
    print("=" * 70)

    # Validate input paths
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f"Root directory not found: {root_path}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load CSV with all columns as strings to preserve formatting
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, encoding='utf-8-sig')

    # Validate required column exists
    if "image_name" not in df.columns:
        raise KeyError(
            "CSV must contain a column named exactly 'image_name'"
        )

    # Add target columns if they don't exist
    if "document_type" not in df.columns:
        df["document_type"] = ""

    if "document_type_confidence" not in df.columns:
        df["document_type_confidence"] = ""

    # Load YOLO model
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    # Configure GPU/CPU usage
    if use_cuda and torch.cuda.is_available():
        try:
            model.to("cuda")
            print("✓ Using GPU for inference")
        except Exception as e:
            print(f"⚠ Warning: Could not move model to CUDA: {e}")
            print("⚠ Continuing on CPU")
    else:
        if use_cuda:
            print("⚠ CUDA requested but not available. Running on CPU.")
        else:
            print("Running on CPU (use --use-cuda for GPU if available)")

    # Display model classes for reference
    try:
        print(f"Model classes: {model.names}")
    except AttributeError:
        print("⚠ Could not retrieve model class names")

    # Get all subfolders from 02_intermediate
    intermediate_path = os.path.join(root_path, '02_intermediate')
    subfolders = [
        d for d in os.listdir(intermediate_path)
        if os.path.isdir(os.path.join(intermediate_path, d))
    ]

    print(f"\nProcessing {len(subfolders)} folders...")

    # Statistics tracking
    missing_images = 0
    failed_predictions = 0
    no_csv_match = 0
    successful_predictions = 0

    # Set up progress iterator
    iterator = (
        tqdm(subfolders, desc="Predicting document types")
        if TQDM_AVAILABLE else subfolders
    )

    # Process each folder
    for sub in iterator:
        sub_path = os.path.join(intermediate_path, sub)

        # Find the preprocessed image
        target_img = find_target_image(sub_path, suffix=image_suffix)
        if target_img is None:
            missing_images += 1
            continue

        # Run YOLO prediction
        class_name, confidence = predict_image(model, target_img)

        # Handle prediction failure
        if class_name is None:
            failed_predictions += 1
            # Mark as unknown in CSV if row exists
            mask = df["image_name"] == sub
            if mask.any():
                df.loc[mask, "document_type"] = "unknown"
                df.loc[mask, "document_type_confidence"] = ""
            else:
                no_csv_match += 1
            continue

        # Update CSV with prediction
        mask = df["image_name"] == sub
        if not mask.any():
            no_csv_match += 1
            continue

        df.loc[mask, "document_type"] = class_name
        conf_str = (
            "" if confidence is None
            else f"{confidence:.4f}"
        )
        df.loc[mask, "document_type_confidence"] = conf_str
        successful_predictions += 1

    # Save updated CSV
    print(f"\nSaving updated CSV to: {output_csv}")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # Print summary statistics
    print("\n" + "-" * 70)
    print("STAGE 2 SUMMARY:")
    print("-" * 70)
    print(f"✓ Successful predictions: {successful_predictions}")
    if missing_images > 0:
        print(f"⚠ Folders with no matching image "
              f"(suffix: '{image_suffix}'): {missing_images}")
    if failed_predictions > 0:
        print(f"⚠ Images with failed predictions: {failed_predictions}")
    if no_csv_match > 0:
        print(f"⚠ Subfolders with no matching CSV row: {no_csv_match}")
    print(f"\n✓ Updated CSV saved to: {output_csv}")


def stage_3_consolidate_data(new_csv_path, existing_csv_path, output_csv):
    """
    Pipeline Stage 3: Consolidate new data with existing records.

    This stage merges the newly processed data with historical records,
    removes duplicates, and filters out incomplete entries. The "last write
    wins" strategy is used for duplicate image names.

    Process:
        1. Load new CSV from Stage 2
        2. Load existing historical CSV (if provided)
        3. Concatenate both DataFrames
        4. Remove duplicates based on image_name (keep last occurrence)
        5. Filter out rows with empty postcode_area
        6. Write consolidated CSV

    Deduplication Strategy:
        - If an image_name appears in both files, keep the version from
          the NEW file (Stage 2 output)
        - This ensures latest processing results are retained

    Data Quality Filtering:
        - Rows with missing or empty postcode_area are removed
        - This ensures only complete, valid records are in final output

    Args:
        new_csv_path (str): Path to CSV from Stage 2 (with predictions).
        existing_csv_path (str or None): Path to existing historical CSV.
                                         If None, only new data is used.
        output_csv (str): Path for final consolidated CSV.

    Returns:
        None: Function writes directly to CSV file.

    Raises:
        FileNotFoundError: If new_csv_path doesn't exist.
        KeyError: If required columns are missing.

    Example:
        >>> stage_3_consolidate_data(
        ...     "stage2_output.csv",
        ...     "historical_data.csv",
        ...     "final_output.csv"
        ... )
        Loading new data from: stage2_output.csv
        Loading existing data from: historical_data.csv
        Merging datasets...
        Removing duplicates (keeping latest)...
        Filtering incomplete records...
        ✓ Final dataset: 9543 unique records
        ✓ Consolidated CSV saved to: final_output.csv
    """
    print("\n" + "=" * 70)
    print("STAGE 3: Consolidating and deduplicating data")
    print("=" * 70)

    # Load new data from Stage 2
    print(f"Loading new data from: {new_csv_path}")
    if not os.path.isfile(new_csv_path):
        raise FileNotFoundError(f"New CSV file not found: {new_csv_path}")
    
    df_new = pd.read_csv(new_csv_path, encoding='utf-8-sig')
    print(f"✓ Loaded {len(df_new)} new records")

    # Load existing historical data if provided
    if existing_csv_path and os.path.isfile(existing_csv_path):
        print(f"Loading existing data from: {existing_csv_path}")
        
        df_existing = pd.read_csv(existing_csv_path, encoding='utf-8-sig')
        print(f"✓ Loaded {len(df_existing)} existing records")

        # Concatenate DataFrames
        print("\nMerging datasets...")
        combined_df = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"✓ Combined total: {len(combined_df)} records")
    else:
        if existing_csv_path:
            print(f"⚠ Existing CSV not found: {existing_csv_path}")
        print("Using only new data (no existing data to merge)")
        combined_df = df_new

    # Remove duplicates based on image_name
    # keep='last' ensures the newest data (df_new) is retained
    print("\nRemoving duplicates (keeping latest)...")
    initial_count = len(combined_df)
    combined_unique = combined_df.drop_duplicates(
        subset="image_name",
        keep="last"
    )
    duplicates_removed = initial_count - len(combined_unique)
    print(f"✓ Removed {duplicates_removed} duplicate records")

    # Filter out rows with empty postcode_area
    # This ensures data quality by removing incomplete records
    print("\nFiltering incomplete records (empty postcode_area)...")
    initial_count = len(combined_unique)
    combined_unique = combined_unique[
        combined_unique["postcode_area"].notna() &
        (combined_unique["postcode_area"].astype(str).str.strip() != "")
    ]
    filtered_count = initial_count - len(combined_unique)
    print(f"✓ Removed {filtered_count} incomplete records")

    # Save final consolidated CSV
    print(f"\nSaving consolidated data to: {output_csv}")
    combined_unique.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # Print final summary
    print("\n" + "-" * 70)
    print("STAGE 3 SUMMARY:")
    print("-" * 70)
    print(f"✓ Final dataset: {len(combined_unique)} unique, complete records")
    print(f"✓ Consolidated CSV saved to: {output_csv}")


# =============================================================================
# MAIN PIPELINE ORCHESTRATION
# =============================================================================


def stage_4_sample_test_data(train_csv_path, test_csv_path,
                             sample_percentage=0.01):
    """
    Pipeline Stage 4: Sample test data and maintain train/test split.

    This stage randomly samples a percentage of the training data and moves
    it to the test set. This ensures proper separation between training and
    testing datasets. The sampled data is removed from the training set and
    added to the test set, with deduplication applied to the test set.

    Process:
        1. Load consolidated training CSV from Stage 3
        2. Randomly sample specified percentage of rows
        3. Remove sampled rows from training data
        4. Save updated training CSV
        5. Load existing test CSV (if exists) or create new DataFrame
        6. Append sampled rows to test data
        7. Deduplicate test data (keep first occurrence)
        8. Save updated test CSV

    Deduplication Strategy (Test Set):
        - If an image_name appears multiple times in test set, keep the
          FIRST occurrence
        - This preserves the original test samples when new data is added

    Args:
        train_csv_path (str): Path to training CSV (consolidated from Stage 3).
        test_csv_path (str): Path to test CSV file (existing or new).
        sample_percentage (float): Fraction of training data to sample for
                                  test set (default: 0.01 = 1%).

    Returns:
        None: Function writes directly to CSV files.

    Raises:
        FileNotFoundError: If train_csv_path doesn't exist.
        ValueError: If sample_percentage is not between 0 and 1.

    Example:
        >>> stage_4_sample_test_data(
        ...     "final_output.csv",
        ...     "test_data.csv",
        ...     sample_percentage=0.01
        ... )
        Loading training data from: final_output.csv
        Sampling 95 records (1.0%) for test set...
        Removing samples from training data...
        Loading existing test data from: test_data.csv
        Merging with test set...
        Deduplicating test set (keeping first)...
        ✓ Training set: 9448 records
        ✓ Test set: 245 unique records
    """
    print("\n" + "=" * 70)
    print("STAGE 4: Sampling test data from training set")
    print("=" * 70)

    # Validate sample percentage
    if not 0 < sample_percentage < 1:
        raise ValueError(
            f"sample_percentage must be between 0 and 1, "
            f"got {sample_percentage}"
        )

    # Load training data
    print(f"Loading training data from: {train_csv_path}")
    if not os.path.isfile(train_csv_path):
        raise FileNotFoundError(
            f"Training CSV file not found: {train_csv_path}"
        )
    df_train = pd.read_csv(train_csv_path, encoding='utf-8-sig')
    initial_train_count = len(df_train)
    print(f"✓ Loaded {initial_train_count} training records")

    # Calculate sample size
    sample_size = min(45, int(len(df_train) * sample_percentage))
    print(f"\nSampling {sample_size} records "
          f"({sample_percentage * 100:.1f}%) for test set...")

    # Randomly sample rows for test set
    df_test_samples = df_train.sample(
        n=sample_size,
        random_state=42  # For reproducibility
    )

    # Remove sampled rows from training data
    print("Removing sampled records from training data...")
    df_train_updated = df_train.drop(df_test_samples.index)
    print(f"✓ Training set reduced to {len(df_train_updated)} records")

    # Save updated training CSV
    print(f"Saving updated training data to: {train_csv_path}")
    df_train_updated.to_csv(train_csv_path, index=False, encoding='utf-8-sig')

    # Load existing test data if it exists
    if os.path.isfile(test_csv_path):
        print(f"\nLoading existing test data from: {test_csv_path}")
        df_test_existing = pd.read_csv(test_csv_path, encoding='utf-8-sig')

        print(f"✓ Loaded {len(df_test_existing)} existing test records")

        # Merge with new test samples
        print("Merging with new test samples...")
        df_test_combined = pd.concat(
            [df_test_existing, df_test_samples],
            ignore_index=True
        )
        print(f"✓ Combined test set: {len(df_test_combined)} records")
    else:
        print(f"\nNo existing test file found at: {test_csv_path}")
        print("Creating new test dataset...")
        df_test_combined = df_test_samples

    # Deduplicate test set (keep first occurrence)
    print("\nDeduplicating test set (keeping first occurrence)...")
    initial_test_count = len(df_test_combined)
    df_test_final = df_test_combined.drop_duplicates(
        subset="image_name",
        keep="first"
    )
    duplicates_removed = initial_test_count - len(df_test_final)
    print(f"✓ Removed {duplicates_removed} duplicate records from test set")

    # Save final test CSV
    print(f"Saving test data to: {test_csv_path}")
    df_test_final.to_csv(test_csv_path, index=False, encoding='utf-8-sig')

    # Print final summary
    print("\n" + "-" * 70)
    print("STAGE 4 SUMMARY:")
    print("-" * 70)
    print(f"✓ Training set: {len(df_train_updated)} records")
    print(f"✓ Test set: {len(df_test_final)} unique records")
    print(f"✓ Sampled {sample_size} new records for testing")
    print(f"✓ Training CSV saved to: {train_csv_path}")
    print(f"✓ Test CSV saved to: {test_csv_path}")



def run_pipeline(root_path, patterns_excel_path, yolo_model_path,
                 output_csv, existing_csv_path=None, test_csv_path=None,
                 use_cuda=False,
                 image_suffix=IMAGE_SUFFIX):
    """
    Execute the complete image data processing pipeline.

    This is the main orchestration function that runs all stages
    sequentially, managing intermediate files and providing comprehensive
    progress reporting.

    Pipeline Flow:
        Stage 1: Folder structure → CSV with pattern matching
        Stage 2: CSV + YOLO model → CSV with document predictions
        Stage 3: New CSV + Existing CSV → Consolidated final CSV
        Stage 4: Sample test data from final CSV (optional)

    Intermediate Files:
        - <output>_stage1.csv: After folder scanning and pattern matching
        - <output>_stage2.csv: After YOLO predictions added

    Final Output:
        - <output>: Consolidated, deduplicated CSV with all columns

    Args:
        root_path (str): Root folder containing data structure
                        (must have 02_intermediate, 03_* subfolders).
        patterns_excel_path (str): Path to patterns.xlsx file for address
                                  matching. Can be None to skip pattern
                                  matching.
        yolo_model_path (str): Path to YOLO .pt classification model.
        output_csv (str): Path for final output CSV file.
        existing_csv_path (str, optional): Path to existing historical data
                                          to merge with. None = new data only.
        test_csv_path (str, optional): Path to test CSV file. If provided,
                                      1% of final data will be sampled and
                                      moved to test set. None = skip test
                                      sampling.
        use_cuda (bool): Whether to use GPU for YOLO inference.
        image_suffix (str): Suffix of preprocessed images for YOLO
                           (default: "_receiver_box_prep").

    Returns:
        None: Function manages file I/O and prints progress to console.

    Example:
        >>> run_pipeline(
        ...     root_path="/data/",
        ...     patterns_excel_path="/data/patterns.xlsx",
        ...     yolo_model_path="/models/classifier.pt",
        ...     output_csv="/output/final_data.csv",
        ...     existing_csv_path="/output/historical_data.csv",
        ...     test_csv_path="/output/test_data.csv",
        ...     use_cuda=True
        ... )
        ======================================================================
        IMAGE DATA PROCESSING PIPELINE
        ======================================================================
        ...
        ✓ Pipeline completed successfully!
        ✓ Final output: /output/final_data.csv
        ✓ Test output: /output/test_data.csv
    """
    print("\n" + "=" * 70)
    print("IMAGE DATA PROCESSING PIPELINE")
    print("=" * 70)
    print(f"Root data path: {root_path}")
    print(f"Patterns file: {patterns_excel_path or 'None (skipping)'}")
    print(f"YOLO model: {yolo_model_path}")
    print(f"Final output: {output_csv}")
    if existing_csv_path:
        print(f"Existing data: {existing_csv_path}")
    else:
        print("Existing data: None (new dataset)")
    if test_csv_path:
        print(f"Test data: {test_csv_path}")
    else:
        print("Test data: None (no test sampling)")
    print(f"GPU acceleration: {'Enabled' if use_cuda else 'Disabled'}")
    print("=" * 70)

    # Generate intermediate file paths
    base_name = os.path.splitext(output_csv)[0]
    stage1_csv = f"{base_name}_stage1.csv"
    stage2_csv = f"{base_name}_stage2.csv"

    try:
        # =====================================================================
        # STAGE 1: Generate CSV from folders with pattern matching
        # =====================================================================
        stage_1_generate_csv_with_patterns(
            root_path=root_path,
            output_csv=stage1_csv,
            patterns_excel_path=patterns_excel_path
        )

        # =====================================================================
        # STAGE 2: Add YOLO document type predictions
        # =====================================================================
        stage_2_add_yolo_predictions(
            root_path=root_path,
            csv_path=stage1_csv,
            model_path=yolo_model_path,
            output_csv=stage2_csv,
            use_cuda=use_cuda,
            image_suffix=image_suffix
        )

        # =====================================================================
        # STAGE 3: Consolidate with existing data and deduplicate
        # =====================================================================
        stage_3_consolidate_data(
            new_csv_path=stage2_csv,
            existing_csv_path=existing_csv_path,
            output_csv=output_csv
        )

        # =====================================================================
        # STAGE 4: Sample test data (optional)
        # =====================================================================
        if test_csv_path:
            stage_4_sample_test_data(
                train_csv_path=output_csv,
                test_csv_path=test_csv_path,
                sample_percentage=0.01
            )

        # =====================================================================
        # PIPELINE COMPLETION
        # =====================================================================
        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"✓ Final output saved to: {output_csv}")
        if test_csv_path:
            print(f"✓ Test data saved to: {test_csv_path}")
        print(f"\nIntermediate files:")
        print(f"  - Stage 1 output: {stage1_csv}")
        print(f"  - Stage 2 output: {stage2_csv}")
        print("\nYou can safely delete intermediate files if not needed.")
        print("=" * 70 + "\n")

    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ PIPELINE FAILED")
        print("=" * 70)
        print(f"Error: {str(e)}")
        print("\nPlease check the error message above and ensure:")
        print("  1. All input paths are correct")
        print("  2. Required folders exist in root_path")
        print("  3. Excel and model files are valid")
        print("  4. You have write permissions for output location")
        print("=" * 70 + "\n")
        raise


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """
    Parse command-line arguments and execute the pipeline.

    This function provides a user-friendly CLI for running the complete
    pipeline with customizable parameters.
    """
    parser = argparse.ArgumentParser(
        description="Image Data Processing Pipeline - Complete workflow for "
                    "extracting, matching, classifying, and consolidating "
                    "image metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with all features
  python pipeline.py \\
    --root /data/ \\
    --patterns /data/patterns.xlsx \\
    --model /models/classifier.pt \\
    --output /output/final_data.csv

  # With GPU acceleration and existing data merge
  python pipeline.py \\
    --root /data/ \\
    --patterns /data/patterns.xlsx \\
    --model /models/classifier.pt \\
    --output /output/final_data.csv \\
    --existing /output/historical_data.csv \\
    --use-cuda

  # Without pattern matching
  python pipeline.py \\
    --root /data/ \\
    --model /models/classifier.pt \\
    --output /output/final_data.csv

For more information, see the script documentation.
        """
    )

    # Required arguments
    parser.add_argument(
        "--root",
        required=True,
        help="Path to root data folder containing 02_intermediate and 03_* "
             "subfolders with text files."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to YOLO .pt classification model for document type "
             "prediction."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for final output CSV file (consolidated results)."
    )

    # Optional arguments
    parser.add_argument(
        "--patterns",
        required=False,
        default=None,
        help="Path to patterns.xlsx file for address pattern matching. "
             "If not provided, pattern matching is skipped and the "
             "'matched_rules' column will be empty."
    )
    parser.add_argument(
        "--test",
        required=False,
        default=None,
        help="Path to test CSV file. If provided, 1%% of the final "
             "consolidated data will be randomly sampled and moved to "
             "the test set. The test set will be deduplicated (keeping "
             "first occurrence). If not provided, no test sampling is "
             "performed."
    )
    parser.add_argument(
        "--existing",
        required=False,
        default=None,
        help="Path to existing CSV file with historical data to merge with. "
             "New data takes precedence for duplicate image_name entries. "
             "If not provided, only new data is used."
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Enable GPU acceleration for YOLO inference if CUDA is "
             "available. Significantly speeds up document classification."
    )
    parser.add_argument(
        "--suffix",
        required=False,
        default=IMAGE_SUFFIX,
        help=f"Filename suffix for preprocessed images to classify "
             f"(default: '{IMAGE_SUFFIX}'). Images must have this suffix "
             f"before the file extension."
    )

    args = parser.parse_args()

    # Execute pipeline with provided arguments
    run_pipeline(
        root_path=args.root,
        patterns_excel_path=args.patterns,
        yolo_model_path=args.model,
        output_csv=args.output,
        existing_csv_path=args.existing,
        test_csv_path=args.test,
        use_cuda=args.use_cuda,
        image_suffix=args.suffix
    )

# python postprocess.py --root D:\Tensor\kedro\data-pipeline\data --patterns D:\Tensor\kedro\data-pipeline\src\data_pipeline\pipelines\utils\patterns.xlsx --model D:\Tensor\kedro\data-pipeline\models\hand_type_cls\hand_type_cls_1.pt --output D:\Tensor\kedro\data-pipeline\records\out_test.csv --existing D:\Tensor\kedro\data-pipeline\records\address_region_handtype_cls_full.csv --test D:\Tensor\kedro\data-pipeline\records\address_region_handtype_cls_test.csv --use-cuda

if __name__ == "__main__":
    main()