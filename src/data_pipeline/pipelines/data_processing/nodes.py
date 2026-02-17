"""Node functions for data processing pipeline."""

"""Node functions for data processing pipeline."""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Any, Dict
from ultralytics import YOLO
from ..utils.checkpoint_manager import load_checkpoint, save_checkpoint, get_pending_images
from ..utils.read_text import read_text
from ..utils.read_postcode import read_postcode
from ..utils.read_postcode_gilan import read_postcode_gilan
from ..utils.read_postcode_sari import read_postcode_sari
from ..utils.load_paddle_model import load_paddle_model
from ..utils.detect_region_by_words import detect_region_by_words
from ..utils.detect_region_by_words_gilan import detect_region_by_words_gilan
from ..utils.detect_region_by_words_sari import detect_region_by_words_sari
from ..utils.detect_obb_coordinates import detect_obb_coordinates
from ..utils.preprocesses import denoise_normalize, apply_adaptive_mean_threshold
from ..utils.correct_angle import predict_angle_to_rotate, rotate_image, pad_to_square
from ..utils.crop_from_coordinates import crop_from_coordinates, predict_classification, pretty_oof_preprocess, is_top_left



def load_and_crop_parcel(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node that loads all images from a folder and returns cropped parcel images.
    Processes images in batches with checkpoint support.
    
    Args:
        parameters: Dictionary containing:
            - 'raw_images_folder': Path to folder containing raw images
            - 'yolo_detect_parcel_model_path': Path to the YOLO model file
            - 'yolo_blur_parcel_classification_model_path': Path to blur classification model
            - 'batch_size': Number of images to process per batch (default: 200)
            - 'checkpoint_dir': Directory for checkpoint files (default: 'data/00_checkpoints/')
            - 'enable_checkpointing': Whether to use checkpointing (default: True)
    
    Returns:
        Dict[str, Any]: Dictionary mapping filename to cropped PIL.Image or error code (int)
    """
    raw_images_folder = parameters["raw_images_folder"]
    batch_size = parameters.get("batch_size", 200)
    checkpoint_dir = parameters.get("checkpoint_dir", "data/00_checkpoints/")
    enable_checkpointing = parameters.get("enable_checkpointing", True)
    
    results = {}
    
    # Load models once (outside the loop for efficiency)
    parcel_detection_model = YOLO(parameters["yolo_detect_parcel_model_path"])
    blur_cls_model = YOLO(parameters["yolo_blur_parcel_classification_model_path"])
    
    # Get all image files
    all_image_files = [f for f in os.listdir(raw_images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Load checkpoint if enabled
    processed_images = set()
    if enable_checkpointing:
        processed_images = load_checkpoint(checkpoint_dir, "node1_processed.json")
        image_files = get_pending_images(all_image_files, processed_images)
    else:
        image_files = all_image_files
    
    print(f"📦 Total images: {len(all_image_files)}")
    print(f"🔄 To process: {len(image_files)} images")
    print(f"📊 Batch size: {batch_size}")
    
    # Process images in batches
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"📦 Processing Batch {batch_num + 1}/{total_batches}")
        print(f"📋 Images {start_idx + 1}-{end_idx} of {len(image_files)}")
        print(f"{'='*60}\n")
        
        batch_processed = set()
        
        for image_file in batch_files:
            filename = os.path.splitext(image_file)[0]
            image_path = os.path.join(raw_images_folder, image_file)
            
            # Create output directory for this image
            output_dir = os.path.join("data/02_intermediate", filename)
            os.makedirs(output_dir, exist_ok=True)
            
            # Read the image
            image = cv2.imread(image_path)

            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Ensure dtype is uint8
            image = image.astype('uint8')

            
            if image is None:
                print(f"Failed to load image: {filename}")
                results[filename] = -10  # New error code for failed to load
                _save_error_file(output_dir, filename, -10, "Failed to load image")
                batch_processed.add(filename)
                continue
            
            # Detect the parcel coordinate from the raw image
            full_box, box = detect_obb_coordinates(
                image, parcel_detection_model, save_path=None, name="parcel"
            )
            
            # If parcel doesn't get detected, return -5 (rejected box)
            if full_box is None:
                print(f"No parcel detected: {filename}")
                results[filename] = -5
                _save_error_file(output_dir, filename, -5, "No parcel detected")
                batch_processed.add(filename)
                continue
            if box is None:
            # Crop the parcel from raw image
                cropped_parcel = crop_from_coordinates(
                    image, full_box, save_path=None, name="parcel"
                )
            else:
                cropped_parcel = crop_from_coordinates(
                    image, box, save_path=None, name="parcel"
                )
            
            # Check if the cropped parcel is blurry
            _, label = predict_classification(cropped_parcel, blur_cls_model)
            if label == "blur":
                print(f"Blurry parcel: {filename}")
                results[filename] = -4
                _save_error_file(output_dir, filename, -4, "Blurry parcel")
                batch_processed.add(filename)
                continue
            
            # Save the cropped parcel
            cropped_output_path = os.path.join(output_dir, f"{filename}_cropped.png")
            cv2.imwrite(cropped_output_path, cropped_parcel)
            
            # Convert to PIL and store
            pil_cropped_parcel = Image.fromarray(cropped_parcel.astype('uint8'))
            results[filename] = pil_cropped_parcel
            print(f"✓ Processed: {filename}")
            
            # Track processed image
            batch_processed.add(filename)
        
        # Save checkpoint after each batch
        if enable_checkpointing:
            processed_images.update(batch_processed)
            save_checkpoint(checkpoint_dir, processed_images, "node1_processed.json")
            print(f"✅ Batch {batch_num + 1}/{total_batches} complete - Checkpoint saved\n")
    
    print(f"\n{'='*60}")
    print(f"✅ All batches processed! Total images: {len(results)}")
    print(f"{'='*60}\n")
    
    return results


def _save_error_file(output_dir: str, filename: str, error_code: int, error_message: str):
    """Helper function to save error information to a file."""
    error_file_path = os.path.join(output_dir, f"{filename}_error.txt")
    with open(error_file_path, 'w') as f:
        f.write(f"Error Code: {error_code}\n")
        f.write(f"Error Message: {error_message}\n")


def rotate_and_preprocess_parcel(
    cropped_parcel_images: Dict[str, Any], 
    parameters: Dict[str, Any]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Node that takes cropped parcel images and returns two processed versions.
    Processes images in batches for memory efficiency.
    
    Args:
        cropped_parcel_images: Dict mapping filename to PIL Image or error code
        parameters: Configuration parameters including:
            - batch_size: Number of images to process per batch (default: 200)
    
    Returns:
        tuple: (rotated_images_dict, rotated_preprocessed_images_dict)
    """
    batch_size = parameters.get("batch_size", 200)
    
    rotated_images = {}
    rotated_preprocessed_images = {}
    
    # Load model once
    angle_correction_model = YOLO(parameters["yolo_angle_correction_model_path"])
    
    # Get all filenames
    all_filenames = list(cropped_parcel_images.keys())
    
    print(f"\n{'='*60}")
    print(f"🔄 NODE 2: Rotate and Preprocess Parcel")
    print(f"📦 Total images: {len(all_filenames)}")
    print(f"📊 Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Process images in batches
    total_batches = (len(all_filenames) + batch_size - 1) // batch_size if all_filenames else 0
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(all_filenames))
        batch_filenames = all_filenames[start_idx:end_idx]
        
        print(f"📦 Node 2 - Batch {batch_num + 1}/{total_batches} (images {start_idx + 1}-{end_idx})")
        
        for filename in batch_filenames:
            cropped_image = cropped_parcel_images[filename]
            
            # Skip if previous step had an error
            if isinstance(cropped_image, int) and cropped_image < 0:
                rotated_images[filename] = cropped_image
                rotated_preprocessed_images[filename] = cropped_image
                continue
            
            output_dir = os.path.join("data/02_intermediate", filename)
            os.makedirs(output_dir, exist_ok=True)

            # Convert PIL to numpy and normalize
            cropped_parcel_image = np.array(cropped_image)
            cropped_parcel_normalized = cv2.normalize(
                cropped_parcel_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            
            # Apply preprocessing
            cropped_parcel_preprocessed = pretty_oof_preprocess(
                cropped_parcel_normalized, save_path=None
            )
            
            # Predict rotation angle
            angle_to_rotate = predict_angle_to_rotate(
                cropped_parcel_preprocessed, angle_correction_model, device="cuda"
            )
            
            # Rotate both versions
            corrected_angle_parcel_preprocessed = rotate_image(
                cropped_parcel_preprocessed, angle_to_rotate
            )
            corrected_angle_parcel_normalized = rotate_image(
                cropped_parcel_normalized, angle_to_rotate
            )
            
            # Save rotated images
            rotated_norm_path = os.path.join(output_dir, f"{filename}_rotated.png")
            rotated_prep_path = os.path.join(output_dir, f"{filename}_rotated_preprocessed.png")

            cv2.imwrite(rotated_norm_path, corrected_angle_parcel_normalized)
            cv2.imwrite(rotated_prep_path, corrected_angle_parcel_preprocessed)
            
            # Convert to PIL and store
            pil_corrected_norm = Image.fromarray(corrected_angle_parcel_normalized.astype('uint8'))
            pil_corrected_prep = Image.fromarray(corrected_angle_parcel_preprocessed.astype('uint8'))
            
            rotated_images[filename] = pil_corrected_norm
            rotated_preprocessed_images[filename] = pil_corrected_prep
            print(f"  ✓ Rotated: {filename}")
        
        print(f"✅ Batch {batch_num + 1}/{total_batches} complete\n")
    
    print(f"{'='*60}")
    print(f"✅ Node 2 Complete! Total: {len(rotated_images)} images")
    print(f"{'='*60}\n")
    
    return rotated_images, rotated_preprocessed_images


def receiver_postcode_detection(
    rotated_images: Dict[str, Any],
    rotated_preprocessed_images: Dict[str, Any],
    parameters: Dict[str, Any]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Node that detects and crops receiver and postcode boxes.
    Processes images in batches for memory efficiency.
    
    Args:
        rotated_images: Dict of rotated normalized images
        rotated_preprocessed_images: Dict of rotated preprocessed images
        parameters: Configuration parameters including:
            - batch_size: Number of images to process per batch (default: 200)
    
    Returns:
        tuple: (postcode_boxes_dict, receiver_boxes_dict)
    """
    batch_size = parameters.get("batch_size", 200)
    
    postcode_boxes = {}
    receiver_boxes = {}
    
    # Load models once
    receiver_detection_model = YOLO(parameters["yolo_receiver_detection_model_path"])
    blury_postcode_model = YOLO(parameters["yolo_blury_postcode_model_path"])
    
    # Get all filenames
    all_filenames = list(rotated_images.keys())
    
    print(f"\n{'='*60}")
    print(f"🔄 NODE 3: Receiver & Postcode Detection")
    print(f"📦 Total images: {len(all_filenames)}")
    print(f"📊 Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Process images in batches
    total_batches = (len(all_filenames) + batch_size - 1) // batch_size if all_filenames else 0
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(all_filenames))
        batch_filenames = all_filenames[start_idx:end_idx]
        
        print(f"📦 Node 3 - Batch {batch_num + 1}/{total_batches} (images {start_idx + 1}-{end_idx})")
        
        for filename in batch_filenames:
            # Skip if previous step had an error
            if isinstance(rotated_images[filename], int) and rotated_images[filename] < 0:
                postcode_boxes[filename] = rotated_images[filename]
                receiver_boxes[filename] = rotated_images[filename]
                continue
            
            output_dir = os.path.join("data/02_intermediate", filename)
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert to numpy
            rotated_image = np.array(rotated_images[filename])
            rotated_preprocessed_image = np.array(rotated_preprocessed_images[filename])
            
            # Detect boxes
            receiver_box, postcode_box = detect_obb_coordinates(
                rotated_preprocessed_image,
                receiver_detection_model,
                save_path=None,
                name="receiver",
            )
            
            # If receiver box not detected, error
            if receiver_box is None:
                print(f"  ⚠ No receiver detected: {filename}")
                postcode_boxes[filename] = -3
                receiver_boxes[filename] = -3
                _save_error_file(output_dir, filename, -3, "No receiver box detected")
                continue
            
            # Process postcode if detected and valid location
            pil_postcode = None
            if postcode_box is not None and not is_top_left(rotated_preprocessed_image.shape, postcode_box):
                # Crop postcode
                cropped_postcode = crop_from_coordinates(
                    rotated_image, postcode_box, save_path=None, name="postcode"
                )

                # Check if blurry
                padded_postcode = pad_to_square(cropped_postcode)
                _, label = predict_classification(padded_postcode, blury_postcode_model, conf_thresh=0.95)

                if label == "blur":
                    print(f"  ⚠ Blurry postcode: {filename}")
                    postcode_boxes[filename] = -2
                    receiver_boxes[filename] = -2
                    _save_error_file(output_dir, filename, -2, "Blurry postcode")
                    # Note: Not continuing here - still process receiver
                
                # Denoise postcode
                denoised_postcode = denoise_normalize(cropped_postcode)
                postcode_path = os.path.join(output_dir, f"{filename}_postcode_box_prep.png")
                cv2.imwrite(postcode_path, denoised_postcode)
                pil_postcode = Image.fromarray(denoised_postcode.astype('uint8'))

            # Crop and preprocess receiver
            cropped_receiver = crop_from_coordinates(
                rotated_image, receiver_box, save_path=None, name="receiver"
            )

            cropped_receiver_preprocessed = apply_adaptive_mean_threshold(
                cropped_receiver, save_path=None
            )
            
            # Save receiver box
            receiver_path_raw = os.path.join(output_dir, f"{filename}_receiver_box_raw.png")
            receiver_path = os.path.join(output_dir, f"{filename}_receiver_box_prep.png")
            
            # Save Images
            cv2.imwrite(receiver_path, cropped_receiver_preprocessed)

            # cropped_receiver = cv2.normalize(cropped_receiver, None, 0, 255, cv2.NORM_MINMAX)
            # cv2.imwrite(receiver_path_raw, cropped_receiver)

            pil_receiver = Image.fromarray(cropped_receiver_preprocessed.astype('uint8'))
            
            postcode_boxes[filename] = pil_postcode
            receiver_boxes[filename] = pil_receiver
            print(f"  ✓ Detected boxes: {filename}")
        
        print(f"✅ Batch {batch_num + 1}/{total_batches} complete\n")
    
    print(f"{'='*60}")
    print(f"✅ Node 3 Complete! Total: {len(receiver_boxes)} images")
    print(f"{'='*60}\n")
    
    return postcode_boxes, receiver_boxes


def read_classify_postcode(
    postcode_boxes: Dict[str, Any],
    parameters: Dict[str, Any]
) -> Dict[str, str]:
    """
    Node that reads postcode digits and predicts region.
    Processes images in batches for memory efficiency.
    
    Args:
        postcode_boxes: Dict of preprocessed postcode boxes
        parameters: Configuration parameters including:
            - batch_size: Number of images to process per batch (default: 200)
    
    Returns:
        Dict[str, str]: Dictionary mapping filename to predicted region (as string)
    """
    batch_size = parameters.get("batch_size", 200)
    
    results = {}
    
    # Load model once
    digit_detection_model = YOLO(parameters["yolo_digit_detection_model_path"])
    
    # Get all filenames
    all_filenames = list(postcode_boxes.keys())
    
    print(f"\n{'='*60}")
    print(f"🔄 NODE 4: Read & Classify Postcode")
    print(f"📦 Total images: {len(all_filenames)}")
    print(f"📊 Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Process images in batches
    total_batches = (len(all_filenames) + batch_size - 1) // batch_size if all_filenames else 0
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(all_filenames))
        batch_filenames = all_filenames[start_idx:end_idx]
        
        print(f"📦 Node 4 - Batch {batch_num + 1}/{total_batches} (images {start_idx + 1}-{end_idx})")
        
        for filename in batch_filenames:
            postcode_image = postcode_boxes[filename]
            
            # Skip if previous step had error or no postcode
            if isinstance(postcode_image, int) and postcode_image < 0:
                results[filename] = str(postcode_image)
                continue
            
            if postcode_image is None:
                results[filename] = "-1"  # No postcode detected
                continue
            
            output_dir = os.path.join("data/03_postcode_regions", filename)
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert to numpy
            postcode_array = np.array(postcode_image)
            
            # Read postcode
            postcodes_region, status, full_postcode = read_postcode(
                postcode_array, digit_detection_model, save_path=None
            )
            
            # Save result
            result_path = os.path.join(output_dir, f"{filename}_postcode_region.txt")
            with open(result_path, 'w') as f:
                f.write(str(postcodes_region))
                f.write('\n')
                f.write(str(full_postcode) if full_postcode is not None else "None")
                f.write('\n')
                f.write(status)
            
            results[filename] = str(postcodes_region)
            print(f"  ✓ Postcode region for {filename}: {postcodes_region}")
        
        print(f"✅ Batch {batch_num + 1}/{total_batches} complete\n")
    
    print(f"{'='*60}")
    print(f"✅ Node 4 Complete! Total: {len(results)} images")
    print(f"{'='*60}\n")
    
    return results


def read_classify_postcode_gilan(
    postcode_boxes: Dict[str, Any],
    parameters: Dict[str, Any]
) -> Dict[str, str]:
    """
    Node that reads postcode digits and predicts region.
    Processes images in batches for memory efficiency.
    
    Args:
        postcode_boxes: Dict of preprocessed postcode boxes
        parameters: Configuration parameters including:
            - batch_size: Number of images to process per batch (default: 200)
    
    Returns:
        Dict[str, str]: Dictionary mapping filename to predicted region (as string)
    """
    batch_size = parameters.get("batch_size", 200)
    
    results = {}
    
    # Load model once
    digit_detection_model = YOLO(parameters["yolo_digit_detection_model_path"])
    
    # Get all filenames
    all_filenames = list(postcode_boxes.keys())
    
    print(f"\n{'='*60}")
    print(f"🔄 NODE 4: Read & Classify Postcode")
    print(f"📦 Total images: {len(all_filenames)}")
    print(f"📊 Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Process images in batches
    total_batches = (len(all_filenames) + batch_size - 1) // batch_size if all_filenames else 0
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(all_filenames))
        batch_filenames = all_filenames[start_idx:end_idx]
        
        print(f"📦 Node 4 - Batch {batch_num + 1}/{total_batches} (images {start_idx + 1}-{end_idx})")
        
        for filename in batch_filenames:
            postcode_image = postcode_boxes[filename]
            
            # Skip if previous step had error or no postcode
            if isinstance(postcode_image, int) and postcode_image < 0:
                results[filename] = str(postcode_image)
                continue
            
            if postcode_image is None:
                results[filename] = "-1"  # No postcode detected
                continue
            
            output_dir = os.path.join("data/03_postcode_regions", filename)
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert to numpy
            postcode_array = np.array(postcode_image)
            
            # Read postcode
            postcodes_region, status, full_postcode = read_postcode_gilan(
                postcode_array, digit_detection_model, save_path=None
            )
            
            # Save result
            result_path = os.path.join(output_dir, f"{filename}_postcode_region.txt")
            with open(result_path, 'w') as f:
                f.write(str(postcodes_region))
                f.write('\n')
                f.write(str(full_postcode) if full_postcode is not None else "None")
                f.write('\n')
                f.write(status)
            
            results[filename] = str(postcodes_region)
            print(f"  ✓ Postcode region for {filename}: {postcodes_region}")
        
        print(f"✅ Batch {batch_num + 1}/{total_batches} complete\n")
    
    print(f"{'='*60}")
    print(f"✅ Node 4 Complete! Total: {len(results)} images")
    print(f"{'='*60}\n")
    
    return results


def read_classify_postcode_sari(
    postcode_boxes: Dict[str, Any],
    parameters: Dict[str, Any]
) -> Dict[str, str]:
    """
    Node that reads postcode digits and predicts region.
    Processes images in batches for memory efficiency.
    
    Args:
        postcode_boxes: Dict of preprocessed postcode boxes
        parameters: Configuration parameters including:
            - batch_size: Number of images to process per batch (default: 200)
    
    Returns:
        Dict[str, str]: Dictionary mapping filename to predicted region (as string)
    """
    batch_size = parameters.get("batch_size", 200)
    
    results = {}
    
    # Load model once
    digit_detection_model = YOLO(parameters["yolo_digit_detection_model_path"])
    
    # Get all filenames
    all_filenames = list(postcode_boxes.keys())
    
    print(f"\n{'='*60}")
    print(f"🔄 NODE 4: Read & Classify Postcode")
    print(f"📦 Total images: {len(all_filenames)}")
    print(f"📊 Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Process images in batches
    total_batches = (len(all_filenames) + batch_size - 1) // batch_size if all_filenames else 0
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(all_filenames))
        batch_filenames = all_filenames[start_idx:end_idx]
        
        print(f"📦 Node 4 - Batch {batch_num + 1}/{total_batches} (images {start_idx + 1}-{end_idx})")
        
        for filename in batch_filenames:
            postcode_image = postcode_boxes[filename]
            
            # Skip if previous step had error or no postcode
            if isinstance(postcode_image, int) and postcode_image < 0:
                results[filename] = str(postcode_image)
                continue
            
            if postcode_image is None:
                results[filename] = "-1"  # No postcode detected
                continue
            
            output_dir = os.path.join("data/03_postcode_regions", filename)
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert to numpy
            postcode_array = np.array(postcode_image)
            
            # Read postcode
            postcodes_region, status, full_postcode = read_postcode_sari(
                postcode_array, digit_detection_model, save_path=None
            )
            
            # Save result
            result_path = os.path.join(output_dir, f"{filename}_postcode_region.txt")
            with open(result_path, 'w') as f:
                f.write(str(postcodes_region))
                f.write('\n')
                f.write(str(full_postcode) if full_postcode is not None else "None")
                f.write('\n')
                f.write(status)
            
            results[filename] = str(postcodes_region)
            print(f"  ✓ Postcode region for {filename}: {postcodes_region}")
        
        print(f"✅ Batch {batch_num + 1}/{total_batches} complete\n")
    
    print(f"{'='*60}")
    print(f"✅ Node 4 Complete! Total: {len(results)} images")
    print(f"{'='*60}\n")
    
    return results


def read_address_ocr(
    receiver_boxes: Dict[str, Any],
    parameters: Dict[str, Any]
) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Node that reads address using OCR and predicts region.
    Processes images in batches for memory efficiency.
    
    Args:
        receiver_boxes: Dict of preprocessed receiver boxes
        parameters: Configuration parameters including:
            - batch_size: Number of images to process per batch (default: 200)
    
    Returns:
        tuple: (address_texts_dict, predicted_regions_dict)
    """
    batch_size = parameters.get("batch_size", 200)
    
    address_texts = {}
    predicted_regions = {}
    
    # Load models once
    print("\n🔄 Loading OCR models...")
    word_digit_hbb = YOLO(parameters["yolo_detect_word_digit_hbb_model_path"])
    real_ocr_model = load_paddle_model(parameters["real_ocr_model_path"])
    print("✅ OCR models loaded\n")
    
    # Get all filenames
    all_filenames = list(receiver_boxes.keys())
    
    print(f"{'='*60}")
    print(f"🔄 NODE 5: Read Address OCR")
    print(f"📦 Total images: {len(all_filenames)}")
    print(f"📊 Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Process images in batches
    total_batches = (len(all_filenames) + batch_size - 1) // batch_size if all_filenames else 0
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(all_filenames))
        batch_filenames = all_filenames[start_idx:end_idx]
        
        print(f"📦 Node 5 - Batch {batch_num + 1}/{total_batches} (images {start_idx + 1}-{end_idx})")
        
        for filename in batch_filenames:
            receiver_image = receiver_boxes[filename]
            
            # Skip if previous step had error
            if isinstance(receiver_image, int) and receiver_image < 0:
                address_texts[filename] = f"Error: {receiver_image}"
                predicted_regions[filename] = str(receiver_image)
                continue
            
            # Create output directories
            address_output_dir = os.path.join("data/03_read_address", filename)
            region_output_dir = os.path.join("data/03_address_region", filename)
            os.makedirs(address_output_dir, exist_ok=True)
            os.makedirs(region_output_dir, exist_ok=True)
            
            # Convert to numpy
            receiver_array = np.array(receiver_image)
            
            # Read address (returns both corrected and raw versions)
            address_string_corrected, address_string_raw = read_text(
                receiver_array,
                word_digit_hbb,
                real_ocr_model,
                save_path=None,
            )
            
            # Predict region from address (use corrected version)
            texts_region = str(detect_region_by_words(address_string_corrected))
            
            # Save results (both versions in one file)
            address_path = os.path.join(address_output_dir, f"{filename}_read_address.txt")
            with open(address_path, 'w', encoding='utf-8') as f:
                f.write(f"{address_string_raw}\n")
                f.write(f"{address_string_corrected}\n")
            
            region_path = os.path.join(region_output_dir, f"{filename}_address_region.txt")
            with open(region_path, 'w') as f:
                f.write(texts_region)
            
            address_texts[filename] = address_string_corrected
            predicted_regions[filename] = texts_region
            print(f"  ✓ Address region for {filename}: {texts_region}")
        
        print(f"✅ Batch {batch_num + 1}/{total_batches} complete\n")
    
    print(f"{'='*60}")
    print(f"✅ Node 5 Complete! Total: {len(address_texts)} images")
    print(f"{'='*60}\n")
    
    return address_texts, predicted_regions



def read_address_ocr_gilan(
    receiver_boxes: Dict[str, Any],
    parameters: Dict[str, Any]
) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Node that reads address using OCR and predicts region.
    Processes images in batches for memory efficiency.
    
    Args:
        receiver_boxes: Dict of preprocessed receiver boxes
        parameters: Configuration parameters including:
            - batch_size: Number of images to process per batch (default: 200)
    
    Returns:
        tuple: (address_texts_dict, predicted_regions_dict)
    """
    batch_size = parameters.get("batch_size", 200)
    
    address_texts = {}
    predicted_regions = {}
    
    # Load models once
    print("\n🔄 Loading OCR models...")
    word_digit_hbb = YOLO(parameters["yolo_detect_word_digit_hbb_model_path"])
    real_ocr_model = load_paddle_model(parameters["real_ocr_model_path"])
    print("✅ OCR models loaded\n")
    
    # Get all filenames
    all_filenames = list(receiver_boxes.keys())
    
    print(f"{'='*60}")
    print(f"🔄 NODE 5: Read Address OCR")
    print(f"📦 Total images: {len(all_filenames)}")
    print(f"📊 Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Process images in batches
    total_batches = (len(all_filenames) + batch_size - 1) // batch_size if all_filenames else 0
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(all_filenames))
        batch_filenames = all_filenames[start_idx:end_idx]
        
        print(f"📦 Node 5 - Batch {batch_num + 1}/{total_batches} (images {start_idx + 1}-{end_idx})")
        
        for filename in batch_filenames:
            receiver_image = receiver_boxes[filename]
            
            # Skip if previous step had error
            if isinstance(receiver_image, int) and receiver_image < 0:
                address_texts[filename] = f"Error: {receiver_image}"
                predicted_regions[filename] = str(receiver_image)
                continue
            
            # Create output directories
            address_output_dir = os.path.join("data/03_read_address", filename)
            region_output_dir = os.path.join("data/03_address_region", filename)
            os.makedirs(address_output_dir, exist_ok=True)
            os.makedirs(region_output_dir, exist_ok=True)
            
            # Convert to numpy
            receiver_array = np.array(receiver_image)
            
            # Read address (returns both corrected and raw versions)
            address_string_corrected, address_string_raw = read_text(
                receiver_array,
                word_digit_hbb,
                real_ocr_model,
                save_path=None,
            )
            
            # Predict region from address (use corrected version)
            texts_region = str(detect_region_by_words_gilan(address_string_corrected))
            
            # Save results (both versions in one file)
            address_path = os.path.join(address_output_dir, f"{filename}_read_address.txt")
            with open(address_path, 'w', encoding='utf-8') as f:
                f.write(f"{address_string_raw}\n")
                f.write(f"{address_string_corrected}\n")
            
            region_path = os.path.join(region_output_dir, f"{filename}_address_region.txt")
            with open(region_path, 'w') as f:
                f.write(texts_region)
            
            address_texts[filename] = address_string_corrected
            predicted_regions[filename] = texts_region
            print(f"  ✓ Address region for {filename}: {texts_region}")
        
        print(f"✅ Batch {batch_num + 1}/{total_batches} complete\n")
    
    print(f"{'='*60}")
    print(f"✅ Node 5 Complete! Total: {len(address_texts)} images")
    print(f"{'='*60}\n")
    
    return address_texts, predicted_regions


def read_address_ocr_sari(
    receiver_boxes: Dict[str, Any],
    parameters: Dict[str, Any]
) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Node that reads address using OCR and predicts region.
    Processes images in batches for memory efficiency.
    
    Args:
        receiver_boxes: Dict of preprocessed receiver boxes
        parameters: Configuration parameters including:
            - batch_size: Number of images to process per batch (default: 200)
    
    Returns:
        tuple: (address_texts_dict, predicted_regions_dict)
    """
    batch_size = parameters.get("batch_size", 200)
    
    address_texts = {}
    predicted_regions = {}
    
    # Load models once
    print("\n🔄 Loading OCR models...")
    word_digit_hbb = YOLO(parameters["yolo_detect_word_digit_hbb_model_path"])
    real_ocr_model = load_paddle_model(parameters["real_ocr_model_path"])
    print("✅ OCR models loaded\n")
    
    # Get all filenames
    all_filenames = list(receiver_boxes.keys())
    
    print(f"{'='*60}")
    print(f"🔄 NODE 5: Read Address OCR")
    print(f"📦 Total images: {len(all_filenames)}")
    print(f"📊 Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Process images in batches
    total_batches = (len(all_filenames) + batch_size - 1) // batch_size if all_filenames else 0
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(all_filenames))
        batch_filenames = all_filenames[start_idx:end_idx]
        
        print(f"📦 Node 5 - Batch {batch_num + 1}/{total_batches} (images {start_idx + 1}-{end_idx})")
        
        for filename in batch_filenames:
            receiver_image = receiver_boxes[filename]
            
            # Skip if previous step had error
            if isinstance(receiver_image, int) and receiver_image < 0:
                address_texts[filename] = f"Error: {receiver_image}"
                predicted_regions[filename] = str(receiver_image)
                continue
            
            # Create output directories
            address_output_dir = os.path.join("data/03_read_address", filename)
            region_output_dir = os.path.join("data/03_address_region", filename)
            os.makedirs(address_output_dir, exist_ok=True)
            os.makedirs(region_output_dir, exist_ok=True)
            
            # Convert to numpy
            receiver_array = np.array(receiver_image)
            
            # Read address (returns both corrected and raw versions)
            address_string_corrected, address_string_raw = read_text(
                receiver_array,
                word_digit_hbb,
                real_ocr_model,
                save_path=None,
            )
            
            # Predict region from address (use corrected version)
            texts_region = str(detect_region_by_words_sari(address_string_corrected))
            
            # Save results (both versions in one file)
            address_path = os.path.join(address_output_dir, f"{filename}_read_address.txt")
            with open(address_path, 'w', encoding='utf-8') as f:
                f.write(f"{address_string_raw}\n")
                f.write(f"{address_string_corrected}\n")
            
            region_path = os.path.join(region_output_dir, f"{filename}_address_region.txt")
            with open(region_path, 'w') as f:
                f.write(texts_region)
            
            address_texts[filename] = address_string_corrected
            predicted_regions[filename] = texts_region
            print(f"  ✓ Address region for {filename}: {texts_region}")
        
        print(f"✅ Batch {batch_num + 1}/{total_batches} complete\n")
    
    print(f"{'='*60}")
    print(f"✅ Node 5 Complete! Total: {len(address_texts)} images")
    print(f"{'='*60}\n")
    
    return address_texts, predicted_regions