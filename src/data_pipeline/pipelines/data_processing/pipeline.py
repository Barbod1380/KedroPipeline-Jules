"""Pipeline for data processing."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    load_and_crop_parcel,
    rotate_and_preprocess_parcel,
    receiver_postcode_detection,
    read_classify_postcode,
    read_address_ocr,
    read_classify_postcode_gilan,
    read_address_ocr_gilan,
    read_classify_postcode_sari,
    read_address_ocr_sari,
    read_classify_postcode_kerman,
    read_address_ocr_kerman,
    read_classify_postcode_kermanshah,
    read_address_ocr_kermanshah
)

def create_pipeline() -> Pipeline:
    """Create the data processing pipeline.
    
    Returns:
        Pipeline: The constructed pipeline
    """
    return Pipeline(
        [
            node(
                func=load_and_crop_parcel,
                inputs="parameters",  
                outputs="cropped_parcel_image",
                name="load_and_crop_parcel_node",
            ),
            node(
                func=rotate_and_preprocess_parcel,
                inputs=["cropped_parcel_image", "parameters"],
                outputs=["rotated_image", "rotated_preprocessed_image"],
                name="rotate_and_preprocess_parcel_node"
            ),
            node(
                func=receiver_postcode_detection,
                inputs=["rotated_image", "rotated_preprocessed_image", "parameters"],
                outputs=["postcode_box_prep_image", "receiver_box_prep_image"],
                name="receiver_postcode_detection_node"
            ),
            node(
                func=read_classify_postcode,
                inputs=["postcode_box_prep_image", "parameters"],
                outputs="postcode_region_predicted",
                name="read_classify_postcode_node"
            ),
            node(
                func=read_address_ocr,
                inputs=["receiver_box_prep_image", "parameters"],
                outputs=["read_address_txt", "predicted_region_by_address"],
                name="read_address_ocr_node"
            ),
        ]
    )



def create_partial_pipeline_from_cropped() -> Pipeline:
    """Create pipeline starting from cropped images (skips node 1).
    
    Returns:
        Pipeline: Pipeline starting from rotate_and_preprocess_parcel_node
    """
    return Pipeline(
        [
            node(
                func=rotate_and_preprocess_parcel,
                inputs=["cropped_parcel_image_from_disk", "parameters"], 
                outputs=["rotated_image", "rotated_preprocessed_image"],
                name="rotate_and_preprocess_parcel_node",
            ),
            node(
                func=receiver_postcode_detection,
                inputs=["rotated_image", "rotated_preprocessed_image", "parameters"],
                outputs=["postcode_box_prep_image", "receiver_box_prep_image"],
                name="receiver_postcode_detection_node",
            ),
            node(
                func=read_classify_postcode,
                inputs=["postcode_box_prep_image", "parameters"],
                outputs="postcode_region_predicted",
                name="read_classify_postcode_node",
            ),
            node(
                func=read_address_ocr,
                inputs=["receiver_box_prep_image", "parameters"],
                outputs=["read_address_txt", "predicted_region_by_address"],
                name="read_address_ocr_node",
            ),
        ]
    )


def create_partial_pipeline_from_cropped_gilan() -> Pipeline:
    """Create pipeline starting from cropped images (skips node 1).
    
    Returns:
        Pipeline: Pipeline starting from rotate_and_preprocess_parcel_node
    """
    return Pipeline(
        [
            node(
                func=rotate_and_preprocess_parcel,
                inputs=["cropped_parcel_image_from_disk", "parameters"], 
                outputs=["rotated_image", "rotated_preprocessed_image"],
                name="rotate_and_preprocess_parcel_node",
            ),
            node(
                func=receiver_postcode_detection,
                inputs=["rotated_image", "rotated_preprocessed_image", "parameters"],
                outputs=["postcode_box_prep_image", "receiver_box_prep_image"],
                name="receiver_postcode_detection_node",
            ),
            node(
                func=read_classify_postcode_gilan,
                inputs=["postcode_box_prep_image", "parameters"],
                outputs="postcode_region_predicted",
                name="read_classify_postcode_node",
            ),
            node(
                func=read_address_ocr_gilan,
                inputs=["receiver_box_prep_image", "parameters"],
                outputs=["read_address_txt", "predicted_region_by_address"],
                name="read_address_ocr_node",
            ),
        ]
    )


def create_partial_pipeline_from_cropped_sari() -> Pipeline:
    """Create pipeline starting from cropped images (skips node 1).
    
    Returns:
        Pipeline: Pipeline starting from rotate_and_preprocess_parcel_node
    """
    return Pipeline(
        [
            node(
                func=rotate_and_preprocess_parcel,
                inputs=["cropped_parcel_image_from_disk", "parameters"], 
                outputs=["rotated_image", "rotated_preprocessed_image"],
                name="rotate_and_preprocess_parcel_node",
            ),
            node(
                func=receiver_postcode_detection,
                inputs=["rotated_image", "rotated_preprocessed_image", "parameters"],
                outputs=["postcode_box_prep_image", "receiver_box_prep_image"],
                name="receiver_postcode_detection_node",
            ),
            node(
                func=read_classify_postcode_sari,
                inputs=["postcode_box_prep_image", "parameters"],
                outputs="postcode_region_predicted",
                name="read_classify_postcode_node",
            ),
            node(
                func=read_address_ocr_sari,
                inputs=["receiver_box_prep_image", "parameters"],
                outputs=["read_address_txt", "predicted_region_by_address"],
                name="read_address_ocr_node",
            ),
        ]
    )


def create_partial_pipeline_from_cropped_kerman() -> Pipeline:
    """Create pipeline starting from cropped images (skips node 1).
    
    Returns:
        Pipeline: Pipeline starting from rotate_and_preprocess_parcel_node
    """
    return Pipeline(
        [
            node(
                func=rotate_and_preprocess_parcel,
                inputs=["cropped_parcel_image_from_disk", "parameters"], 
                outputs=["rotated_image", "rotated_preprocessed_image"],
                name="rotate_and_preprocess_parcel_node",
            ),
            node(
                func=receiver_postcode_detection,
                inputs=["rotated_image", "rotated_preprocessed_image", "parameters"],
                outputs=["postcode_box_prep_image", "receiver_box_prep_image"],
                name="receiver_postcode_detection_node",
            ),
            node(
                func=read_classify_postcode_kerman,
                inputs=["postcode_box_prep_image", "parameters"],
                outputs="postcode_region_predicted",
                name="read_classify_postcode_node",
            ),
            node(
                func=read_address_ocr_kerman,
                inputs=["receiver_box_prep_image", "parameters"],
                outputs=["read_address_txt", "predicted_region_by_address"],
                name="read_address_ocr_node",
            ),
        ]
    )


def create_partial_pipeline_from_cropped_kermanshah() -> Pipeline:
    """Create pipeline starting from cropped images (skips node 1).
    
    Returns:
        Pipeline: Pipeline starting from rotate_and_preprocess_parcel_node
    """
    return Pipeline(
        [
            node(
                func=rotate_and_preprocess_parcel,
                inputs=["cropped_parcel_image_from_disk", "parameters"], 
                outputs=["rotated_image", "rotated_preprocessed_image"],
                name="rotate_and_preprocess_parcel_node",
            ),
            node(
                func=receiver_postcode_detection,
                inputs=["rotated_image", "rotated_preprocessed_image", "parameters"],
                outputs=["postcode_box_prep_image", "receiver_box_prep_image"],
                name="receiver_postcode_detection_node",
            ),
            node(
                func=read_classify_postcode_kermanshah,
                inputs=["postcode_box_prep_image", "parameters"],
                outputs="postcode_region_predicted",
                name="read_classify_postcode_node",
            ),
            node(
                func=read_address_ocr_kermanshah,
                inputs=["receiver_box_prep_image", "parameters"],
                outputs=["read_address_txt", "predicted_region_by_address"],
                name="read_address_ocr_node",
            ),
        ]
    )