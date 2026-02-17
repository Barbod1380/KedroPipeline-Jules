"""Prepare cropped images for partial pipeline run."""

import pickle
from PIL import Image
from pathlib import Path


def prepare_cropped_images(input_folder: str, output_file: str):
    """
    Convert images from folder to pickle format for Kedro pipeline.
    
    Args:
        input_folder: Path to folder with cropped parcel images
        output_file: Where to save the pickle file
    """
    cropped_images = {}
    
    # Load all images
    for img_path in Path(input_folder).glob("*.png"):
        filename = img_path.stem  # name without .png
        img = Image.open(img_path)
        cropped_images[filename] = img
        print(f"✓ Loaded: {filename}")
    
    # Save as pickle
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(cropped_images, f)
    
    print(f"\n✅ Saved {len(cropped_images)} images to {output_file}")


if __name__ == "__main__":
    # Modify these paths as needed
    prepare_cropped_images(
        input_folder=r"data\01_raw", 
        output_file=r"data\02_intermediate\cropped_parcel_images.pkl"
    )