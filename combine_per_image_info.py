"""
Script to merge all image-related data into consolidated folders.
"""

import os
import shutil

def merge_image_data(data_folder: str, output_directory: str):
    """
    Merges all image-related data from multiple subfolders into consolidated folders.
    
    Args:
        data_folder: Path to the main data folder containing all subfolders
        output_directory: Path where consolidated folders will be created
    """
    
    # Define paths
    raw_folder = os.path.join(data_folder, "01_raw")
    intermediate_folder = os.path.join(data_folder, "02_intermediate")
    address_region_folder = os.path.join(data_folder, "03_address_region")
    postcode_regions_folder = os.path.join(data_folder, "03_postcode_regions")
    read_address_folder = os.path.join(data_folder, "03_read_address")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Get all images from 01_raw
    if not os.path.exists(raw_folder):
        print(f"❌ Error: {raw_folder} does not exist!")
        return
    
    image_files = [f for f in os.listdir(raw_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"❌ No images found in {raw_folder}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    
    # Process each image
    for image_file in image_files:
        # Get image name without extension
        image_name = os.path.splitext(image_file)[0]
        
        # Create output folder for this image
        output_image_folder = os.path.join(output_directory, image_name)
        os.makedirs(output_image_folder, exist_ok=True)
        
        print(f"Processing: {image_name}")
        
        # 1. Copy original image from 01_raw
        src_image = os.path.join(raw_folder, image_file)
        dst_image = os.path.join(output_image_folder, image_file)
        shutil.copy2(src_image, dst_image)
        print(f"  ✓ Copied original image")
        
        # 2. Copy all files from 02_intermediate/{image_name}/
        intermediate_src = os.path.join(intermediate_folder, image_name)
        if os.path.exists(intermediate_src) and os.path.isdir(intermediate_src):
            files = os.listdir(intermediate_src)
            for file in files:
                src_file = os.path.join(intermediate_src, file)
                dst_file = os.path.join(output_image_folder, file)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
            print(f"  ✓ Copied {len(files)} files from 02_intermediate")
        else:
            print(f"  ⚠ No data in 02_intermediate/{image_name}")
        
        # 3. Copy all files from 03_address_region/{image_name}/
        address_region_src = os.path.join(address_region_folder, image_name)
        if os.path.exists(address_region_src) and os.path.isdir(address_region_src):
            files = os.listdir(address_region_src)
            for file in files:
                src_file = os.path.join(address_region_src, file)
                dst_file = os.path.join(output_image_folder, file)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
            print(f"  ✓ Copied {len(files)} files from 03_address_region")
        else:
            print(f"  ⚠ No data in 03_address_region/{image_name}")
        
        # 4. Copy all files from 03_postcode_regions/{image_name}/
        postcode_regions_src = os.path.join(postcode_regions_folder, image_name)
        if os.path.exists(postcode_regions_src) and os.path.isdir(postcode_regions_src):
            files = os.listdir(postcode_regions_src)
            for file in files:
                src_file = os.path.join(postcode_regions_src, file)
                dst_file = os.path.join(output_image_folder, file)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
            print(f"  ✓ Copied {len(files)} files from 03_postcode_regions")
        else:
            print(f"  ⚠ No data in 03_postcode_regions/{image_name}")
        
        # 5. Copy all files from 03_read_address/{image_name}/
        read_address_src = os.path.join(read_address_folder, image_name)
        if os.path.exists(read_address_src) and os.path.isdir(read_address_src):
            files = os.listdir(read_address_src)
            for file in files:
                src_file = os.path.join(read_address_src, file)
                dst_file = os.path.join(output_image_folder, file)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
            print(f"  ✓ Copied {len(files)} files from 03_read_address")
        else:
            print(f"  ⚠ No data in 03_read_address/{image_name}")
        
        print(f"✅ Completed: {image_name}\n")
    
    print(f"\n{'='*60}")
    print(f"✅ All done! Merged data for {len(image_files)} images")
    print(f"Output location: {output_directory}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Example usage - modify these paths as needed
    DATA_FOLDER = r"D:\Tensor\kedro\data-pipeline\data"
    OUTPUT_DIRECTORY = r"D:\Tensor\analysis\gilan_analysis"
    
    merge_image_data(DATA_FOLDER, OUTPUT_DIRECTORY)