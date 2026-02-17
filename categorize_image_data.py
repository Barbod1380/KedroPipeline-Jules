"""
Script to categorize image data by file type.
Extracts specific file types from individual image folders and organizes them into category folders.
"""

import os
import shutil


def categorize_image_data(merged_data_directory: str, output_directory: str):
    """
    Categorizes files from merged image folders into type-specific folders.
    
    Args:
        merged_data_directory: Path to the merged data folder (output from previous script)
        output_directory: Path where categorized folders will be created
    """
    
    # Define the mapping: folder_name -> file_ending
    CATEGORY_MAPPING = {
        "address_region_pred": "_address_region.txt",
        "falses_preprocessed_rotated": "_rotated_preprocessed.png",
        "postcode_img_preprocessed": "_postcode_box_prep.png",
        "read_postcode": "_postcode_region.txt",
        "read_words": "_read_address.txt",
        "receiver_image_preprocessed": "_receiver_box_prep.png"
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Create category folders
    print("Creating category folders...")
    for category in CATEGORY_MAPPING.keys():
        category_path = os.path.join(output_directory, category)
        os.makedirs(category_path, exist_ok=True)
        print(f"  ✓ Created: {category}/")
    
    print()
    
    # Check if merged data directory exists
    if not os.path.exists(merged_data_directory):
        print(f"❌ Error: {merged_data_directory} does not exist!")
        return
    
    # Get all image folders
    image_folders = [f for f in os.listdir(merged_data_directory) 
                     if os.path.isdir(os.path.join(merged_data_directory, f))]
    
    if not image_folders:
        print(f"❌ No image folders found in {merged_data_directory}")
        return
    
    print(f"Found {len(image_folders)} image folders to process\n")
    
    # Statistics
    stats = {category: 0 for category in CATEGORY_MAPPING.keys()}
    
    # Process each image folder
    for image_folder in image_folders:
        image_folder_path = os.path.join(merged_data_directory, image_folder)
        print(f"Processing: {image_folder}")
        
        # Get all files in this image folder
        files = os.listdir(image_folder_path)
        
        # Check each category
        for category, file_ending in CATEGORY_MAPPING.items():
            # Find files that match this category
            matching_files = [f for f in files if f.endswith(file_ending)]
            
            if matching_files:
                for file_name in matching_files:
                    src_file = os.path.join(image_folder_path, file_name)
                    dst_file = os.path.join(output_directory, category, file_name)
                    
                    # Copy the file
                    shutil.copy2(src_file, dst_file)
                    stats[category] += 1
                    print(f"  ✓ Copied {file_name} → {category}/")
            else:
                print(f"  ⚠ No file ending with '{file_ending}' found")
        
        print()
    
    # Print summary
    print(f"\n{'='*60}")
    print("✅ Categorization Complete!")
    print(f"{'='*60}")
    print("\nSummary:")
    for category, count in stats.items():
        print(f"  {category}: {count} files")
    print(f"\nOutput location: {output_directory}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Example usage - modify these paths as needed
    MERGED_DATA_DIRECTORY = r"D:\Tensor\analysis\gilan_analysis"  # Output from previous script
    OUTPUT_DIRECTORY = r"D:\Tensor\analysis\gilan_analysis_categorize"
    
    categorize_image_data(MERGED_DATA_DIRECTORY, OUTPUT_DIRECTORY)