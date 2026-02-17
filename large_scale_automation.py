import os
import shutil
import subprocess
import time

# ================= CONFIGURATION =================
# 1. DIRECTORY PATHS
FOLDER_A = r"H:\other_cities_data_extraction\rasht_images_new"
FOLDER_B = r"data\01_raw"
FOLDER_C = r"H:\other_cities_data_extraction\rasht_images"

# 2. EXECUTABLES AND SCRIPTS
PYTHON_EXE = r"C:/Users/ASUS/AppData/Local/Programs/Python/Python310/python.exe"
PROJECT_ROOT = r"."

PREPARE_SCRIPT = r"src\data_pipeline\prepare_cropped_data.py"
POSTPROCESS_SCRIPT = r"postprocess.py"
CLEAN_SCRIPT = r"clean_data_folder.py"

# 3. POSTPROCESS ARGUMENTS
# I have mapped your specific command requirements here
POST_ARGS = [
    "--root", r"data",
    "--model", r"models\hand_type_cls\hand_type_cls_1.pt",
    "--patterns", r"src\data_pipeline\pipelines\utils\CONSTS\patterns.xlsx",
    "--output", r"H:\other_cities_data_extraction\rasht_csv\rasht_train_data.csv",
    "--existing", r"H:\other_cities_data_extraction\rasht_csv\rasht_train_data.csv",
    "--test", r"H:\other_cities_data_extraction\rasht_csv\rasht_test_data.csv",
    "--use-cuda"
]

BATCH_SIZE = 300 
# =================================================

def get_files(folder):
    """Returns a list of filenames in the folder, ignoring hidden files."""
    return [f for f in os.listdir(folder) if not f.startswith('.')]

def main():
    print("🚀 Starting High-Scale Automation Pipeline...")
    
    while True:
        files_in_a = get_files(FOLDER_A)
        count_a = len(files_in_a)
        
        if count_a == 0:
            print("✅ Folder A is empty. All batches complete!")
            break
            
        print(f"\n--- 📦 Processing Batch. Images remaining in A: {count_a} ---")
        
        # 1. Move Batch from A to B
        batch = files_in_a[:BATCH_SIZE]
        print(f"Step 1: Moving {len(batch)} images to Raw folder...")
        for filename in batch:
            shutil.move(os.path.join(FOLDER_A, filename), os.path.join(FOLDER_B, filename))
            
        # 2. Run Pre-processing
        print("Step 2: Running prepare_cropped_data.py...")
        try:
            subprocess.run([PYTHON_EXE, PREPARE_SCRIPT], cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ CRITICAL ERROR in Pre-processing: {e}")
            break

        # 3. Run Kedro Pipeline
        print("Step 3: Running kedro pipeline...")
        try:
            subprocess.run("kedro run --pipeline=from_cropped", cwd=PROJECT_ROOT, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ CRITICAL ERROR in Kedro Pipeline: {e}")
            break

        # 4. Run Post-processing with specific arguments
        print("Step 4: Running postprocess.py with CUDA...")
        try:
            # Combining the script path with the arguments list
            post_cmd = [PYTHON_EXE, POSTPROCESS_SCRIPT] + POST_ARGS
            subprocess.run(post_cmd, cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ CRITICAL ERROR in Post-processing: {e}")
            break

        # 5. Archive (Move B to C)
        print("Step 5: Archiving processed images to Folder C...")
        files_in_b = get_files(FOLDER_B)
        for filename in files_in_b:
            src = os.path.join(FOLDER_B, filename)
            dst = os.path.join(FOLDER_C, filename)
            if os.path.exists(dst):
                name, ext = os.path.splitext(filename)
                dst = os.path.join(FOLDER_C, f"{name}_processed{ext}")
            shutil.move(src, dst)

        # 6. Clean Data Folder
        print("Step 6: Running clean_data_folder.py...")
        try:
            subprocess.run([PYTHON_EXE, CLEAN_SCRIPT], cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ CRITICAL ERROR in Cleanup: {e}")
            break

        print("✨ Batch complete. Waiting 2 seconds for file system...")
        time.sleep(2)

if __name__ == "__main__":
    main()