import os
import shutil
from typing import NoReturn


def clean_folder(root_folder: str) -> NoReturn:
    """
    Cleans all files and subfolders inside each first-level folder
    within the given root folder. Keeps the first-level folders intact.

    Args:
        root_folder (str): Path to the root directory to clean.

    Returns:
        NoReturn
    """
    # Iterate through all items in the root folder
    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)

        # Process only directories (first-level folders)
        if os.path.isdir(item_path):
            for inner_item in os.listdir(item_path):
                inner_item_path = os.path.join(item_path, inner_item)

                try:
                    if os.path.isfile(inner_item_path) or os.path.islink(inner_item_path):
                        os.remove(inner_item_path)
                        print(f"Deleted file: {inner_item_path}")
                    elif os.path.isdir(inner_item_path):
                        shutil.rmtree(inner_item_path)
                        print(f"Deleted subfolder: {inner_item_path}")
                except Exception as e:
                    print(f"Failed to delete {inner_item_path}: {e}")


def main() -> None:
    """
    Main entry point for the script.
    Prompts for a root folder and cleans it.
    """
    root_directory = r"D:\Tensor\kedro\data-pipeline\data"

    if os.path.exists(root_directory):
        clean_folder(root_directory)
        print("\n✅ Cleaning complete. First-level folders kept.")
    else:
        print(f"❌ The folder '{root_directory}' does not exist.")


if __name__ == "__main__":
    main()
