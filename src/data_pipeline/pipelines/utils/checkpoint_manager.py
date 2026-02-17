"""Checkpoint management for batch processing."""

import os
import json
from typing import Dict, Set, Any


def load_checkpoint(checkpoint_dir: str, checkpoint_name: str = "processed_images.json") -> Set[str]:
    """
    Load the set of already processed image filenames.
    
    Args:
        checkpoint_dir: Directory where checkpoint files are stored
        checkpoint_name: Name of the checkpoint file
    
    Returns:
        Set of processed image filenames (without extensions)
    """
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                print(f"✓ Loaded checkpoint: {len(data['processed'])} images already processed")
                return set(data['processed'])
        except Exception as e:
            print(f"⚠ Warning: Could not load checkpoint: {e}")
            return set()
    
    return set()


def save_checkpoint(
    checkpoint_dir: str,
    processed_images: Set[str],
    checkpoint_name: str = "processed_images.json"
) -> None:
    """
    Save the set of processed image filenames to checkpoint.
    
    Args:
        checkpoint_dir: Directory where checkpoint files are stored
        processed_images: Set of processed image filenames
        checkpoint_name: Name of the checkpoint file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump({'processed': list(processed_images)}, f, indent=2)
        print(f"✓ Checkpoint saved: {len(processed_images)} images processed")
    except Exception as e:
        print(f"⚠ Warning: Could not save checkpoint: {e}")


def get_pending_images(all_images: list, processed_images: Set[str]) -> list:
    """
    Get list of images that haven't been processed yet.
    
    Args:
        all_images: List of all image filenames
        processed_images: Set of already processed image filenames
    
    Returns:
        List of pending image filenames
    """
    pending = [img for img in all_images if os.path.splitext(img)[0] not in processed_images]
    
    if processed_images:
        print(f"📊 Progress: {len(processed_images)}/{len(all_images)} images already processed")
        print(f"📋 Remaining: {len(pending)} images to process")
    
    return pending