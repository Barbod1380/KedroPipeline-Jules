from typing import Optional
import numpy as np
from ultralytics import YOLO
from .CONSTS_sari import CONSTS_sari
from .read_postcode_base import read_postcode_base

def read_postcode_sari(
    image: np.ndarray, digit_model: YOLO, save_path: Optional[str] = None
) -> int:
    return read_postcode_base(image, digit_model, CONSTS_sari, save_path)
