from typing import Optional
import numpy as np
from ultralytics import YOLO
from .CONSTS_tehran import CONSTS_tehran
from .read_postcode_base import read_postcode_base

def read_postcode_tehran(
    image: np.ndarray, digit_model: YOLO, save_path: Optional[str] = None
) -> int:
    return read_postcode_base(image, digit_model, CONSTS_tehran, save_path)