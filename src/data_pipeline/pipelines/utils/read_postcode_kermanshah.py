from typing import Optional
import numpy as np
from ultralytics import YOLO
from .CONSTS_kermanshah import CONSTS_kermanshah
from .read_postcode_base import read_postcode_base

def read_postcode_kermanshah(
    image: np.ndarray, digit_model: YOLO, save_path: Optional[str] = None
) -> int:
    return read_postcode_base(image, digit_model, CONSTS_kermanshah, save_path)
