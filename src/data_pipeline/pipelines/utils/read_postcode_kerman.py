from typing import Optional
import numpy as np
from ultralytics import YOLO
from .CONSTS_kerman import CONSTS_kerman
from .read_postcode_base import read_postcode_base

def read_postcode_kerman(
    image: np.ndarray, digit_model: YOLO, save_path: Optional[str] = None
) -> int:
    return read_postcode_base(image, digit_model, CONSTS_kerman, save_path)
