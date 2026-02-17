from typing import Optional
from paddleocr import TextRecognition


def load_paddle_model(model_dir: str, device: str = "gpu") -> Optional[TextRecognition]:
    """
    Initializes the PaddleOCR model in recognition-only mode.
    
    This function should be called only once when your application starts.

    Parameters
    ----------
    model_dir : str
        Path to the fine-tuned inference model directory.
    dict_path : str
        Path to the character dictionary file used during training.
    use_gpu : bool, optional
        Whether to use the GPU for inference, by default True.

    Returns
    -------
    Optional[PaddleOCR]
        An initialized PaddleOCR instance configured for recognition-only,
        or None if initialization fails.
    """
    try:
        # Initialize PaddleOCR with detection disabled (det=False)
        # and specify the paths to your fine-tuned recognition model.
        ocr_engine = TextRecognition(
            model_dir=model_dir,
            model_name='PP-OCRv5_mobile_rec',
            device=device,
        )
        print("✅ Recognizer initialized successfully using the PaddleOCR class.")
        return ocr_engine
    except Exception as e:
        print(f"❌ Error initializing recognizer: {e}") 
        return None