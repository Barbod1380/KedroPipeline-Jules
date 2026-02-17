from typing import Tuple
import cv2
from ultralytics import YOLO
import numpy as np


# this function has written for two classes but it should get name or dictionary of calsses and be flexible to numbert of classes.
def detect_obb_coordinates(
    image: np.ndarray, model: YOLO, save_path: str = None, name: str = ""
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict objects in an image using the YOLO model.

    This function performs predictions on the provided image
    using the 'cuda' device, and identifies objects from two classes: 'receiver' and 'postcode'. For each class, it saves
    the coordinates in 'xyxyxyxy' format of the object with the highest probability. Optionally, the predictions can be
    saved to a file.

    Args:
        image (np.ndarray): The input image's path.
        model (YOLO): A preloaded YOLO model object.
        save_path (str, optional): Whether to save the predicted image with bounding boxes. Defaults to None and not save.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            A tuple containing 'xyxyxyxy' coordinates of the most probable 'receiver' and 'postcode' objects as numpy arrays.
    """
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    # print("==============================")
    # print(image_norm.shape, len(image_norm.shape))

    # print("SAVING IMAGE:")
    # cv2.imwrite(r"C:\Users\ASUS\Desktop\sample_image_test\kedro_res\before_pass_into_predict.png", image_norm)
    # print("SAVING DONE")

    # Perform prediction
    results = model.predict(image_norm, device="cuda", verbose=False)
    
    # Initialize variables to hold the coordinates
    receiver_box = None
    postcode_box = None
    max_receiver_conf = 0
    max_postcode_conf = 0
    
    # Iterate through predictions
    for result in results:
        if hasattr(result, "obb"):
            for i, conf in enumerate(result.obb.conf):
                cls = int(result.obb.cls[i])  # Class index
                confidence = float(conf)  # Confidence value
                coordinates = result.obb.xyxyxyxy[i]  # xyxyxyxy coordinates
                if cls == 0 and confidence > max_receiver_conf:  # Receiver
                    max_receiver_conf = confidence
                    receiver_box = coordinates

                elif cls == 1 and confidence > max_postcode_conf:  # Postcode
                    max_postcode_conf = confidence
                    postcode_box = coordinates

    # Optionally save the prediction image
    save_path=r"C:\Users\ASUS\Desktop\sample_image_test\kedro_res_sub"
    
    if save_path:
        annotated_image = results[0].plot()  # Annotate the image with bounding boxes
        cv2.imwrite(f"{save_path}/{name}_detection_image.png", annotated_image)
    if receiver_box is not None:
        receiver_box = receiver_box.cpu().numpy()  # Move to CPU and convert to NumPy
    if postcode_box is not None:
        postcode_box = postcode_box.cpu().numpy()  # Move to CPU and convert to NumPy
    return receiver_box, postcode_box
