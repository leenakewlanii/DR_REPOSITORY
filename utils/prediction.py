"""
prediction.py
--------------
Model loading, image preprocessing, and inference utilities for the
Diabetic Retinopathy classification app.

IMPORTANT ARCHITECTURE NOTE
============================
The saved model (`final_DR_model.keras`) was built as:

    Input(224, 224, 3)
        -> RandomFlip / RandomRotation / RandomZoom / RandomContrast   (train-time only, inactive at inference)
        -> ResNet50 "caffe-style" preprocessing (RGB->BGR + per-channel
           ImageNet mean subtraction) -- THIS IS BAKED INTO THE MODEL GRAPH
        -> ResNet50 backbone (ImageNet weights, frozen)
        -> GlobalAveragePooling2D
        -> Dropout
        -> Dense(3, softmax)

Because the ResNet50 preprocessing step is *inside* the saved model graph
(verified by inspecting the model's layer graph), the image tensor handed
to `model.predict()` must be:

    * RGB channel order
    * float32
    * pixel values in the raw 0-255 range (NOT rescaled to 0-1)
    * NOT already run through `keras.applications.resnet50.preprocess_input`

Applying `preprocess_input` externally would double-preprocess the image
and silently corrupt every prediction. Do not add a Rescaling(1./255)
step here either -- the training pipeline never used one for this model.
"""

import json
import os

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 224


class ModelLoadError(Exception):
    """Raised when the model or class mapping fails to load."""
    pass


def load_class_mapping(path: str) -> dict:
    """Load the {index: class_name} mapping from JSON.

    Returns a dict keyed by int index, e.g. {0: "Normal", 1: "NPDR", 2: "PDR"}.
    """
    try:
        with open(path, "r") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    except Exception as exc:
        raise ModelLoadError(f"Failed to load class mapping from {path}: {exc}")


def load_model(model_path: str):
    """Load the trained Keras model from disk.

    Does NOT retrain, recompile with a different architecture, or alter
    weights in any way -- this only deserializes the .keras file exactly
    as provided.
    """
    if not os.path.exists(model_path):
        raise ModelLoadError(
            f"Model file not found at '{model_path}'. Please place "
            f"final_DR_model.keras inside the models/ directory."
        )
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as exc:
        raise ModelLoadError(f"Failed to load model from {model_path}: {exc}")


def validate_image(pil_image: Image.Image) -> bool:
    """Basic sanity check that this looks like a usable image."""
    if pil_image is None:
        return False
    if pil_image.width < 32 or pil_image.height < 32:
        return False
    return True


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL image into the exact tensor format the model expects.

    Steps:
        1. Correct EXIF rotation (common with phone-camera fundus photos).
        2. Convert to RGB (drops alpha channel / handles grayscale input).
        3. Resize to 224x224.
        4. Cast to float32, keeping pixel values in the raw 0-255 range.
        5. Add a batch dimension -> shape (1, 224, 224, 3).

    No augmentation and no ResNet50 preprocess_input call here -- both are
    already handled by the model itself at inference time (see module
    docstring).
    """
    image = ImageOps.exif_transpose(pil_image)
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    array = np.asarray(image).astype("float32")  # 0-255 range, HWC, RGB
    batched = np.expand_dims(array, axis=0)       # (1, 224, 224, 3)
    return batched


def predict(model, class_mapping: dict, pil_image: Image.Image):
    """Run inference on a single PIL image.

    Returns a dict with:
        predicted_class_idx : int
        predicted_class_name: str
        confidence          : float (0-100)
        probabilities        : dict {class_name: probability (0-100)}
        preprocessed_batch   : np.ndarray, the (1,224,224,3) tensor fed to
                                the model (also reused by Grad-CAM so the
                                heatmap matches exactly what was predicted)
    """
    batch = preprocess_image(pil_image)
    raw_probs = model.predict(batch, verbose=0)[0]  # shape (3,)

    predicted_idx = int(np.argmax(raw_probs))
    predicted_name = class_mapping.get(predicted_idx, str(predicted_idx))
    confidence = float(raw_probs[predicted_idx]) * 100.0

    probabilities = {
        class_mapping.get(i, str(i)): float(p) * 100.0
        for i, p in enumerate(raw_probs)
    }

    return {
        "predicted_class_idx": predicted_idx,
        "predicted_class_name": predicted_name,
        "confidence": confidence,
        "probabilities": probabilities,
        "preprocessed_batch": batch,
    }
