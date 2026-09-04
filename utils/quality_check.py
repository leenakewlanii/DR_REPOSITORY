"""Non-clinical technical checks for uploaded images."""

import numpy as np
from PIL import Image


def technical_quality_warning(image: Image.Image) -> str | None:
    """Flag obvious technical issues without claiming clinical quality assessment."""
    if image.width < 160 or image.height < 160:
        return "The image resolution is low."
    sample = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32)
    brightness = float(sample.mean())
    contrast = float(sample.std())
    if brightness < 25:
        return "The image appears very dark."
    if brightness > 235:
        return "The image appears very bright."
    if contrast < 12:
        return "The image has very low contrast and may be blurred."
    return None
