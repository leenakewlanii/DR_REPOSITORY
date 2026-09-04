"""Printable, self-contained screening summary generation."""

import base64
import html
import io
from datetime import datetime

from PIL import Image


def _image_data(image) -> str:
    if image is None:
        return ""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_printable_report(result, timestamp, gradcam_images, urgency, next_step) -> str:
    """Build an HTML report that can be printed to PDF from the browser."""
    name = html.escape(result["predicted_class_name"])
    rows = "".join(f"<tr><td>{html.escape(label)}</td><td>{value:.2f}%</td></tr>" for label, value in result["probabilities"].items())
    images = ""
    if gradcam_images:
        images = "<h2>Explainability</h2><div class='images'>" + "".join(f"<figure><img src='data:image/png;base64,{_image_data(image)}'><figcaption>{label}</figcaption></figure>" for image, label in zip(gradcam_images, ["Original", "Heatmap", "Overlay"])) + "</div>"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>DrishtiAI screening report</title><style>body{{font:16px Arial,sans-serif;color:#142b3a;max-width:850px;margin:40px auto;padding:0 20px}}h1{{color:#123248}}h2{{border-bottom:1px solid #dce8e3;padding-bottom:8px}}table{{border-collapse:collapse;width:100%}}td{{border-bottom:1px solid #dce8e3;padding:10px}}.label{{color:#667a83}}.images{{display:flex;gap:14px;flex-wrap:wrap}}figure{{margin:0;width:30%}}img{{max-width:100%}}figcaption{{text-align:center;color:#667a83}}.warning{{background:#fff8e9;padding:14px;border-left:4px solid #c8872e}}</style></head><body><h1>DrishtiAI screening summary</h1><p class='label'>Generated: {html.escape(timestamp or datetime.now().strftime('%Y-%m-%d %H:%M'))}</p><div class='warning'><strong>Research & Educational Prototype</strong><br>This is an experimental AI screening prediction, not a medical diagnosis. It should not be used to make clinical decisions.</div><h2>Result</h2><p><strong>Predicted class:</strong> {name}</p><p><strong>Model confidence:</strong> {result['confidence']:.2f}%</p><table>{rows}</table><h2>Suggested follow-up</h2><p><strong>{html.escape(urgency)}</strong></p><p>{html.escape(next_step)}</p>{images}<h2>Model information</h2><p>Supplied model: ResNet50. Input: 224 x 224 x 3. Dataset: DR_512, 1,608 images total and 326 test images.</p><p class='label'>AI predictions may be incorrect and should be confirmed by a qualified healthcare professional.</p></body></html>"""
