"""
gradcam.py
----------
Grad-CAM explainability for the trained ResNet50 diabetic-retinopathy model.

WHY THIS IMPLEMENTATION LOOKS THE WAY IT DOES
==============================================
The trained model nests the ResNet50 backbone as a single sub-model
("resnet50") inside the outer Functional model. In Keras 3, you generally
CANNOT build `tf.keras.Model(inputs=outer_model.inputs,
outputs=[base_model.get_layer(name).output, outer_model.output])` and get a
working gradient path, because the nested sub-model is executed as one
opaque call in the outer graph -- its internal tensors aren't wired as
separate nodes of the outer graph. This exact pattern was attempted first
in the original notebook (an earlier, simpler `make_gradcam_heatmap`) and
does not reliably produce usable gradients for this architecture.

The notebook's final, WORKING approach (used here) instead:
    1. Builds a small model directly from the nested ResNet50 sub-model
       alone: `Model(inputs=base_model.input, outputs=target_layer.output)`.
       This is valid because it is built entirely from the sub-model's own
       layers, not by reaching into the outer graph.
    2. Manually replays the outer model's forward pass under a
       `tf.GradientTape`:
         - augmentation layers (RandomFlip/Rotation/Zoom/Contrast) called
           with training=False, so they are inert / pass-through
         - ResNet50's "caffe-style" preprocessing (RGB->BGR + ImageNet mean
           subtraction), applied explicitly here via
           `keras.applications.resnet50.preprocess_input`, because that
           step is baked into the outer model's graph as low-level ops
           and is not reachable as a discrete layer we can call by hand.
         - the ResNet50 sub-model (via the small Grad-CAM model from step 1)
         - the classification head (GlobalAveragePooling2D -> Dropout ->
           Dense) called manually, layer by layer, using training=False on
           the dropout layer.
    3. Computes gradients of the predicted class's score with respect to
       the last convolutional feature map (`conv5_block3_out`), then
       produces the standard Grad-CAM weighted, ReLU'd, normalized heatmap.

This mirrors the model architecture exactly (see utils/prediction.py's
module docstring for the preprocessing rationale) and does not alter the
trained weights in any way.
"""

import cv2
import numpy as np

LAST_CONV_LAYER_NAME = "conv5_block3_out"
RESNET_SUBMODEL_NAME = "resnet50"


class GradCAMError(Exception):
    """Raised when Grad-CAM generation fails."""
    pass


def _get_augmentation_layers(model):
    """Return the model's augmentation layers, in order.

    These sit between the InputLayer and the ResNet50 sub-model in the
    saved architecture (RandomFlip, RandomRotation, RandomZoom,
    RandomContrast). We call them explicitly with training=False so they
    are guaranteed to be pass-through, matching real inference behaviour.
    """
    resnet_idx = None
    for i, layer in enumerate(model.layers):
        if layer.name == RESNET_SUBMODEL_NAME:
            resnet_idx = i
            break
    if resnet_idx is None:
        raise GradCAMError(
            f"Could not find a sub-layer named '{RESNET_SUBMODEL_NAME}' "
            f"in the loaded model. Grad-CAM cannot proceed."
        )
    # Layers between the InputLayer (index 0) and the ResNet50 sub-model.
    return model.layers[1:resnet_idx], resnet_idx


def make_gradcam_heatmap(preprocessed_batch: np.ndarray, model,
                          last_conv_layer_name: str = LAST_CONV_LAYER_NAME):
    """Compute a Grad-CAM heatmap for a single preprocessed image batch.

    Args:
        preprocessed_batch: np.ndarray of shape (1, 224, 224, 3), raw
            0-255 RGB float32 -- the SAME tensor that was fed to
            `model.predict()` for the prediction being explained.
        model: the loaded Keras model (outer model, with nested ResNet50).
        last_conv_layer_name: name of the ResNet50 layer to explain from.

    Returns:
        heatmap        : np.ndarray, shape (7, 7), values in [0, 1]
        predicted_class : int, the class index Grad-CAM's forward pass agreed on
        probabilities   : np.ndarray, shape (3,), softmax probabilities
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.resnet50 import preprocess_input

        base_model = model.get_layer(RESNET_SUBMODEL_NAME)
    except ValueError as exc:
        raise GradCAMError(f"ResNet50 sub-model not found: {exc}")

    try:
        target_layer = base_model.get_layer(last_conv_layer_name)
    except ValueError as exc:
        raise GradCAMError(
            f"Layer '{last_conv_layer_name}' not found inside the ResNet50 "
            f"sub-model: {exc}"
        )

    # Small model built directly from the nested ResNet50 sub-model's own
    # layers -- this is the piece that makes gradients actually flow.
    gradcam_feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=target_layer.output,
    )

    augmentation_layers, resnet_idx = _get_augmentation_layers(model)
    head_layers = model.layers[resnet_idx + 1:]  # GAP, Dropout, Dense (in order)

    img_tensor = tf.convert_to_tensor(preprocessed_batch)

    with tf.GradientTape() as tape:
        x = img_tensor

        # Replay augmentation layers (inert at training=False).
        for layer in augmentation_layers:
            x = layer(x, training=False)

        # Replay the ResNet50 "caffe-style" preprocessing that is baked
        # into the outer model's graph (RGB->BGR + mean subtraction).
        x = preprocess_input(x)

        # Feature extraction through the (separately built) ResNet50 model
        # so that its intermediate tensors are trackable by the tape.
        conv_outputs = gradcam_feature_model(x, training=False)
        tape.watch(conv_outputs)

        # Replay the classification head layer by layer.
        head_x = conv_outputs
        for layer in head_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                head_x = layer(head_x, training=False)
            else:
                head_x = layer(head_x)
        predictions = head_x

        predicted_class = tf.argmax(predictions[0])
        class_score = predictions[0, predicted_class]

    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        raise GradCAMError(
            "Gradient computation returned None. The Grad-CAM forward "
            "pass may not match the model's actual architecture."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs_0 = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs_0 * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)  # normalize to [0,1]

    return (
        heatmap.numpy(),
        int(predicted_class.numpy()),
        predictions.numpy()[0],
    )


def overlay_heatmap(original_rgb_uint8: np.ndarray, heatmap: np.ndarray,
                     alpha: float = 0.6) -> tuple:
    """Resize the low-res heatmap to the image size, colorize it, and blend
    it with the original image.

    Args:
        original_rgb_uint8: np.ndarray, shape (H, W, 3), dtype uint8, RGB.
        heatmap: np.ndarray, shape (h, w), values in [0, 1].
        alpha: weight given to the original image in the blend
               (1 - alpha is given to the heatmap color).

    Returns:
        heatmap_color_rgb: np.ndarray (H, W, 3) uint8 -- colorized heatmap only
        overlay_rgb      : np.ndarray (H, W, 3) uint8 -- blended overlay
    """
    h, w = original_rgb_uint8.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    overlay_rgb = cv2.addWeighted(
        original_rgb_uint8, alpha, heatmap_color_rgb, 1 - alpha, 0
    )

    return heatmap_color_rgb, overlay_rgb
