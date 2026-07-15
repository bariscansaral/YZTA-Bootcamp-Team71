
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib


def _split_model(model):
    """
    Modeli ikiye ayırır:
      backbone    : nested MobileNetV2 (conv feature map üretir)
      head_layers : backbone'dan sonraki katmanlar (GAP, Dropout, Dense...)
    """
    backbone, backbone_idx = None, None
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        if isinstance(layer, tf.keras.Model) and not isinstance(layer, tf.keras.Sequential):
            backbone, backbone_idx = layer, i
            break
    if backbone is None:
        raise ValueError(
            "Nested backbone bulunamadi. Modelin MobileNetV2'yi bir alt-model "
            "olarak icerdiginden emin ol."
        )
    head_layers = model.layers[backbone_idx + 1:]
    return backbone, head_layers


def make_gradcam_heatmap(img_array, model, class_index=None):
    """
    img_array   : (1, 224, 224, 3) HAM goruntu (0-255). preprocess YAPMA.
    class_index : None ise en yuksek olasilikli sinif kullanilir.
    return      : (heatmap (7,7) [0,1] numpy, predictions (num_classes,) numpy)
    """
    backbone, head_layers = _split_model(model)

    x = tf.convert_to_tensor(img_array, dtype=tf.float32)
    # Augmentation inference'ta kimliktir (bypass). Preprocess deterministiktir,
    # bu yuzden burada elle uyguluyoruz.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    with tf.GradientTape() as tape:
        conv_output = backbone(x, training=False)   # (1,7,7,1280)
        tape.watch(conv_output)

        h = conv_output
        for layer in head_layers:                   # GAP -> Dropout -> Dense
            h = layer(h, training=False)
        predictions = h                             # (1, num_classes)

        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        raise RuntimeError("Gradient hesaplanamadi.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))     # (1280,)
    heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]  # (7,7,1)
    heatmap = tf.squeeze(heatmap)                             # (7,7)

    heatmap = tf.maximum(heatmap, 0)                          # ReLU
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy(), predictions.numpy()[0]


def overlay_heatmap(raw_img_224, heatmap, alpha=0.4, colormap="jet"):
    """
    raw_img_224 : (224,224,3) 0-255 numpy VEYA PIL.Image (orijinal MR).
    heatmap     : make_gradcam_heatmap'ten donen heatmap (7x7, [0,1]).
    return      : PIL.Image (isi haritasi orijinal goruntu uzerine bindirilmis).
    """
    if isinstance(raw_img_224, Image.Image):
        base = np.array(raw_img_224.convert("RGB")).astype("float32")
    else:
        base = np.array(raw_img_224).astype("float32")
        if base.ndim == 2:
            base = np.stack([base] * 3, axis=-1)

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    palette = matplotlib.colormaps[colormap](np.arange(256))[:, :3]
    colored = np.uint8(255 * palette[heatmap_uint8])
    colored = np.array(
        Image.fromarray(colored).resize(
            (base.shape[1], base.shape[0]), resample=Image.BILINEAR
        )
    ).astype("float32")

    blended = np.clip(colored * alpha + base * (1 - alpha), 0, 255).astype("uint8")
    return Image.fromarray(blended)
