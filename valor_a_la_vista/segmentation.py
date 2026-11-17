"""Segmentación opcional de escenas. No detecta materiales ni daños por sí sola."""
from __future__ import annotations

import io
from PIL import Image

from . import bootstrap  # Configura las cachés también al importar este módulo.

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"


def load_segmenter():
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def segment_image(raw: bytes, processor, model):
    """Devuelve imagen superpuesta y porcentajes de superficie de clases observadas."""
    import numpy as np
    import torch

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    # Evitar imágenes gigantes en memoria.
    image.thumbnail((960, 960))
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    mask = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0].cpu().numpy()

    rgb = np.asarray(image).astype(np.float32)
    result = rgb.copy()
    counts = np.bincount(mask.ravel(), minlength=len(model.config.id2label))
    labels: list[tuple[str, float]] = []
    # Mezcla colores solo para clases suficientemente grandes.
    for label_id in np.argsort(counts)[::-1]:
        ratio = float(counts[label_id] / mask.size)
        if ratio < 0.015:
            continue
        hue = int(label_id)
        color = np.asarray([(hue * 73 + 48) % 256, (hue * 37 + 120) % 256,
                            (hue * 109 + 90) % 256], dtype=np.float32)
        result[mask == label_id] = rgb[mask == label_id] * 0.54 + color * 0.46
        labels.append((model.config.id2label[int(label_id)].strip(), round(ratio * 100, 1)))
    overlay = Image.fromarray(np.uint8(np.clip(result, 0, 255)))
    out = io.BytesIO()
    overlay.save(out, format="PNG")
    return out.getvalue(), labels[:12]
