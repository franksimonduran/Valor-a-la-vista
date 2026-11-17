from __future__ import annotations

import hashlib
import importlib.util
import io
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .core import DAMAGE_CATALOG
from .vision import validate_image
from .model_release import read_active_card

DATA_ROOT = Path(__file__).resolve().parent.parent / ".data"
DEFAULT_WEIGHTS = DATA_ROOT / "models" / "terrenos" / "best.pt"
TERRAIN_CLASSES = tuple(code for code in DAMAGE_CATALOG["terreno"] if code != "otro_visible")


def detector_status():
    if not DEFAULT_WEIGHTS.is_file():
        return False, "YOLO no activado: falta seleccionar y publicar el modelo evaluado (.data/models/terrenos/best.pt). El análisis con OpenAI sí está disponible."
    if importlib.util.find_spec("ultralytics") is None:
        return False, "Falta instalar requirements-yolo.txt para usar el detector local."
    try:
        card = read_active_card(DEFAULT_WEIGHTS)
    except (ValueError, OSError):
        return False, "La ficha o el hash de los pesos no son válidos. YOLO no se cargará; revisa la publicación del modelo."
    return True, f"YOLO local experimental · solo residuos · {card['completed_epochs']} épocas. No validado para uso real."


def detector_card():
    try:
        return read_active_card(DEFAULT_WEIGHTS)
    except (ValueError, OSError):
        return None


def load_terrain_detector(weights=None, *, factory=None):
    card = read_active_card(DEFAULT_WEIGHTS) if weights is None else None
    path = Path(DEFAULT_WEIGHTS if weights is None else weights).resolve()
    if not path.is_relative_to(DATA_ROOT.resolve()) or path.suffix.lower() != ".pt" or not path.is_file():
        raise ValueError("Se requieren pesos .pt locales y confiables dentro de .data; no se descargan pesos genéricos.")
    if factory is None:
        os.environ.setdefault("YOLO_OFFLINE", "true")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        try:
            from ultralytics import YOLO, settings
        except ImportError:
            raise RuntimeError("Instala requirements-yolo.txt para usar YOLO local.") from None
        settings.update({"sync": False, "weights_dir": str(DATA_ROOT / "models"), "runs_dir": str(DATA_ROOT / "runs")})
        import torch
        from ultralytics.utils.events import events
        events.enabled = False
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        factory = YOLO
    model = factory(str(path), task="detect")
    names = model.names
    labels = list(names.values()) if isinstance(names, dict) else list(names)
    if (not labels or any(not isinstance(label, str) or label not in TERRAIN_CLASSES for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("El detector no tiene clases de daños de terrenos compatibles; un YOLO genérico no sirve como detector entrenado de estos daños.")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if card is not None and (digest != card["weights_sha256"] or labels != list(card["class_names"].values())):
        raise ValueError("Los pesos cargados no coinciden con la ficha del modelo activo.")
    return model, digest


def detect_regions(raw, model, *, threshold=.35):
    validate_image(raw)
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Umbral YOLO inválido.")
    with Image.open(io.BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((1280, 1280))
    predictions = model.predict(source=image, conf=threshold, iou=.7, device="cpu", imgsz=416, save=False, verbose=False, max_det=100)
    regions = []
    for result in predictions:
        for box in result.boxes:
            index = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            kind = model.names[index]
            coords = [float(value) for value in box.xyxy[0].tolist()]
            if (kind not in TERRAIN_CLASSES or not math.isfinite(confidence) or not threshold <= confidence <= 1
                    or len(coords) != 4 or not all(math.isfinite(v) for v in coords)):
                continue
            x1, y1, x2, y2 = coords
            if not 0 <= x1 < x2 <= image.width or not 0 <= y1 < y2 <= image.height:
                continue
            regions.append({"tipo": kind, "etiqueta": DAMAGE_CATALOG["terreno"][kind].nombre,
                            "confianza_detector": round(confidence, 4),
                            "caja_normalizada_xyxy": [round(x1/image.width, 6), round(y1/image.height, 6), round(x2/image.width, 6), round(y2/image.height, 6)]})
            if len(regions) >= 100:
                return regions
    return regions


def annotated_image(raw, regions):
    with Image.open(io.BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((1280, 1280))
    draw = ImageDraw.Draw(image)
    for region in regions:
        x1, y1, x2, y2 = region["caja_normalizada_xyxy"]
        box = (x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height)
        draw.rectangle(box, outline="#e18b28", width=3)
        draw.text((box[0] + 4, box[1] + 4), region["tipo"], fill="#e18b28", stroke_width=1, stroke_fill="black")
    return image
