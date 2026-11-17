"""Inferencia YOLO11n experimental mediante OpenCV DNN, sin PyTorch en Render.

Solo admite el archivo local assets/models/terrenos/best.onnx junto con su
model_card.json vinculado por SHA-256. La salida no mide área ni verifica daños.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .core import DAMAGE_CATALOG
from .model_release import validate_card
from .vision import validate_image

MODEL_DIR = Path(__file__).resolve().parent.parent / "assets" / "models" / "terrenos"
ONNX_PATH = MODEL_DIR / "best.onnx"
ONNX_CARD = MODEL_DIR / "model_card.json"
MAX_ONNX_BYTES = 50 * 1024 * 1024
SIZE = 416
CONFIDENCE = .35
NMS_IOU = .7
MAX_DETECTIONS = 100


@dataclass
class OpenCVYoloDetector:
    net: object
    path: Path


def read_onnx_card(weights=None, card_path=None):
    weights = Path(weights) if weights is not None else ONNX_PATH
    card_path = Path(card_path) if card_path is not None else ONNX_CARD
    if (not weights.is_file() or not card_path.is_file()
            or not 0 < weights.stat().st_size <= MAX_ONNX_BYTES
            or not 0 < card_path.stat().st_size < 100_000):
        raise ValueError("Faltan pesos ONNX y ficha de un modelo experimental evaluado.")
    try:
        card = validate_card(json.loads(card_path.read_text(encoding="utf-8")))
        export = card.get("onnx_export")
        if (not isinstance(export, dict) or export.get("format") != "yolo11_raw_xywh"
                or export.get("imgsz") != SIZE or export.get("opset") != 12
                or export.get("backend") != "opencv_dnn"
                or not re.fullmatch(r"[a-f0-9]{64}", str(export.get("sha256", "")))):
            raise ValueError("La ficha no corresponde al export ONNX previsto.")
    except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
        raise ValueError("Ficha del modelo ONNX inválida.") from exc
    with weights.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != export["sha256"]:
        raise ValueError("El SHA-256 del modelo ONNX no coincide con su ficha.")
    return card


def load_onnx_detector():
    card = read_onnx_card()
    try:
        net = cv2.dnn.readNetFromONNX(str(ONNX_PATH))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except cv2.error as exc:
        raise RuntimeError("No se pudo cargar el modelo ONNX con OpenCV DNN.") from exc
    return OpenCVYoloDetector(net, ONNX_PATH), card["onnx_export"]["sha256"]


def _predictions(output):
    """Normaliza la salida YOLO11: [1,5,N] o [1,N,5] para una sola clase."""
    if isinstance(output, (list, tuple)):
        if len(output) != 1:
            raise ValueError("Se esperaba una salida única del detector.")
        output = output[0]
    result = np.asarray(output)
    if result.ndim != 3 or result.shape[0] != 1:
        raise ValueError("Dimensiones de salida ONNX incompatibles.")
    matrix = result[0]
    if matrix.shape[0] == 5:
        matrix = matrix.T
    elif matrix.shape[1] != 5:
        raise ValueError("El modelo debe tener exactamente una clase: residuos.")
    if matrix.shape[0] > 10000 or matrix.shape[1] != 5:
        raise ValueError("Salida ONNX demasiado grande o incompatible.")
    return matrix


def detect_onnx_regions(raw, model: OpenCVYoloDetector, *, threshold=CONFIDENCE):
    validate_image(raw)
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Confianza YOLO inválida.")
    with Image.open(io.BytesIO(raw)) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    height, width = image.shape[:2]
    scale = min(SIZE / width, SIZE / height)
    target_w, target_h = round(width * scale), round(height * scale)
    left, top = (SIZE - target_w) // 2, (SIZE - target_h) // 2
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    padded[top:top + target_h, left:left + target_w] = resized
    blob = cv2.dnn.blobFromImage(padded, scalefactor=1 / 255.0,
                                 size=(SIZE, SIZE), swapRB=False, crop=False)
    try:
        model.net.setInput(blob)
        predictions = _predictions(model.net.forward())
    except (cv2.error, ValueError) as exc:
        raise RuntimeError("El modelo ONNX no pudo procesar esta imagen en OpenCV.") from exc
    boxes, scores, originals = [], [], []
    for cx, cy, bw, bh, confidence in predictions:
        if (not all(math.isfinite(float(x)) for x in (cx, cy, bw, bh, confidence))
                or confidence < threshold or confidence > 1 or bw <= 0 or bh <= 0):
            continue
        x1 = max(0.0, (float(cx - bw / 2) - left) / scale)
        y1 = max(0.0, (float(cy - bh / 2) - top) / scale)
        x2 = min(float(width), (float(cx + bw / 2) - left) / scale)
        y2 = min(float(height), (float(cy + bh / 2) - top) / scale)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            continue
        originals.append((x1, y1, x2, y2))
        boxes.append([int(x1), int(y1), max(1, math.ceil(x2 - x1)), max(1, math.ceil(y2 - y1))])
        scores.append(float(confidence))
    if not boxes:
        return []
    selected = cv2.dnn.NMSBoxes(boxes, scores, threshold, NMS_IOU)
    indexes = [int(i) for i in np.asarray(selected).reshape(-1)]
    indexes.sort(key=lambda i: scores[i], reverse=True)
    return [
        {"tipo": "residuos", "etiqueta": DAMAGE_CATALOG["terreno"]["residuos"].nombre,
         "confianza_detector": round(scores[i], 4),
         "caja_normalizada_xyxy": [
             round(originals[i][0] / width, 6), round(originals[i][1] / height, 6),
             round(originals[i][2] / width, 6), round(originals[i][3] / height, 6)]}
        for i in indexes[:MAX_DETECTIONS]
    ]
