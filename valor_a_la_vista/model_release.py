"""Publicación explícita y trazable de un detector experimental ya evaluado."""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from .training import count_metrics

DATA_ROOT = Path(__file__).resolve().parent.parent / ".data"
ACTIVE_WEIGHTS = DATA_ROOT / "models" / "terrenos" / "best.pt"
ACTIVE_CARD = ACTIVE_WEIGHTS.with_name("model_card.json")


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def probability(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1


def validate_card(card):
    if (not isinstance(card, dict) or card.get("schema_version") != 1 or card.get("status") != "experimental"
            or card.get("validated_for_real_use") is not False or card.get("class_names") != {"0": "residuos"}
            or not isinstance(card.get("weights_sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", card["weights_sha256"])
            or not isinstance(card.get("model_id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", card["model_id"])
            or isinstance(card.get("completed_epochs"), bool) or not isinstance(card.get("completed_epochs"), int)
            or not 1 <= card["completed_epochs"] <= 100):
        raise ValueError("Ficha del modelo experimental inválida.")
    inference = card.get("inference", {})
    if inference != {"imgsz": 416, "confidence": .35, "nms_iou": .7, "max_det": 100}:
        raise ValueError("Parámetros de inferencia distintos de la evaluación publicada.")
    for key in ("mAP50", "mAP50_95"):
        if not probability(card.get("test", {}).get(key)):
            raise ValueError("Faltan métricas de prueba válidas.")
    for key in ("precision", "recall", "f1"):
        if not probability(card.get("test_fixed_threshold", {}).get(key)):
            raise ValueError("Faltan métricas de umbral fijo válidas.")
    if not isinstance(card.get("limitations"), list) or not card["limitations"]:
        raise ValueError("Faltan las limitaciones del modelo.")
    return card


def read_active_card(weights=None, card_path=None, *, verify_hash=True):
    weights = Path(weights) if weights is not None else ACTIVE_WEIGHTS
    card_path = Path(card_path) if card_path is not None else weights.with_name("model_card.json")
    if not weights.is_file() or not card_path.is_file() or card_path.stat().st_size > 100_000:
        raise ValueError("Faltan pesos y ficha de un modelo entrenado, evaluado y activado.")
    try:
        card = validate_card(json.loads(card_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError, KeyError):
        raise ValueError("No se pudo validar la ficha del modelo.") from None
    if verify_hash and file_sha256(weights) != card["weights_sha256"]:
        raise ValueError("Los pesos no coinciden con el SHA-256 de la ficha; no se cargará el archivo.")
    return card


def prepare_release(run):
    run = Path(run).resolve()
    if not run.is_relative_to((DATA_ROOT / "runs").resolve()):
        raise ValueError("El experimento debe estar dentro de .data/runs.")
    report_path = run / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completado_experimental" or report.get("class_names") != {"0": "residuos"}:
        raise ValueError("El entrenamiento y su evaluación todavía no están completos.")
    if (not isinstance(report.get("completed_epochs"), int) or report["completed_epochs"] < 1
            or not (run / "test_predictions.json").is_file()):
        raise ValueError("Faltan épocas o predicciones de prueba registradas.")
    source = (run / "train" / "weights" / "best.pt").resolve()
    if (not source.is_relative_to(run) or not source.is_file()
            or Path(report.get("checkpoint", "")).resolve() != source
            or file_sha256(source) != report.get("sha256_checkpoint")):
        raise ValueError("El checkpoint evaluado no coincide con los pesos del experimento.")
    fixed = report.get("test_at_conf_035_iou_050", {})
    if (fixed.get("confidence") != .35 or fixed.get("matching_iou") != .5 or fixed.get("nms_iou") != .7
            or fixed.get("max_det") != 100 or fixed.get("images") != report.get("dataset_audit", {}).get("splits", {}).get("test")
            or report.get("parameters", {}).get("imgsz") != 416):
        raise ValueError("La evaluación no coincide con el punto de operación de la app.")
    predictions = json.loads((run / "test_predictions.json").read_text(encoding="utf-8"))
    if (not isinstance(predictions, list) or not isinstance(fixed.get("images"), int) or fixed["images"] < 1
            or len(predictions) != fixed["images"] or len({row.get("image_id") for row in predictions}) != len(predictions)):
        raise ValueError("Las predicciones de prueba están incompletas o repetidas.")
    counts = {key: 0 for key in ("tp", "fp", "fn")}
    for row in predictions:
        for key in counts:
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Conteos de detección inválidos.")
            counts[key] += value
    recomputed = count_metrics(counts)
    if any(fixed.get(key) != value for key, value in counts.items()) or any(
            not probability(fixed.get(key)) or abs(fixed[key] - recomputed[key]) > 1e-10 for key in ("precision", "recall", "f1")):
        raise ValueError("Las métricas no coinciden con las predicciones por imagen.")
    card = {"schema_version": 1, "model_id": run.name, "architecture": "YOLO11n", "status": "experimental",
            "class_names": {"0": "residuos"}, "validated_for_real_use": False,
            "completed_epochs": report["completed_epochs"], "weights_sha256": report["sha256_checkpoint"],
            "dataset_manifest_sha256": report["sha256_dataset_manifest"], "dataset_splits": report["dataset_audit"]["splits"],
            "test": report.get("test", {}), "test_fixed_threshold": fixed,
            "inference": {"imgsz": 416, "confidence": .35, "nms_iou": .7, "max_det": 100},
            "limitations": report.get("limitations", []), "versions": report.get("versions", {}),
            "license": "Ultralytics AGPL-3.0 / revisar condiciones Enterprise y permisos del dataset antes de distribuir o usar comercialmente",
            "source_report": report_path.relative_to(DATA_ROOT).as_posix(),
            "activated_utc": datetime.now(timezone.utc).isoformat()}
    validate_card(card)
    return source, card


def publish_release(run, *, target=None):
    """Nunca sustituye otros pesos. Permite repetir la misma publicación sin cambios."""
    source, card = prepare_release(run)
    target = Path(target) if target is not None else ACTIVE_WEIGHTS
    target = target.resolve()
    if not target.is_relative_to((DATA_ROOT / "models").resolve()):
        raise ValueError("Destino fuera de .data/models.")
    card_path = target.with_name("model_card.json")
    if target.exists() or card_path.exists():
        previous = read_active_card(target, card_path)
        if previous["weights_sha256"] == card["weights_sha256"] and previous["model_id"] == card["model_id"]:
            return previous
        raise ValueError("Ya existe un modelo activo distinto. Se conserva; requiere respaldo y selección explícita antes de sustituirlo.")
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != card["weights_sha256"]:
        raise ValueError("El checkpoint cambió durante la publicación.")
    with target.open("xb") as handle:
        handle.write(raw)
    with card_path.open("x", encoding="utf-8") as handle:
        json.dump(card, handle, ensure_ascii=False, indent=2, allow_nan=False)
    read_active_card(target, card_path)
    return card
