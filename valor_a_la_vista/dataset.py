"""Conversión auditable de anotaciones públicas COCO a un piloto YOLO de residuos."""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
from collections import defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

from .vision import perceptual_hash, validate_image

TACO_COMMIT = "3844bcd3535cd12ec0218494d1d79f1372f553cc"
TACO_ANNOTATIONS_URL = f"https://raw.githubusercontent.com/pedropro/TACO/{TACO_COMMIT}/data/annotations.json"
TACO_RECORD_URL = "https://zenodo.org/api/records/3587843"
TACO_SOURCE_URL = "https://github.com/pedropro/TACO"


def outdoor_candidates(coco, *, seed=42):
    """Tierra/vegetación según metadatos públicos; no confirma un terreno peruano."""
    categories = {row["id"] for row in coco["categories"]}
    scenes = defaultdict(set)
    for row in coco.get("scene_annotations", []):
        scenes[row["image_id"]].update(row.get("background_ids", []))
    annotations = defaultdict(list)
    for row in coco["annotations"]:
        if row.get("category_id") in categories:
            annotations[row["image_id"]].append(row)
    groups = defaultdict(list)
    for image in coco["images"]:
        ident = image["id"]
        backgrounds = scenes[ident]
        path = PurePosixPath(image["file_name"])
        if (not isinstance(ident, int) or len(path.parts) != 2 or path.parts[0] in {".", ".."}
                or not annotations[ident] or not backgrounds.intersection({3, 5}) or 1 in backgrounds):
            continue
        groups[path.parts[0]].append({"image": image, "annotations": annotations[ident],
                                      "scene_ids": sorted(backgrounds), "group": path.parts[0]})
    for group, rows in groups.items():
        random.Random(f"{seed}:{group}").shuffle(rows)
    return dict(sorted(groups.items()))


def coco_boxes_to_yolo(annotations, width, height, *, category_ids):
    """Todas las categorías TACO se fusionan en clase 0=residuos; nunca inventa cajas."""
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
               for v in (width, height)):
        raise ValueError("Dimensiones de anotación inválidas.")
    lines, seen = [], set()
    for row in annotations:
        if row.get("category_id") not in category_ids or row.get("iscrowd", 0) != 0:
            raise ValueError("Categoría desconocida o región crowd: requiere revisión, no se convierte como negativo.")
        bbox = row.get("bbox")
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox)):
            raise ValueError("Caja COCO inválida.")
        x, y, w, h = bbox
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width + 1 or y + h > height + 1:
            raise ValueError("Caja fuera de la imagen: requiere revisión.")
        x2, y2 = min(x + w, width), min(y + h, height)
        if x >= x2 or y >= y2:
            raise ValueError("Caja sin superficie.")
        values = ((x + x2) / (2 * width), (y + y2) / (2 * height), (x2 - x) / width, (y2 - y) / height)
        line = "0 " + " ".join(f"{v:.8f}" for v in values)
        if line not in seen:
            seen.add(line)
            lines.append(line)
    if not lines:
        raise ValueError("Imagen sin cajas válidas: no se crea un negativo sin revisar.")
    return "\n".join(lines) + "\n"


def normalize_training_image(raw, width, height):
    """Conserva el encuadre: cambia resolución, no recorta ni gira cajas a ciegas."""
    validate_image(raw)
    with Image.open(io.BytesIO(raw)) as source:
        if source.getexif().get(274, 1) not in (None, 1):
            raise ValueError("Orientación EXIF no normalizada; revisar cajas antes de usar.")
        if abs((source.width / source.height) / (width / height) - 1) > .01:
            raise ValueError("Proporción distinta de las anotaciones; no se estiran ni se recortan cajas.")
        image = source.convert("RGB")
        image.thumbnail((640, 640))
        output = io.BytesIO()
        image.save(output, "JPEG", quality=92, optimize=True)
        encoded = output.getvalue()
        return encoded, image.width, image.height, hashlib.sha256(encoded).hexdigest(), perceptual_hash(encoded)


def split_by_group(records, *, seed=42):
    """Holdout aproximado 70/20/10 por lote; no inventa identificadores de terreno."""
    groups = sorted({row["group"] for row in records})
    if len(groups) < 3:
        raise ValueError("Se necesitan al menos tres grupos para separar train, val y test.")
    random.Random(seed).shuffle(groups)
    n_test = max(1, round(len(groups) * .1))
    n_val = max(1, round(len(groups) * .2))
    if n_test + n_val >= len(groups):
        n_test, n_val = 1, 1
    assignment = {group: "test" if i < n_test else "val" if i < n_test + n_val else "train"
                  for i, group in enumerate(groups)}
    return [{**row, "split": assignment[row["group"]]} for row in records]


def audit_records(records):
    groups, digests = {}, set()
    phashes = []
    for row in records:
        group, split = row["group"], row["split"]
        if split not in {"train", "val", "test"} or (group in groups and groups[group] != split):
            raise ValueError("Un grupo se ha mezclado entre divisiones.")
        groups[group] = split
        if row["sha256"] in digests:
            raise ValueError("Imagen repetida en el piloto.")
        digests.add(row["sha256"])
        current = int(row["dhash"], 16)
        if any((current ^ previous).bit_count() <= 4 for previous in phashes):
            raise ValueError("Posible imagen casi duplicada: separar o revisar antes de entrenar.")
        phashes.append(current)
    if {row["split"] for row in records} != {"train", "val", "test"}:
        raise ValueError("Alguna división está vacía.")
    return {split: sum(row["split"] == split for row in records) for split in ("train", "val", "test")}


def dataset_yaml(path):
    # JSON entre comillas también es un escalar YAML seguro, sin dependencia PyYAML.
    return ("# Piloto de residuos; pendiente de revisión local, no modelo validado.\n"
            f"path: {json.dumps(path.resolve().as_posix(), ensure_ascii=False)}\n"
            "train: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: residuos\n")


def audit_dataset(path):
    """Revisa ficheros publicados, hashes, pares imagen/etiqueta y separación de lotes."""
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    if not isinstance(records, list) or not 1 <= len(records) <= 500 or len(records) != manifest["count"]:
        raise ValueError("Número de imágenes del manifiesto inválido.")
    if manifest.get("class_names") != {"0": "residuos"}:
        raise ValueError("El piloto solo contiene la clase residuos.")
    stats = audit_records(records)
    images, labels = set(), set()
    boxes = 0
    for row in records:
        expected_image = f"images/{row['split']}/{row['id']}.jpg"
        expected_label = f"labels/{row['split']}/{row['id']}.txt"
        if row["image_path"] != expected_image or row["label_path"] != expected_label:
            raise ValueError("Ruta de imagen o etiqueta inesperada.")
        image_path, label_path = (root / row[key] for key in ("image_path", "label_path"))
        if not all(p.resolve().is_relative_to(root) for p in (image_path, label_path)):
            raise ValueError("Un archivo sale del directorio del piloto.")
        if row["image_path"] in images or row["label_path"] in labels:
            raise ValueError("Ruta repetida en el manifiesto.")
        images.add(row["image_path"])
        labels.add(row["label_path"])
        raw, label_raw = image_path.read_bytes(), label_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"] or hashlib.sha256(label_raw).hexdigest() != row["sha256_labels"]:
            raise ValueError("Un archivo no coincide con su hash.")
        validate_image(raw)
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError("Dimensiones publicadas distintas del manifiesto.")
        if f"{perceptual_hash(raw):016x}" != row["dhash"]:
            raise ValueError("Hash perceptual distinto del manifiesto.")
        lines = label_raw.decode("utf-8").splitlines()
        if not lines or len(lines) != row["boxes"]:
            raise ValueError("Cajas incompletas o negativo no revisado.")
        for line in lines:
            fields = line.split()
            if len(fields) != 5 or fields[0] != "0":
                raise ValueError("Etiqueta YOLO inválida.")
            cx, cy, w, h = map(float, fields[1:])
            if (not all(math.isfinite(v) for v in (cx, cy, w, h)) or w <= 0 or h <= 0
                    or cx-w/2 < -1e-7 or cy-h/2 < -1e-7 or cx+w/2 > 1+1e-7 or cy+h/2 > 1+1e-7):
                raise ValueError("Caja YOLO fuera de la imagen.")
        boxes += len(lines)
    actual_images = {p.relative_to(root).as_posix() for p in (root / "images").rglob("*") if p.is_file()}
    actual_labels = {p.relative_to(root).as_posix() for p in (root / "labels").rglob("*") if p.is_file()}
    if actual_images != images or actual_labels != labels:
        raise ValueError("Hay imágenes o etiquetas extra/sin pareja.")
    if stats != manifest["splits"]:
        raise ValueError("Las divisiones no coinciden con el manifiesto.")
    if (root / "dataset.yaml").read_text(encoding="utf-8") != dataset_yaml(root):
        raise ValueError("YAML distinto del piloto local esperado.")
    return {"images": len(records), "boxes": boxes, "splits": stats,
            "review_status": manifest.get("review_status"), "trained": manifest.get("trained", False)}
