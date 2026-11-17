"""Descarga un piloto público TACO con trazabilidad; no entrena ni consulta OpenAI."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from valor_a_la_vista.dataset import (TACO_ANNOTATIONS_URL, TACO_RECORD_URL, TACO_COMMIT, TACO_SOURCE_URL,
    outdoor_candidates, coco_boxes_to_yolo, normalize_training_image, split_by_group, audit_records, audit_dataset, dataset_yaml)

DATASETS = ROOT / ".data" / "datasets"
ALLOWED_HOSTS = {"raw.githubusercontent.com", "zenodo.org", "img.tacodataset.org"}
MAX_IMAGE_BYTES = 12 * 1024 * 1024


def download(url, *, limit=MAX_IMAGE_BYTES):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or parsed.port not in (None, 443)
            or parsed.username or parsed.password or parsed.query or parsed.fragment or "\\" in url
            or any(char.isspace() for char in url)):
        raise ValueError("URL fuera de las fuentes públicas permitidas.")
    # No usa la clave de la app, cookies de sesión ni credenciales de .netrc.
    for attempt in range(2):
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.get(url, timeout=(10, 40), stream=True, allow_redirects=False,
                                 headers={"User-Agent": "ValorALaVista-dataset-pilot/1.0"}) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"HTTP {response.status_code} al descargar recurso público.")
                    chunks, size = [], 0
                    for chunk in response.iter_content(64 * 1024):
                        size += len(chunk)
                        if size > limit:
                            raise ValueError("Recurso público demasiado grande para el límite del piloto.")
                        chunks.append(chunk)
                    return b"".join(chunks)
        except requests.RequestException:
            if attempt == 1:
                raise RuntimeError("No se pudo descargar el recurso público.") from None
    raise RuntimeError("No se pudo descargar el recurso público.")


def save_new(path, raw):
    """Nunca sobrescribe archivos distintos; permite reusar exactamente lo descargado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Se conserva el archivo existente; el contenido difiere: {path.name}")
        return
    with path.open("xb") as handle:
        handle.write(raw)


def fetch_cached(url, path, *, limit=MAX_IMAGE_BYTES):
    if path.is_file():
        if path.stat().st_size > limit:
            raise ValueError("Archivo existente supera el límite de descarga.")
        return path.read_bytes()
    raw = download(url, limit=limit)
    save_new(path, raw)
    return raw


def prepare_one(candidate, cache, category_ids):
    image = candidate["image"]
    try:
        labels = coco_boxes_to_yolo(candidate["annotations"], image["width"], image["height"], category_ids=category_ids)
        url = image["flickr_640_url"]
        raw = fetch_cached(url, cache / f"taco_{image['id']:06d}.jpg")
        encoded, width, height, digest, phash = normalize_training_image(raw, image["width"], image["height"])
        source_id = Path(urlsplit(image.get("flickr_url", "")).path).stem
        record = {"id": f"taco_{image['id']:06d}", "source_image_id": image["id"], "group": candidate["group"],
            "source_file": image["file_name"], "source_url": url, "source_original_url": image.get("flickr_url"),
            "original_photo_page": f"https://www.flickr.com/photo.gne?id={source_id}" if source_id.isdigit() else None,
            "scene_ids": candidate["scene_ids"], "source_width": image["width"], "source_height": image["height"],
            "width": width, "height": height, "sha256_download": hashlib.sha256(raw).hexdigest(), "sha256": digest,
            "dhash": f"{phash:016x}", "source_annotation_ids": [row["id"] for row in candidate["annotations"]],
            "source_category_ids": sorted({row["category_id"] for row in candidate["annotations"]}),
            "boxes": len(labels.splitlines()), "dataset_license": "CC-BY-4.0 (declarada en Zenodo)",
            "individual_license_in_annotations": image.get("license"), "individual_license_verified": False,
            "human_review": "pendiente", "changes": "Versión 640px, JPEG normalizado sin EXIF; 60 clases TACO fusionadas en residuos; cajas normalizadas sin recorte."}
        return record, encoded, labels, None
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        return None, None, None, {"source_image_id": image.get("id"), "reason": str(exc)}


def review_html(records):
    e = lambda v: html.escape(str(v), quote=True)
    cards = []
    for row in records:
        boxes = []
        for line in row["labels"].splitlines():
            _, cx, cy, w, h = line.split()
            cx, cy, w, h = map(float, (cx, cy, w, h))
            boxes.append(f'<rect x="{cx-w/2:.8f}" y="{cy-h/2:.8f}" width="{w:.8f}" height="{h:.8f}"/>')
        cards.append(f'<article><h2>{e(row["id"])} · {e(row["split"])}</h2><div class="frame">'
                     f'<img loading="lazy" src="{e(row["image_path"])}" alt="Imagen pública TACO para revisión">'
                     f'<svg viewBox="0 0 1 1" preserveAspectRatio="none">{"".join(boxes)}</svg></div>'
                     f'<p>{row["boxes"]} cajas · {e(row["group"])} · revisión pendiente</p>'
                     f'<p>Clases originales: {e(row["source_category_ids"])}</p>'
                     f'<a href="{e(row["source_url"])}" rel="noreferrer">Imagen de origen</a></article>')
    return ('<!doctype html><html lang="es"><meta charset="utf-8"><title>Revisión piloto TACO</title>'
            '<style>body{font:15px system-ui;margin:24px;color:#172b28}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}'
            'article{border:1px solid #ccd5d2;padding:12px}h2{font-size:16px}.frame{position:relative;width:100%}img{display:block;width:100%}'
            'svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}rect{stroke:#ff7100;stroke-width:.004;fill:none}</style>'
            '<h1>Piloto público TACO · residuos</h1><p>Las cajas provienen del dataset, no han sido revisadas localmente. '
            'Comprueba entorno, cajas, imágenes casi repetidas, datos personales y permisos de reutilización. '
            'No usar como evaluación independiente de terrenos peruanos. No hay ejemplos negativos verificados.</p>'
            '<main>' + ''.join(cards) + '</main></html>')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DATASETS / "terrenos" / "taco_residuos_piloto_v1")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not 30 <= args.count <= 500 or not 1 <= args.workers <= 4:
        parser.error("El piloto admite 30–500 imágenes y 1–4 descargas simultáneas.")
    if not output.is_relative_to(DATASETS.resolve()):
        parser.error("El destino debe estar dentro de .data/datasets.")
    if output.exists():
        parser.error("El destino ya existe. Se conserva sin cambios; elige una carpeta nueva.")
    if shutil.disk_usage(ROOT).free < 512 * 1024 * 1024:
        parser.error("Se necesitan al menos 512 MB libres para preparar el piloto.")
    sources = DATASETS / "fuentes" / "taco" / TACO_COMMIT
    annotations_raw = fetch_cached(TACO_ANNOTATIONS_URL, sources / "annotations.json", limit=30 * 1024 * 1024)
    record_raw = fetch_cached(TACO_RECORD_URL, sources / "zenodo_record_3587843.json", limit=1024 * 1024)
    coco, license_record = json.loads(annotations_raw), json.loads(record_raw)
    if license_record["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("La licencia pública declarada no coincide con la esperada; revisar antes de descargar imágenes.")
    # Algunos lotes proceden de OpenLitterMap/ODbL y no tienen miniatura TACO.
    # No mezclar licencias ni reinterpretarlas como CC-BY por la licencia global.
    excluded_sources = []
    selected_images = []
    for image in coco["images"]:
        url = image.get("flickr_640_url")
        if (image.get("license") not in (None, "CC BY 4.0", "CC-BY-4.0")
                or not isinstance(url, str) or urlsplit(url).hostname != "img.tacodataset.org"):
            excluded_sources.append({"source_image_id": image["id"], "license": image.get("license"),
                                     "reason": "Licencia individual diferente o sin miniatura en la fuente TACO permitida; no se descarga."})
        else:
            selected_images.append(image)
    groups = outdoor_candidates({**coco, "images": selected_images}, seed=args.seed)
    if len(groups) < 3:
        raise ValueError("No hay suficientes grupos exteriores.")
    category_ids = {row["id"] for row in coco["categories"]}
    records, content, rejected = [], {}, []
    accepted_hashes, accepted_phashes = set(), []
    # Orden intercalado de lotes: permite compensar lotes pequeños sin mezclar licencias.
    group_names = list(groups)
    ordered = [groups[group][index] for index in range(max(map(len, groups.values())))
               for group in group_names if index < len(groups[group])]
    cursor = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while len(records) < args.count and cursor < len(ordered):
            chunk = ordered[cursor:cursor + min(args.workers, args.count - len(records))]
            cursor += len(chunk)
            for record, encoded, labels, error in executor.map(lambda item: prepare_one(item, sources / "images_640", category_ids), chunk):
                if error:
                    rejected.append(error)
                    continue
                phash = int(record["dhash"], 16)
                if record["sha256"] in accepted_hashes or any((phash ^ previous).bit_count() <= 4 for previous in accepted_phashes):
                    rejected.append({"source_image_id": record["source_image_id"], "reason": "Duplicado exacto o posible similitud dHash <=4; excluido conservadoramente, sin afirmar identidad física."})
                    continue
                accepted_hashes.add(record["sha256"])
                accepted_phashes.append(phash)
                records.append(record)
                content[record["id"]] = (encoded, labels)
            if len(records) % 25 < args.workers or len(records) == args.count:
                print(f"Piloto: {len(records)}/{args.count} imágenes aceptadas; descartes técnicos: {len(rejected)}", flush=True)
    if len(records) != args.count:
        raise RuntimeError(f"Solo se pudieron preparar {len(records)}/{args.count}. Se conserva la caché de fuentes; no se publica un dataset parcial.")
    records = split_by_group(records, seed=args.seed)
    stats = audit_records(records)
    output.mkdir(parents=True, exist_ok=False)
    review_records = []
    for row in records:
        encoded, labels = content[row["id"]]
        row["image_path"] = f"images/{row['split']}/{row['id']}.jpg"
        row["label_path"] = f"labels/{row['split']}/{row['id']}.txt"
        row["sha256_labels"] = hashlib.sha256(labels.encode("utf-8")).hexdigest()
        save_new(output / row["image_path"], encoded)
        save_new(output / row["label_path"], labels.encode("utf-8"))
        review_records.append({**row, "labels": labels})
    manifest = {"dataset": "TACO piloto exterior de residuos", "created_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(records), "seed": args.seed, "splits": stats, "class_names": {"0": "residuos"},
        "source": TACO_SOURCE_URL, "source_commit": TACO_COMMIT, "annotations_url": TACO_ANNOTATIONS_URL,
        "sha256_annotations": hashlib.sha256(annotations_raw).hexdigest(), "dataset_license_record": "https://zenodo.org/records/3587843",
        "license": "CC-BY-4.0 según la publicación del dataset; metadatos por imagen sin licencia individual especificada",
        "attribution": "Pedro F. Proença y Pedro Simões, TACO: Trash Annotations in Context for Litter Detection, 2020; conservar atribución y procedencia individual.",
        "review_status": "pendiente", "trained": False, "negative_examples": 0,
        "limitations": ["Piloto de residuos, no dataset completo de terrenos ni imágenes necesariamente peruanas.",
            "Selección por escenas tierra/vegetación sin interior según anotaciones de origen; requiere revisión visual.",
            "Lotes separados entre train/val/test; no hay identificadores fiables de terreno/autor: no garantiza independencia de escenas físicas.",
            "El filtro dHash es heurístico, no demuestra ni descarta todas las repeticiones.",
            "No hay negativos verificados; añadir negativos revisados antes de entrenar.",
            "No incluye anotaciones de maleza, agua, erosión ni otros daños; no tratarlas como clases etiquetadas.",
            "Versión de imágenes 640px: revisar objetos pequeños; no inventar cajas ni negativos.",
            "Licencia del dataset registrada; no se ha verificado independientemente el permiso de cada fotografía ni datos personales."],
        "source_images_excluded_by_policy": excluded_sources,
        "rejected": rejected, "records": records}
    save_new(output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    save_new(output / "dataset.yaml", dataset_yaml(output).encode("utf-8"))
    save_new(output / "revision.html", review_html(review_records).encode("utf-8"))
    save_new(output / "LICENSE-DATASET.txt", ("TACO dataset: CC BY 4.0 según https://zenodo.org/records/3587843\n"
        "https://creativecommons.org/licenses/by/4.0/\nhttps://creativecommons.org/licenses/by/4.0/legalcode\n"
        + manifest["attribution"] + f"\n\nModificaciones: selección de escenas exteriores, subconjunto de {len(records)}, reducción/normalización JPEG sin EXIF y fusión de categorías a residuos.\n"
        "Las anotaciones de origen no especifican licencia individual por foto; conservar manifest y revisar derechos antes de redistribuir.\n").encode("utf-8"))
    save_new(output / "README.md", ("# Piloto TACO · residuos\n\n"
        f"{len(records)} imágenes con etiquetas COCO convertidas a YOLO; splits: {stats}. Clase única `0: residuos`.\n\n"
        "**No entrenado. Revisión humana y negativos pendientes.** Abre `revision.html` para inspeccionar todas las cajas.\n"
        "Conserva `manifest.json`, `LICENSE-DATASET.txt` y atribución al reutilizar. No es un conjunto de validación local de terrenos peruanos.\n"
        "No mezclar lotes entre divisiones ni presentar este piloto como anotaciones de maleza/agua.\n").encode("utf-8"))
    audit_dataset(output)
    print(json.dumps({"output": str(output), "images": len(records), "boxes": sum(row["boxes"] for row in records),
                      "splits": stats, "rejected": len(rejected), "review_status": "pendiente", "trained": False}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
