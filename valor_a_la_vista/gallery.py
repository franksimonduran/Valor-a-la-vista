"""Galería de un mismo terreno, conservada únicamente en la sesión."""
from __future__ import annotations

import hashlib

from .core import MAX_PHOTOS, MAX_TOTAL_PHOTO_BYTES
from .vision import validate_image


def add_photos(gallery, uploads):
    """Devuelve una copia; rechaza duplicados y límites sin perder fotos previas."""
    result = dict(gallery)
    notices = []
    for name, raw in uploads:
        ident = hashlib.sha256(raw).hexdigest()
        if ident in result:
            notices.append(f"{name}: la fotografía ya está añadida.")
            continue
        try:
            validate_image(raw)
            if len(result) >= MAX_PHOTOS:
                raise ValueError(f"Máximo {MAX_PHOTOS} fotografías por terreno.")
            if sum(len(item['raw']) for item in result.values()) + len(raw) > MAX_TOTAL_PHOTO_BYTES:
                raise ValueError("Máximo 64 MB en total por terreno.")
        except ValueError as exc:
            notices.append(f"{name}: {exc}")
            continue
        result[ident] = {"nombre": str(name)[:160], "raw": raw, "categoria": "Detalle adicional", "origen": {"tipo": "foto"}}
    return result, notices


def inspection_inputs(gallery):
    photos, categories = {}, {}
    for ident, item in gallery.items():
        label = f"Foto {ident[:12]} · {item['nombre']}"
        photos[label] = item["raw"]
        categories[label] = item["categoria"]
    return photos, categories


def add_video_frames(gallery, frames):
    """Mismos límites que las fotos; conserva evidencia de origen por fotograma."""
    result = dict(gallery)
    notices = []
    for frame in frames:
        ident = hashlib.sha256(frame["raw"]).hexdigest()
        previous = ident in result
        result, notes = add_photos(result, [(frame["nombre"], frame["raw"])])
        notices.extend(notes)
        if not previous and ident in result:
            result[ident]["origen"] = dict(frame["origen"])
    return result, notices


def media_origins(gallery):
    return {f"Foto {ident[:12]} · {item['nombre']}": dict(item.get("origen", {"tipo": "foto"}))
            for ident, item in gallery.items()}
