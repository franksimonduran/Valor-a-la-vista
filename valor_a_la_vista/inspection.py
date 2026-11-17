"""Orquestación de inspecciones, sin dependencias de la interfaz."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

from .core import ASSETS, MAX_PHOTOS, MAX_TOTAL_PHOTO_BYTES, combine_findings, validate_damage_findings
from .vision import DEFAULT_API_BASE, PROMPT_VERSION, SCHEMA_VERSION, inspect_image, unique_photo_hashes
from .usage_limits import QuotaExceeded


def photo_fingerprint(photos, photo_categories=None, origins=None):
    digest = hashlib.sha256()
    for name, raw in sorted(photos.items()):
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(raw).digest())
        digest.update((photo_categories or {}).get(name, name).encode("utf-8"))
        if origins is not None:
            digest.update(context_fingerprint(origins.get(name, {})).encode("ascii"))
    return digest.hexdigest()


def context_fingerprint(*values):
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def analysis_is_current(analysis, signature, model, base_url=DEFAULT_API_BASE):
    metadata = (analysis or {}).get("metadata", {})
    return bool(analysis and analysis.get("signature") == signature
                and metadata.get("modelo") == model
                and metadata.get("proveedor") == "openai"
                and metadata.get("api_base") == base_url
                and metadata.get("version_prompt") == PROMPT_VERSION
                and metadata.get("version_esquema") == SCHEMA_VERSION)


def analyze_photos(asset, photos, *, api_key, model, cache=None, force=False,
                    progress=None, inspector=None, base_url=DEFAULT_API_BASE, photo_categories=None, origins=None,
                    before_api_call=None):
    """Reutiliza cada foto sin cambios; solo un reanálisis explícito fuerza la API."""
    if asset not in ASSETS or not photos:
        raise ValueError("Selecciona un activo válido y al menos una foto.")
    if len(photos) > MAX_PHOTOS or sum(len(raw) for raw in photos.values()) > MAX_TOTAL_PHOTO_BYTES:
        raise ValueError("Se permiten hasta 16 fotografías y 64 MB por terreno.")
    if unique_photo_hashes(photos)[1]:
        raise ValueError("Quita las fotografías idénticas antes de analizar.")
    photo_categories = photo_categories or {name: name for name in photos}
    cache = {} if cache is None else cache
    inspector = inspector or inspect_image
    analyses, summaries, materials, warnings, failures, observations = [], [], [], [], [], []
    damages = []
    detail_complete = True
    for index, (name, raw) in enumerate(photos.items()):
        try:
            digest = hashlib.sha256(raw).hexdigest()
            category = photo_categories.get(name, "Detalle adicional")
            key = (asset, name, digest, category, model, base_url, PROMPT_VERSION, SCHEMA_VERSION)
            if force or key not in cache:
                if before_api_call is not None:
                    before_api_call()
                result = inspector(raw, asset, category, model=model, api_key=api_key, base_url=base_url)
                if not result.get("hallazgos"):
                    raise RuntimeError("La respuesta no contiene criterios válidos; vuelve a intentarlo.")
                if len(cache) >= 64:
                    cache.pop(next(iter(cache)))
                cache[key] = {"result": result, "fecha": datetime.now().astimezone().isoformat(timespec="seconds")}
            entry = cache[key]
            result = entry["result"]
            details, detail_warnings, valid = validate_damage_findings(asset, result.get("danos"))
            detail_complete = detail_complete and valid and result.get("detalle_danos_valido", valid)
            warnings.extend(f"{name}: {warning}" for warning in detail_warnings)
            analyses.append(result["hallazgos"])
            # Una lesión visible no puede quedar oculta detrás de un 'bueno'.
            visible = [{"criterio": item["criterio"], "estado": "malo" if item["severidad"] == "alta" else "regular",
                        "evidencia": item["evidencia"]} for item in details if item["certeza"] == "visible"]
            if visible:
                analyses.append(visible)
            for item in details:
                identity = context_fingerprint(asset, name, digest, item["tipo"], item["criterio"],
                                               item["elemento"].casefold(), item["ubicacion"].casefold())[:20]
                damages.append({**item, "id": "d_" + identity, "foto": name,
                                 "fecha_analisis": entry["fecha"], "sha256": digest,
                                 "origen": dict((origins or {}).get(name, {"tipo": "foto"}))})
            summaries.append((name, result.get("observacion_general", "")))
            warnings.extend(f"{name}: {warning}" for warning in result.get("advertencias", []))
            materials.extend({"foto": name, **item} for item in result.get("materiales_visibles", []))
            observations.append({"foto": name, "sha256": digest, "fecha_analisis": entry["fecha"],
                                  "hallazgos": result["hallazgos"], "danos": details,
                                  "origen": dict((origins or {}).get(name, {"tipo": "foto"}))})
        except Exception as exc:
            # Si agotamos la cuota, no hacer más consultas de fotos.
            failures.append(f"{name}: {exc}")
            detail_complete = False
            if isinstance(exc, QuotaExceeded):
                break
        if progress:
            progress((index + 1) / len(photos), name)
    statuses, evidence = combine_findings(asset, analyses)
    for code in statuses:
        states = {item.get("estado") for findings in analyses for item in findings
                  if item.get("criterio") == code and item.get("estado") in {"bueno", "regular", "malo"}}
        if len(states) > 1:
            warnings.append(f"Discrepancias entre propuestas visuales para {code}; se conserva el peor estado y requiere revisión.")
    if len({item["foto"] for item in damages}) > 1:
        warnings.append("Los daños de distintas fotos no se suman como superficies independientes. Agrupa los que correspondan a una misma zona física antes de presupuestar.")
    if any(item.get("tipo") == "video" for item in (origins or {}).values()):
        warnings.append("Video analizado por muestreo de fotogramas, sin audio: puede omitir daños entre capturas. Un daño repetido en varios momentos no es más superficie ni otra partida.")
    missing_details = {code for code, status in statuses.items() if status in {"regular", "malo"}} - {item["criterio"] for item in damages}
    if missing_details:
        detail_complete = False
        warnings.append("Deterioro sin lesión concreta identificada: " + ", ".join(sorted(missing_details)) + ". Repite una toma de detalle.")
    covered = sorted({photo_categories.get(name) for name, _ in summaries} & {name for name, _ in ASSETS[asset]["fotos"]})
    photos_complete = len(covered) == len(ASSETS[asset]["fotos"]) and not failures
    return {
        "signature": photo_fingerprint(photos, photo_categories, origins), "analizadas": [name for name, _ in summaries],
        "categorias_cubiertas": covered, "fotos_completas": photos_complete,
        "total": len(photos), "fallos": failures, "advertencias": warnings,
        "statuses": statuses, "evidence": evidence, "summary": summaries, "materials": materials,
        "danos": damages, "detalle_completo": detail_complete,
        "proposal": {code: {"estado": state, "evidencia": evidence.get(code, "")}
                     for code, state in statuses.items()},
        "metadata": {"modelo": model, "proveedor": "openai", "api_base": base_url, "version_prompt": PROMPT_VERSION,
                      "api_endpoint": "responses", "origenes_por_foto": origins or {},
                      "categorias_por_foto": dict(photo_categories), "categorias_cubiertas": covered,
                     "version_esquema": SCHEMA_VERSION, "fotos_analizadas": [n for n, _ in summaries],
                     "fallos": failures.copy(), "advertencias": warnings.copy(),
                      "evidencias_por_foto": observations,
                      "detalle_danos_completo": detail_complete,
                     "fecha_procesamiento": datetime.now().astimezone().isoformat(timespec="seconds")},
    }


def analysis_message(analysis, evaluation, expected_photos):
    processed = len(analysis["analizadas"])
    if not processed:
        return "error", "Sin resultados. Revisa los fallos."
    if evaluation.verified_criteria == 0:
        return "warning", "Fotos procesadas, pero ningún criterio es verificable. No se puede concluir ausencia de daños."
    if not analysis.get("detalle_completo", False):
        return "warning", "Procesamiento con detalle de lesiones incompleto; revisa avisos y repite las tomas necesarias."
    if not analysis.get("fotos_completas", False) or evaluation.coverage < 100:
        return "warning", (f"Cobertura parcial: {len(analysis.get('categorias_cubiertas', []))} de {expected_photos} vistas guiadas; "
                           f"{evaluation.coverage:g} % de criterios observables.")
    return "success", f"Procesamiento terminado: {processed} tomas; criterios observables completos, sin validación técnica."
