
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from datetime import date
from pathlib import Path

from valor_a_la_vista import bootstrap 
import streamlit as st

if __name__ == "__main__":
    from streamlit import runtime
    if not runtime.exists():
        import sys
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", __file__,
                    "--server.showEmailPrompt", "false",
                    "--browser.gatherUsageStats", "false"]
        sys.exit(stcli.main())

from valor_a_la_vista.core import (
    ASSETS, STATUS_LABELS, DAMAGE_CATALOG, SEVERITY_LABELS, CERTAINTY_LABELS,
    WORK_STAGES, UNIT_LABELS, evaluate, recommend_improvements,
    MAX_PHOTOS,
)
from valor_a_la_vista.pricing import build_damage_budget
from valor_a_la_vista.inspection import (
    analyze_photos, analysis_is_current, analysis_message, context_fingerprint, photo_fingerprint,
)
from valor_a_la_vista.online_prices import search_repair_references
from valor_a_la_vista.gallery import add_photos, add_video_frames, inspection_inputs, media_origins
from valor_a_la_vista.video import extract_video_frames, MAX_VIDEO_FRAMES, MAX_VIDEO_BYTES
from valor_a_la_vista.detector import detector_status, detector_card, load_terrain_detector, detect_regions, annotated_image
from valor_a_la_vista.report import build_report_data, html_report, report_digest
from valor_a_la_vista import ui
from valor_a_la_vista.vision import (
    PROMPT_VERSION, openai_config, image_quality_warnings,
    perceptual_hash, prepare_image, unique_photo_hashes,
)


def active_mode():
    """Tema activo según el cliente; claro si aún no está disponible."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


st.set_page_config(
    page_title="Valor A la Vista · Mejoras de terrenos",
    page_icon=str(Path(__file__).resolve().parent / "assets" / "logo.svg"),
    layout="wide", initial_sidebar_state="expanded"
)

st.markdown(ui.styles(active_mode()), unsafe_allow_html=True)


def ai_access_granted():
    """Restringe las consultas de IA con coste sin ocultar la interfaz pública."""
    required = os.getenv("APP_REQUIRE_PASSWORD", "").strip().lower() in {"1", "true", "yes"}
    expected = os.getenv("APP_ACCESS_PASSWORD", "")
    if not expected:
        if required:
            with st.sidebar:
                st.warning("IA deshabilitada: configura APP_ACCESS_PASSWORD en Render.")
            return False
        return True
    with st.sidebar:
        supplied = st.text_input("Contraseña para habilitar IA", type="password", key="vav_ai_access")
        allowed = hmac.compare_digest(supplied, expected)
        st.caption("IA habilitada en esta sesión." if allowed else "Las consultas de IA requieren contraseña.")
        return allowed


ai_access = ai_access_granted()


def cached_report_image(raw):
    digest = hashlib.sha256(raw).hexdigest()
    cache = st.session_state.setdefault("report_image_cache", {})
    if digest not in cache:
        if len(cache) >= 8:
            cache.pop(next(iter(cache)))
        cache[digest] = prepare_image(raw)
    return cache[digest]


def cached_photo_details(raw):
    digest = hashlib.sha256(raw).hexdigest()
    cache = st.session_state.setdefault("photo_details_cache", {})
    if digest not in cache:
        if len(cache) >= 32:
            cache.pop(next(iter(cache)))
        cache[digest] = (image_quality_warnings(raw), perceptual_hash(raw), digest)
    return cache[digest]


@st.cache_resource(max_entries=1, show_spinner=False)
def cached_detector(weights_hash):
    """Modelo compartido, invalidado al cambiar los pesos evaluados."""
    model, actual_hash = load_terrain_detector()
    if actual_hash != weights_hash:
        raise ValueError("El modelo activo cambió; vuelve a ejecutar la detección.")
    return model, threading.RLock()


def reset_inspection(asset_code):
    """Limpia únicamente los datos del activo seleccionado."""
    markers = (f"_{asset_code}", f"_{asset_code}_")
    for key in list(st.session_state):
        if any(marker in key for marker in markers):
            del st.session_state[key]
    st.session_state.pop("photo_details_cache", None)
    st.session_state.pop("report_image_cache", None)


def run_analysis(asset, photos, api_key, model_name, force=False, photo_categories=None, base_url=None, origins=None):
    """Actualiza la inspección sin repetir fotos ya procesadas."""
    cache = st.session_state.setdefault(f"analysis_cache_{asset}", {})
    bar = st.progress(0, text="Analizando tomas...")
    try:
        st.session_state[f"analysis_{asset}"] = analyze_photos(
            asset, photos, api_key=api_key, model=model_name, cache=cache, force=force,
            photo_categories=photo_categories, base_url=base_url, origins=origins,
            progress=lambda value, name: bar.progress(value, text=f"Procesando {name}"),
        )
    finally:
        bar.empty()


masthead_slot = st.empty()
rail_slot = st.empty()

with st.sidebar:
    st.markdown(ui.sidebar_brand(), unsafe_allow_html=True)
    asset = "terreno"
    st.markdown(ui.sidebar_meta("Especialidad", "Mejoras de terrenos"), unsafe_allow_html=True)
    try:
        api_key, model_name, base_url = openai_config()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.markdown(ui.sidebar_meta("Modelo OpenAI", model_name), unsafe_allow_html=True)
    if not api_key:
        st.markdown(ui.callout("warning", "Falta configurar OPENAI_API_KEY."), unsafe_allow_html=True)
    st.markdown(ui.rule(), unsafe_allow_html=True)
    with st.expander("Nueva inspección"):
        st.caption("Borra las fotos y resultados de este terreno de la sesión.")
        st.button("Reiniciar terreno", on_click=reset_inspection, args=(asset,))
    with st.expander("Privacidad y límites"):
        st.markdown(ui.callout("note", "Fotos y fotogramas en memoria durante la sesión. El video usa un archivo temporal local que se elimina al terminar la extracción; no se transmite video ni audio."), unsafe_allow_html=True)
        st.markdown(ui.callout("warning", "El resultado no equivale a un peritaje, una tasación ni una aprobación de crédito."), unsafe_allow_html=True)
    st.markdown(ui.sidebar_footer(), unsafe_allow_html=True)

spec = ASSETS[asset]
st.markdown(ui.section_header(1, "Fotos y videos", "Carga una foto o un video y podrás inspeccionarlo sin completar varias vistas."), unsafe_allow_html=True)
st.caption("Un solo terreno por inspección. Hasta 16 fotos, 12 MB por foto y 64 MB en total; JPEG, PNG o WebP. Evita personas y documentos.")
st.caption("Vistas guiadas: " + " · ".join(f"{name}: {instruction}" for name, instruction in spec["fotos"]))
stored = st.session_state.setdefault(f"gallery_{asset}", {})
upload_version = st.session_state.get(f"upload_version_{asset}", 0)
source = st.radio("Añadir desde", ["Dispositivo", "Cámara", "Video"], horizontal=True, key=f"source_{asset}")
uploads = []
if source == "Dispositivo":
    selected = st.file_uploader("Seleccionar más imágenes", type=["jpg", "jpeg", "png", "webp"],
                                accept_multiple_files=True, max_upload_size=12, key=f"upload_{asset}_{upload_version}")
    uploads = [(item.name, item.getvalue()) for item in (selected or [])]
elif source == "Cámara":
    selected = st.camera_input("Tomar otra fotografía", key=f"camera_{asset}_{upload_version}")
    if selected is not None:
        uploads = [(selected.name, selected.getvalue())]
else:
    st.caption(f"Video corto de un mismo terreno: MP4, MOV, AVI o WebM, máximo 60 segundos, {MAX_VIDEO_BYTES // (1024 * 1024)} MB y resolución 4K. No se analiza audio.")
    selected_video = st.file_uploader("Seleccionar video del terreno", type=["mp4", "mov", "avi", "webm"],
                                     max_upload_size=MAX_VIDEO_BYTES // (1024 * 1024), key=f"video_{asset}_{upload_version}")
    frame_count = st.slider("Fotogramas a extraer", 1, MAX_VIDEO_FRAMES, 8, key=f"video_frames_{asset}")
    remaining = MAX_PHOTOS - len(stored)
    st.caption(f"Espacios disponibles para fotos y fotogramas: {remaining}. La extracción es local y no genera cargos de API.")
    video_selection_key = f"video_selection_{asset}"
    if selected_video is None:
        st.session_state.pop(video_selection_key, None)
    else:
        # La extracción local ocurre una vez por archivo/cantidad, sin consultar ni cobrar la API.
        video_bytes = selected_video.getvalue()
        selection = (hashlib.sha256(video_bytes).hexdigest(), frame_count)
        if st.session_state.get(video_selection_key) != selection:
            st.session_state[video_selection_key] = selection
            if remaining < 1:
                st.session_state[f"upload_notices_{asset}"] = [
                    "La galería está llena: quita una fotografía para preparar el video."
                ]
            else:
                try:
                    with st.spinner("Preparando fotogramas del video..."):
                        frames, notes = extract_video_frames(video_bytes, selected_video.name,
                                                            max_frames=min(frame_count, remaining))
                        stored, notices = add_video_frames(stored, frames)
                    st.session_state[f"gallery_{asset}"] = stored
                    st.session_state[f"upload_notices_{asset}"] = [*notes, *notices]
                except (ValueError, RuntimeError, OSError) as exc:
                    st.session_state[f"upload_notices_{asset}"] = [f"Video no procesado: {exc}"]

if source != "Video":
    # Guardar desde la primera toma, sin botón extra ni recargas infinitas.
    upload_selection_key = f"upload_selection_{asset}_{source}"
    if not uploads:
        st.session_state.pop(upload_selection_key, None)
    else:
        fingerprint = hashlib.sha256()
        for name, raw in uploads:
            fingerprint.update(str(name).encode("utf-8", errors="replace"))
            fingerprint.update(hashlib.sha256(raw).digest())
        selection = fingerprint.hexdigest()
        if st.session_state.get(upload_selection_key) != selection:
            st.session_state[upload_selection_key] = selection
            stored, notices = add_photos(stored, uploads)
            st.session_state[f"gallery_{asset}"] = stored
            st.session_state[f"upload_notices_{asset}"] = notices
for notice in st.session_state.get(f"upload_notices_{asset}", []):
    st.warning(notice)
categories = ["Detalle adicional", *[name for name, _ in spec["fotos"]]]
gallery = st.grid(2, min_column_width=300, gap="small", key="photo_gallery")
for ident, item in list(stored.items()):
    with gallery.cell():
        st.image(item["raw"], width="stretch", caption=item["nombre"])
        if item.get("origen", {}).get("tipo") == "video":
            origin = item["origen"]
            st.caption(f"Video: {origin['video_nombre']} · tiempo aproximado {origin['tiempo_aproximado']}. Puedes quitar este fotograma antes de enviarlo.")
        item["categoria"] = st.selectbox("Vista de esta foto", categories, index=categories.index(item["categoria"]), key=f"category_{asset}_{ident}")
        for note in cached_photo_details(item["raw"])[0]:
            st.caption("Aviso: " + note)
        if st.button("Quitar fotografía", key=f"remove_{asset}_{ident}"):
            del stored[ident]
            st.session_state.pop(f"category_{asset}_{ident}", None)
            st.rerun()
photos, photo_categories = inspection_inputs(stored)
origins = media_origins(stored)

unique_photos, duplicates = unique_photo_hashes(photos)
for name, original in duplicates.items():
    st.markdown(ui.callout("warning", f"Las tomas «{original}» y «{name}» son idénticas; solo cuentan una vez. Sustituye la copia."), unsafe_allow_html=True)
perceptual_hashes = {}
for name, raw in unique_photos.items():
    _, phash, _ = cached_photo_details(raw)
    for previous_name, previous_hash in perceptual_hashes.items():
        if (phash ^ previous_hash).bit_count() <= 4:
            st.caption(f"Aviso: «{previous_name}» y «{name}» podrían ser similares; compruébalo manualmente.")
    perceptual_hashes[name] = phash
photos = unique_photos
num_photos = len(photos)
covered_views = set(photo_categories.values()) & {name for name, _ in spec["fotos"]}
st.progress(len(covered_views) / len(spec["fotos"]), text=f"{num_photos}/{MAX_PHOTOS} fotos · {len(covered_views)}/4 vistas guiadas asignadas (no verificadas)")

with st.container(key="declared_details"):
    st.markdown(ui.details_heading(spec["nombre"]), unsafe_allow_html=True)
    area_column, location_column = st.columns([1, 2])
    with area_column:
        area = st.number_input(
            "Área declarada (m²)", min_value=0.0, value=0.0, step=1.0,
            key=f"area_{asset}", max_value=1_000_000.0,
            help="Contexto del inmueble; nunca se interpreta como superficie dañada."
        )
    with location_column:
        location = st.text_input(
            "Ubicación declarada", key=f"location_{asset}", max_chars=120,
            placeholder="Distrito, ciudad"
        )
    st.caption("Área y ubicación son declaradas, no verificadas.")

signature = photo_fingerprint(photos, photo_categories, origins)
analysis = st.session_state.get(f"analysis_{asset}")
if not analysis_is_current(analysis, signature, model_name, base_url):
    analysis = None

st.markdown(ui.section_header(2, "Análisis", "Propuesta automática de OpenAI; sin revisión humana."), unsafe_allow_html=True)
st.caption(f"Desde la primera foto o los primeros fotogramas del video puedes analizar. Al pulsar Analizar terreno, autorizas enviar {len(photos)} foto(s)/fotograma(s) a OpenAI (hasta una consulta por imagen nueva). Puede generar cargos. El video y el audio no se envían.")
with st.expander("Qué daños se buscan"):
    st.write("; ".join(spec.nombre for spec in DAMAGE_CATALOG[asset].values()))
    st.caption("El catálogo guía la revisión, no garantiza detectar todos los daños. Usa tomas generales y detalles nítidos desde un lugar seguro; si falta evidencia se requiere otra foto o inspección.")
analyze = st.button(
    "Analizar terreno", key="analyze", type="primary",
    disabled=not photos or not api_key or not ai_access,
    help="Reutiliza las fotos ya analizadas; solo procesa fotos nuevas o fallidas."
)
reanalyze = st.button("Reanalizar todas las fotos", key="reanalyze", disabled=not analysis or not api_key or not ai_access,
                      help="Fuerza nuevas llamadas a OpenAI y puede generar cargos.")
if not photos:
    st.markdown(ui.callout("note", "Selecciona una fotografía o un video: una sola toma es suficiente para comenzar. El análisis requiere que se prepare al menos un fotograma del video."), unsafe_allow_html=True)
elif not api_key:
    st.markdown(ui.callout("note", "Configura OPENAI_API_KEY para habilitar el análisis."), unsafe_allow_html=True)
elif not ai_access:
    st.markdown(ui.callout("warning", "Para analizar fotos, introduce la contraseña de IA en la barra lateral."), unsafe_allow_html=True)

if (analyze or reanalyze) and photos and api_key and ai_access:
    run_analysis(asset, photos, api_key, model_name, force=reanalyze, photo_categories=photo_categories, base_url=base_url, origins=origins)
    analysis = st.session_state.get(f"analysis_{asset}")

statuses = analysis["statuses"] if analysis else {}
evidence = analysis["evidence"] if analysis else {}
damages = analysis.get("danos", []) if analysis else []
result = evaluate(asset, statuses)
actions = recommend_improvements(asset, statuses)
has_results = bool(analysis and analysis["analizadas"])

local_detections = st.session_state.get(f"yolo_{asset}", {})
card = detector_card()
active_weights_hash = card["weights_sha256"] if card else None
if (local_detections.get("signature") != signature
        or local_detections.get("sha256_pesos") != active_weights_hash):
    local_detections = {}
with st.expander("Detección local con YOLO (opcional)"):
    ready, detector_note = detector_status()
    st.caption(detector_note)
    st.caption("YOLO local detecta residuos en fotos y fotogramas, sin enviar imágenes a una API. No detecta todavía maleza, agua, erosión ni todos los daños. Sus cajas no son m², diagnóstico ni partidas de presupuesto.")
    if card:
        metrics = card["test_fixed_threshold"]
        st.warning("Modelo experimental: puede omitir residuos y producir falsas alarmas. No está validado en terrenos locales ni para decisiones técnicas o crediticias.")
        st.caption(f"Prueba del piloto: mAP50 {card['test']['mAP50']:.1%}; precisión {metrics['precision']:.1%} y recall {metrics['recall']:.1%} a confianza 0,35 e IoU 0,50. No son una garantía de precisión en tu terreno.")
        accepted = st.checkbox("Entiendo que YOLO es experimental y solo detecta residuos", key=f"yolo_ack_{asset}_{active_weights_hash}")
    else:
        accepted = False
    if st.button("Detectar regiones con YOLO", key="detect_yolo", disabled=not photos or not ready or not accepted):
        try:
            with st.spinner("Ejecutando detector local en fotos/fotogramas..."):
                detector, detector_lock = cached_detector(active_weights_hash)
                weights_hash = active_weights_hash
                with detector_lock:
                    detections = [{"foto": name, "origen": origins.get(name, {}), "regiones": detect_regions(raw, detector)}
                                  for name, raw in photos.items()]
            local_detections = {"signature": signature, "sha256_pesos": weights_hash, "umbral": .35,
                                 "resultados": detections, "validado": False, "modelo": card,
                                "limitacion": "Propuestas de localización no verificadas; la confianza del detector no es probabilidad calibrada de daño. Sin efecto en score ni presupuesto."}
            st.session_state[f"yolo_{asset}"] = local_detections
        except (RuntimeError, ValueError, OSError):
            st.error("No se pudo ejecutar YOLO. Revisa dependencias, clases y pesos locales de terrenos. No se descargó un modelo alternativo.")
    if local_detections:
        st.download_button("Descargar detecciones YOLO (JSON)", json.dumps(local_detections, ensure_ascii=False, indent=2),
                           "detecciones_residuos_experimental.json", "application/json")
        for detection in local_detections["resultados"]:
            st.image(annotated_image(photos[detection["foto"]], detection["regiones"]), caption=detection["foto"], width="stretch")
            if detection["regiones"]:
                st.dataframe(detection["regiones"], hide_index=True)
            else:
                st.caption("Sin regiones propuestas por YOLO; no demuestra ausencia de daños.")

if analysis:
    kind, message = analysis_message(analysis, result, len(spec["fotos"]))
    st.markdown(ui.callout(kind, message), unsafe_allow_html=True)
    for failure in analysis["fallos"]:
        st.markdown(ui.callout("error", f"No se pudo analizar: {failure}"), unsafe_allow_html=True)
    for warning in analysis["advertencias"]:
        st.markdown(ui.callout("warning", warning), unsafe_allow_html=True)
    st.markdown(ui.score_panel(result.score, result.coverage, result.verified_criteria,
                               result.total_criteria, result.category, result.coverage < 100),
                unsafe_allow_html=True)
    st.caption("Lectura solo de criterios observados; no es un diagnóstico.")
    st.markdown("**Lesiones específicas propuestas**")
    if damages:
        st.dataframe([
            {"ID": item["id"], "Foto": item["foto"],
             "Momento del video (aprox.)": item.get("origen", {}).get("tiempo_aproximado", "—"), "Elemento": item["elemento"],
             "Ubicación en la foto": item["ubicacion"], "Daño aparente": item["dano"],
             "Severidad aparente": SEVERITY_LABELS[item["severidad"]],
             "Certeza declarada por IA": CERTAINTY_LABELS[item["certeza"]],
             "Evidencia": item["evidencia"], "Acción orientativa": item["intervencion_sugerida"],
             "Revisión previa": "Necesaria" if item["requiere_revision"] else "Contrastar alcance"}
            for item in damages
        ], width="stretch", hide_index=True)
        if any(item["requiere_revision"] for item in damages):
            st.markdown(ui.callout("warning", "Hay indicios inciertos o lesiones que requieren evaluación técnica antes de definir y presupuestar la reparación."), unsafe_allow_html=True)
    else:
        st.caption("No se propusieron lesiones específicas. No demuestra ausencia de daños; si un criterio está deteriorado, repite una toma cercana y nítida.")
    st.dataframe(
        [{"Criterio": title, "Estado": STATUS_LABELS[statuses.get(code, "no_visible")],
          "Evidencia": evidence.get(code, "")}
         for code, (title, _, _) in spec["criterios"].items()],
        width="stretch", hide_index=True
    )
    if analysis["materials"]:
        with st.expander("Materiales aparentes (sin verificación)"):
            st.dataframe(analysis["materials"], width="stretch", hide_index=True)
    if analysis["summary"]:
        with st.expander("Observaciones por foto"):
            for name, observation in analysis["summary"]:
                if observation:
                    st.caption(f"**{name}:** {observation}")

st.markdown(ui.section_header(3, "Costo aproximado de mejora", "S/ por m² afectado; otras unidades cuando corresponda. No es el valor del terreno."), unsafe_allow_html=True)
web_result = None
budget = None
if not has_results:
    st.markdown(ui.callout("note", "El resultado aparece cuando termina el análisis."), unsafe_allow_html=True)
else:
    damaged = [code for code in spec["criterios"] if statuses.get(code) in {"regular", "malo"}]
    reference_signature = context_fingerprint("damages-2", asset, statuses, evidence, damages, location.strip(), model_name, base_url, PROMPT_VERSION)
    references = st.session_state.get(f"references_{asset}", {})
    if references.get("signature") != reference_signature:
        references = {}
    with st.expander("Buscar fuentes oficiales (opcional)"):
        st.caption("Prioridad: MVCS, MTC y entidad competente. Envía ubicación y hallazgos a OpenAI con búsqueda web, sin reenviar fotos; puede generar cargos. No convierte índices o aranceles en costos de mejora.")
        st.link_button("Normas y documentos del MVCS", "https://www.gob.pe/institucion/vivienda/normas-legales")
        st.link_button("Normas y documentos del MTC", "https://www.gob.pe/institucion/mtc/normas-legales")
        st.link_button("INEI: índices, no tarifas de reparación", "https://www.gob.pe/inei")
        search = st.button("Buscar precios en Internet", key="search_prices",
                           disabled=not (damaged or damages) or not location.strip() or not api_key or not ai_access)
        refresh = st.button("Actualizar referencias", key="refresh_prices",
                            disabled=not references or not api_key or not ai_access,
                            help="Hace una nueva consulta y puede generar cargos.")
        if not location.strip():
            st.caption("Indica distrito y ciudad para contextualizar la búsqueda.")
        if (search or refresh) and ai_access:
            if refresh or not references:
                references = {"signature": reference_signature, "web": None, "errors": []}
                try:
                    with st.spinner("Buscando referencias..."):
                        references["web"] = search_repair_references(
                            asset, statuses, evidence, location.strip(), api_key=api_key, model=model_name, damages=damages, base_url=base_url)
                except (RuntimeError, ValueError) as ex:
                    references["errors"].append(str(ex))
                if references["web"]:
                    st.session_state[f"references_{asset}"] = references
        for error in references.get("errors", []):
            st.markdown(ui.callout("warning", error), unsafe_allow_html=True)
        web_result = references.get("web")
        if web_result:
            st.text(web_result["texto"])
            if web_result.get("candidatos"):
                st.dataframe(web_result["candidatos"], hide_index=True)
            for source in web_result["fuentes"]:
                st.link_button(source["titulo"], source["url"])
            st.caption("Consulta: " + web_result["consultado"])

    quotes = {}
    quote_context = context_fingerprint("damages-1", signature, statuses, damages, location.strip())[:16]
    if damages:
        candidates = (web_result or {}).get("candidatos", [])
        if candidates and st.button("Preparar referencias candidatas para revisión"):
            for candidate in candidates:
                if candidate["dano_id"] not in {item["id"] for item in damages}:
                    continue
                prefix = f"quote_{asset}_{quote_context}_{candidate['dano_id']}"
                # Nunca sobrescribe una referencia ya editada o aprobada.
                existing = any(st.session_state.get(prefix + suffix) for suffix in
                               ("_low", "_high", "_source", "_url", "_document", "_validity", "_scope", "_confirmed"))
                default_intervention = next(item["intervencion_sugerida"][:160] for item in damages if item["id"] == candidate["dano_id"])
                custom_intervention = st.session_state.get(prefix + "_intervention", default_intervention) != default_intervention
                if existing or custom_intervention:
                    continue
                mapping = {"intervencion": ("_intervention", 160), "fuente": ("_source", 200),
                           "url_fuente": ("_url", 2000), "documento": ("_document", 300),
                           "vigencia": ("_validity", 200), "nota": ("_scope", 300)}
                for field, (suffix, limit) in mapping.items():
                    st.session_state[prefix + suffix] = candidate[field][:limit]
                st.session_state[prefix + "_low"] = float(candidate["precio_min"])
                st.session_state[prefix + "_high"] = float(candidate["precio_max"])
                st.session_state[prefix + "_unit"] = candidate["unidad"]
                st.session_state[prefix + "_date"] = date.fromisoformat(candidate["fecha"])
                st.session_state[prefix + "_type"] = "costo_reparacion"
                st.session_state[prefix + "_confirmed"] = False
            st.info("Sugerencias cargadas sin confirmar. Revisa documento, unidad y ámbito; declara cantidad y zona. No se certifican los precios.")
        st.caption("Cada precio debe estar respaldado por un documento oficial aplicable. Indica partida, vigencia y zona. Sin respaldo oficial, la reparación queda pendiente y no se inventa un importe.")
        st.caption("Agrupa fotos o lesiones de una misma reparación en una sola partida. La cantidad corresponde a la zona afectada conjunta, no a la suma de fotos. Las zonas y trabajos declarados deben evitar solapamientos.")
        damage_labels = {item["id"]: f"{index + 1}. {item['dano']} · {item['foto']} · {item['ubicacion']}"
                         for index, item in enumerate(damages)}
        with st.form(f"quotes_{asset}_{quote_context}"):
            for index, damage in enumerate(damages):
                code = damage["id"]
                title = damage_labels[code]
                prefix = f"quote_{asset}_{quote_context}_{code}"
                with st.expander(f"Referencia oficial · {title}", expanded=len(damages) == 1):
                    included_in = st.selectbox(f"Agrupar reparación · {title}", ["", *[item["id"] for item in damages[:index]]],
                                              format_func=lambda x: damage_labels[x] if x else "Partida propia", key=prefix + "_group")
                    if included_in:
                        quotes[code] = {"incluido_en": included_in}
                        st.caption("Esta lesión se cubre con la cantidad, precio y alcance de la partida elegida; no se sumará otra vez.")
                        continue
                    st.caption(damage["intervencion_sugerida"])
                    zone = st.text_input(f"Zona física afectada · {title}", max_chars=120,
                                         placeholder="Ej.: acceso norte / sector central del terreno", key=prefix + "_zone")
                    stages = st.multiselect(f"Trabajos incluidos · {title}", list(WORK_STAGES), default=damage["trabajos_sugeridos"],
                                            format_func=lambda x: WORK_STAGES[x], key=prefix + "_stages")
                    st.session_state.setdefault(prefix + "_intervention", damage["intervencion_sugerida"][:160])
                    intervention = st.text_input(f"Intervención · {title}", max_chars=160, key=prefix + "_intervention")
                    unit_options = [damage["unidad_sugerida"], *[unit for unit in UNIT_LABELS if unit != damage["unidad_sugerida"]]]
                    unit = st.selectbox(f"Unidad · {title}", unit_options, format_func=lambda x: UNIT_LABELS[x], key=prefix + "_unit")
                    quantity = st.number_input(f"Cantidad afectada · {title}", min_value=0.0, max_value=1_000_000.0, step=0.1, format="%.3f", key=prefix + "_quantity")
                    low = st.number_input(f"Precio mínimo por unidad (S/) · {title}", min_value=0.0, max_value=1_000_000.0, step=1.0, key=prefix + "_low")
                    high = st.number_input(f"Precio máximo por unidad (S/) · {title}", min_value=0.0, max_value=1_000_000.0, step=1.0, key=prefix + "_high")
                    source = st.text_input(f"Entidad pública · {title}", max_chars=200, placeholder="MVCS, MTC, otra entidad competente...", key=prefix + "_source")
                    source_url = st.text_input(f"Enlace al documento oficial · {title}", max_chars=2000, placeholder="https://www.gob.pe/...", key=prefix + "_url")
                    document = st.text_input(f"Documento y página o partida · {title}", max_chars=300, key=prefix + "_document")
                    validity = st.text_input(f"Período y ámbito geográfico · {title}", max_chars=200, key=prefix + "_validity")
                    reference_type = st.selectbox(f"Tipo de referencia · {title}", ["sin_clasificar", "costo_reparacion", "valor_edificacion", "arancel", "indice", "otro"],
                                                  format_func=lambda x: {"sin_clasificar": "Selecciona el tipo", "costo_reparacion": "Costo unitario de reparación aplicable", "valor_edificacion": "Valor unitario de edificación", "arancel": "Valor arancelario", "indice": "Índice de precios", "otro": "Otro / no aplicable"}[x], key=prefix + "_type")
                    st.session_state.setdefault(prefix + "_date", date.today())
                    quoted_date = st.date_input(f"Fecha del documento · {title}", value=None, max_value=date.today(), key=prefix + "_date")
                    scope = st.text_input(f"Alcance y exclusiones · {title}", max_chars=300, placeholder="Materiales, mano de obra, exclusiones...", key=prefix + "_scope")
                    technical_review = st.checkbox(f"Declaro evaluación técnica previa de las lesiones de esta partida · {title}", key=prefix + "_review",
                                                   help="Necesaria para indicios posibles, severidad alta/no determinable y daños que requieren diagnóstico. No equivale a una certificación del sistema.")
                    confirmed = st.checkbox(f"Contrasté el documento oficial, su vigencia, aplicabilidad y cantidad · {title}", key=prefix + "_confirmed")
                    quotes[code] = {"intervencion": intervention, "unidad": unit, "cantidad": quantity,
                                    "precio_min": low, "precio_max": high, "fuente": source,
                                    "fecha": quoted_date.isoformat() if quoted_date else "", "nota": scope, "confirmado": confirmed,
                                    "url_fuente": source_url, "documento": document, "vigencia": validity,
                                    "tipo_referencia": reference_type, "zona_afectada": zone, "trabajos": stages,
                                    "revision_tecnica_confirmada": technical_review, "incluido_en": ""}
            st.form_submit_button("Calcular presupuesto", type="primary")
    budget = build_damage_budget(asset, statuses, area, damages, quotes,
                                 photos_complete=analysis.get("fotos_completas", False),
                                 detail_complete=analysis.get("detalle_completo", False))
    st.markdown(ui.cost_panel(budget, location), unsafe_allow_html=True)
    if budget.partial:
        st.caption("El costo es parcial si faltan tomas, criterios observables o cotizaciones válidas.")
    for limitation in budget.limitations:
        st.markdown(ui.callout("warning", limitation), unsafe_allow_html=True)
    if budget.missing:
        missing_text = "; ".join(f"{item.titulo} ({item.motivo})" for item in budget.missing)
        st.markdown(ui.callout("warning", f"Partidas pendientes: {missing_text}"), unsafe_allow_html=True)
    elif budget.damaged == 0:
        st.markdown(ui.callout("note", "No se propusieron partidas con lo observable. Esto no demuestra ausencia de daños."), unsafe_allow_html=True)
    if budget.items:
        st.dataframe([
            {"Mejora": item.intervencion, "Zona": item.zona_afectada,
             "Unidad": UNIT_LABELS[item.unidad], "Cantidad afectada": item.cantidad,
             "Precio mínimo (S/ por unidad)": item.precio_min, "Precio máximo (S/ por unidad)": item.precio_max,
             "Subtotal mínimo (S/)": item.subtotal_min, "Subtotal máximo (S/)": item.subtotal_max}
            for item in budget.items
        ], hide_index=True, width="stretch")
        st.caption("En partidas m²: costo de mejora = m² afectados × S/ por m². No se divide entre toda el área del terreno ni se suman unidades incompatibles.")
        st.markdown(ui.cost_table(budget.items), unsafe_allow_html=True)
    with st.expander("Supuestos y límites"):
        st.write("El área y la ubicación son declaradas y no se verifican.")
        st.write("Se exige procedencia oficial, pero el contenido del documento, su vigencia y aplicabilidad se contrastan según declaración del usuario; el sistema no los verifica independientemente.")
        st.write("No se estima el área afectada sin una medición confiable; no se usa el área total como área dañada.")
        st.write("No agrega IGV, fletes, permisos, imprevistos ni diagnóstico estructural; declara en el alcance cualquier concepto ya incluido en la cotización.")

if not photos:
    state_label = "Sin fotos"
elif not analysis:
    state_label = "Sin análisis"
elif analysis["fallos"] and not analysis["analizadas"]:
    state_label = "Sin resultados"
elif analysis["fallos"]:
    state_label = "Análisis parcial"
elif not analysis.get("detalle_completo", False):
    state_label = "Detalle incompleto"
elif result.coverage < 100 or not analysis.get("fotos_completas", False):
    state_label = "Cobertura parcial"
else:
    state_label = "Procesamiento terminado"
step = 4 if has_results else (2 if photos else 1)
masthead_slot.markdown(ui.masthead(spec["nombre"], num_photos, MAX_PHOTOS, state_label), unsafe_allow_html=True)
rail_slot.markdown(ui.progress_rail(step), unsafe_allow_html=True)

st.markdown(ui.section_header(4, "Informe", "Borrador automático, sin revisión humana."), unsafe_allow_html=True)
if not has_results:
    st.markdown(ui.callout("note", "El informe aparece cuando termina el análisis."), unsafe_allow_html=True)
else:
    st.markdown(ui.callout("warning", "Borrador automático, sin revisión humana."), unsafe_allow_html=True)
    data = build_report_data(
        asset, statuses, evidence, result, budget, area, location, list(photos),
        actions=actions, materials=analysis["materials"], proposal=analysis["proposal"],
        ai_metadata=analysis["metadata"],
        photo_hashes={name: hashlib.sha256(raw).hexdigest() for name, raw in photos.items()},
        web_references=web_result,
        damages=damages,
        local_detections=local_detections,
    )
    report_signature = context_fingerprint({key: value for key, value in data.items() if key != "fecha_local"})
    previous_report = st.session_state.get(f"report_data_{asset}", {})
    if previous_report.get("signature") == report_signature:
        data = previous_report["data"]
    else:
        data["sha256_contenido"] = report_digest(data)
        st.session_state[f"report_data_{asset}"] = {"signature": report_signature, "data": data}
    st.caption("El HTML incluye las fotos; compártelo con cuidado.")
    st.caption("SHA-256: " + data["sha256_contenido"] + " (no es firma digital).")
    st.download_button(
        "Descargar informe HTML (imprimible en PDF)",
        html_report(data, photos, image_preparer=cached_report_image),
        "informe_valor_a_vista.html", "text/html"
    )
    st.download_button(
        "Descargar datos JSON", json.dumps(data, ensure_ascii=False, indent=2),
        "informe_valor_a_vista.json", "application/json"
    )
