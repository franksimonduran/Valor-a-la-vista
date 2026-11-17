"""Exportación local del borrador automático en HTML y JSON."""
from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime
from html import escape

from .core import ASSETS, STATUS_LABELS, SEVERITY_LABELS, CERTAINTY_LABELS, UNIT_LABELS, WORK_STAGES, Evaluation
from .pricing import RepairBudget, budget_to_dict
from .vision import prepare_image
from .ui import ACCENT, BACKGROUND, BORDER, INK, MUTED

DRAFT_LABEL = "Borrador automático, sin revisión humana"


def report_digest(data: dict) -> str:
    """SHA-256 de JSON canónico sin el campo de integridad; no es una firma."""
    payload = {k: v for k, v in data.items() if k != "sha256_contenido"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_report_data(asset, statuses, evidence, evaluation: Evaluation, budget: RepairBudget,
                      area, location, photo_names, actions=None, materials=None,
                      proposal=None, ai_metadata=None, photo_hashes=None, web_references=None,
                       published_rates=None, damages=None, local_detections=None):
    """Datos del informe. No hay revisor ni confirmación humana por diseño."""
    return {
        "sistema": "Valor A la Vista · Borrador automático",
        "etiqueta": DRAFT_LABEL,
        "fecha_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tipo_activo": ASSETS[asset]["nombre"],
        "area_declarada_m2": area,
        "area_verificada": False,
        "ubicacion_declarada": location,
        "ubicacion_verificada": False,
        "fotografias": photo_names,
        "sha256_fotografias_originales": photo_hashes or {},
        "version_reglas": "8-terrenos-video",
        "objetivo": "Costo aproximado de mejora del terreno, no valor de venta ni garantía",
        "politica_fuentes": "Fuentes oficiales peruanas; procedencia de URL comprobada, contenido y aplicabilidad no verificados independientemente",
        "confianza_modelo": None,
        "score_condicion": evaluation.score,
        "cobertura_criterios_pct": evaluation.coverage,
        "clasificacion": evaluation.category,
        "revision_humana": False,
        "analisis_ia": ai_metadata,
        "detecciones_yolo": local_detections or None,
        "propuesta_ia": proposal,
        "danos_detectados": damages or [],
        "hallazgos": [
            {"codigo": code, "criterio": title, "estado": statuses.get(code, "no_visible"),
             "evidencia": evidence.get(code, "")}
            for code, (title, _, _) in ASSETS[asset]["criterios"].items()
        ],
        "mejoras_sugeridas": actions or [],
        "materiales_aparentes": materials or [],
        "costo_reparacion": budget_to_dict(budget),
        "referencias_web": web_references,
        "tarifas_publicadas": published_rates or [],
        "supuestos": [
            "El área y la ubicación son declaradas por el usuario y no se verifican.",
            "No se estima el área afectada sin una medición confiable.",
            "Las cantidades y cotizaciones se contrastan según declaración del usuario; el sistema no las verifica independientemente.",
            "Se exige documento con URL oficial, referencia de página o partida y vigencia; los valores de edificación, aranceles e índices no se convierten en costos de reparación.",
            "Las lesiones son propuestas visuales; no confirman causas ocultas, funcionamiento ni seguridad.",
            "Las fotos no proporcionan medidas reales sin una escala confiable; cada cantidad es declarada para la zona conjunta.",
            "Una agrupación de lesiones se presupuesta una sola vez; se bloquean fases repetidas en una misma zona declarada. Identificar zonas distintas incorrectamente puede impedir detectar solapamientos.",
            "La evaluación técnica previa es una declaración del usuario para la partida, no una certificación ni revisión final del informe.",
            "Las referencias web son atribuciones de OpenAI; precios y aplicabilidad no se verifican independientemente ni se suman al total sin contraste declarado.",
            "No agrega IGV, fletes, permisos, imprevistos ni diagnóstico estructural; el alcance debe declarar conceptos ya incluidos en la cotización.",
            "La ausencia de una partida no demuestra que no exista un daño.",
            "El costo es parcial si faltan tomas, criterios observables o precios.",
            "Los videos se muestrean en fotogramas: no se analiza audio ni se garantiza detectar lo que ocurre entre capturas. Los tiempos son aproximados.",
            "Una región YOLO es una propuesta no verificada, no una medida de superficie ni una reparación adicional. No modifica automáticamente el score ni el presupuesto.",
        ],
        "advertencia": "Costo referencial de reparación; no es el valor del activo.",
        "limitacion": "No es una tasación, diagnóstico técnico ni decisión de crédito.",
    }


def _money(value) -> str:
    return "Sin precio" if value is None else f"S/ {value:,.2f}"


def html_report(data: dict, images: dict[str, bytes], image_preparer=prepare_image) -> str:
    e = lambda value: escape(str(value), quote=True)
    cost = data.get("costo_reparacion") or {}
    partidas = cost.get("partidas") or []
    missing = cost.get("sin_precio_referencial") or []
    hallazgos = data.get("hallazgos") or []
    score = data.get("score_condicion")
    score_text = "No calculado" if score is None else f"{score}/100"
    coverage = data.get("cobertura_criterios_pct")
    coverage_text = "No calculada" if coverage is None else f"{coverage}%"
    total_min = cost.get("total_min_soles")
    total_max = cost.get("total_max_soles")
    total_central = cost.get("total_central_soles")
    if total_min is None:
        total_text = "Sin precio referencial"
    elif total_min == total_max:
        total_text = _money(total_min)
    else:
        total_text = f"{_money(total_min)} – {_money(total_max)}"
    central_text = "No calculado" if total_central is None else _money(total_central)
    area = data.get("area_declarada_m2")
    area_text = f"{area:g} m²" if area else "No indicada"
    location = data.get("ubicacion_declarada") or "No indicada"

    improvements = "".join(
        f"<li><strong>{e(action.get('prioridad', ''))} · {e(action.get('criterio', ''))}</strong>: "
        f"{e(action.get('accion', ''))}</li>"
        for action in data.get("mejoras_sugeridas", [])
    )
    materials_html = "".join(
        f"<li>{e(m.get('foto', 'Foto'))}: {e(m.get('elemento', ''))} — "
        f"{e(m.get('material_aparente', ''))} ({e(m.get('evidencia', ''))})</li>"
        for m in data.get("materiales_aparentes", [])
    )
    rows = "".join(
        f"<tr><td>{e(h.get('criterio', ''))}</td>"
        f"<td>{e(STATUS_LABELS.get(h.get('estado'), h.get('estado', '')))}</td>"
        f"<td>{e(h.get('evidencia') or 'Sin descripción')}</td></tr>"
        for h in hallazgos
    )
    damage_rows = "".join(
        f"<tr><td>{e(item.get('id', ''))}<br>{e(item.get('foto', ''))}"
        f"<br>{e(item.get('origen', {}).get('tiempo_aproximado', ''))}</td>"
        f"<td>{e(item.get('elemento', ''))}<br>{e(item.get('ubicacion', ''))}</td>"
        f"<td>{e(item.get('dano', item.get('tipo', '')))}</td>"
        f"<td>{e(SEVERITY_LABELS.get(item.get('severidad'), 'No determinable'))}<br>"
        f"{e(CERTAINTY_LABELS.get(item.get('certeza'), 'Requiere comprobación'))}</td>"
        f"<td>{e(item.get('evidencia', ''))}<br>{e(item.get('intervencion_sugerida', ''))}"
        f"<br><small>{'Evaluación técnica previa necesaria.' if item.get('requiere_revision') else 'Contrastar alcance.'}</small></td></tr>"
        for item in data.get("danos_detectados", [])
    )
    partida_rows_list = []
    for item in partidas:
        note = f"<br><small>{e(item.get('nota'))}</small>" if item.get("nota") else ""
        if item.get("validacion"):
            note += f"<br><small>{e(item['validacion'])}</small>"
        if item.get("danos_incluidos"):
            note += f"<br><small>Daños incluidos: {e(', '.join(item['danos_incluidos']))}. Zona: {e(item.get('zona_afectada', ''))}.</small>"
            note += "<br><small>Trabajos: " + e(", ".join(WORK_STAGES.get(stage, stage) for stage in item.get("trabajos", []))) + ".</small>"
            if item.get("revision_tecnica_declarada"):
                note += "<br><small>Evaluación técnica previa declarada por el usuario; no verificada por el sistema.</small>"
        source_detail = "<br>".join(e(item.get(key, "")) for key in ("fuente", "fecha", "documento", "vigencia", "url_fuente") if item.get(key))
        unit = UNIT_LABELS.get(item.get("unidad"), item.get("unidad", ""))
        partida_rows_list.append(
            f"<tr><td>{e(item.get('intervencion', ''))}{note}</td>"
            f"<td>{e(item.get('cantidad', ''))} {e(unit)}</td>"
            f"<td>{e(_money(item.get('precio_min')))} – {e(_money(item.get('precio_max')))} por {e(unit)}</td>"
            f"<td>{e(_money(item.get('subtotal_min')))} – {e(_money(item.get('subtotal_max')))}</td>"
            f"<td>{source_detail}</td></tr>")
    partida_rows = "".join(partida_rows_list)
    missing_html = "".join(
        f"<li><strong>{e(item.get('titulo', item.get('criterio', '')))}</strong>: {e(item.get('motivo', ''))}</li>"
        for item in missing
    )
    assumptions = "".join(f"<li>{e(item)}</li>" for item in data.get("supuestos", []))
    web = data.get("referencias_web") or {}
    web_sources = "".join(
        f"<li>{e(item.get('titulo', 'Fuente'))}: {e(item.get('url', ''))}<br><small>{e(item.get('extracto', ''))}</small></li>"
        for item in web.get("fuentes", [])
    )
    web_html = (f"<p>{e(web.get('texto', ''))}</p><p>Consulta: {e(web.get('consultado', ''))}</p><ul>{web_sources}</ul>"
                if web_sources else "<p>Sin referencias web disponibles.</p>")
    rates_html = "".join(
        f"<li>{e(item.get('intervencion', ''))}: S/ {e(item.get('precio_min', ''))} – "
        f"S/ {e(item.get('precio_max', ''))} por {e(item.get('unidad', ''))}. "
        f"{e(item.get('alcance', ''))} Fuente: {e(item.get('url', ''))}; "
        f"consulta: {e(item.get('consultado', ''))}; fecha de publicación no indicada.</li>"
        for item in data.get("tarifas_publicadas", [])
    )
    pics = []
    for name, raw in images.items():
        b64 = base64.b64encode(image_preparer(raw)).decode("ascii")
        pics.append(f'<figure><img alt="{e(name)}" src="data:image/jpeg;base64,{b64}"><figcaption>{e(name)}</figcaption></figure>')
    ai = data.get("analisis_ia")
    if ai:
        ai_html = (f"Proveedor: {e(ai.get('proveedor', 'No indicado'))}; "
                   f"modelo: {e(ai.get('modelo', 'No indicado'))}; "
                   f"prompt: {e(ai.get('version_prompt', 'No indicada'))}; "
                   f"esquema: {e(ai.get('version_esquema', 'No indicada'))}; "
                    f"fotos analizadas: {e(', '.join(ai.get('fotos_analizadas', [])) or 'Ninguna')}. "
                    f"Procesamiento: {e(ai.get('fecha_procesamiento', 'No indicada'))}.")
        for label, key in (("Fallos", "fallos"), ("Advertencias", "advertencias")):
            if ai.get(key):
                ai_html += f"<br><strong>{label}:</strong> {e('; '.join(ai[key]))}"
    else:
        ai_html = "No se ejecutó análisis de IA"
    complete = bool(cost.get("completo"))
    total_label = "Total referencial" if complete else "Total parcial"
    draft = e(data.get("etiqueta") or DRAFT_LABEL)
    limitations = "".join(f"<li>{e(item)}</li>" for item in cost.get("limitaciones", []))
    photo_evidence = "".join(
        f"<li><strong>{e(item.get('foto', ''))}</strong> · {e(item.get('fecha_analisis', ''))}"
        f" · Video: {e(item.get('origen', {}).get('video_nombre', 'No aplica'))}"
        f" · Tiempo aproximado: {e(item.get('origen', {}).get('tiempo_aproximado', 'No aplica'))}<ul>"
        + "".join(f"<li>{e(h.get('criterio', ''))}: {e(STATUS_LABELS.get(h.get('estado'), 'No verificable'))} — {e(h.get('evidencia', ''))}</li>"
                  for h in item.get("hallazgos", [])) + "</ul></li>"
        for item in (ai or {}).get("evidencias_por_foto", [])
    )
    yolo = data.get("detecciones_yolo") or {}
    yolo_rows = "".join(f"<li>{e(item.get('foto', ''))}: {e(region.get('tipo', ''))} · "
                        f"confianza del detector {e(region.get('confianza_detector', ''))} · "
                        f"caja normalizada {e(region.get('caja_normalizada_xyxy', []))}</li>"
                        for item in yolo.get("resultados", []) for region in item.get("regiones", []))
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Informe Valor A la Vista</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px "Inter","Segoe UI",sans-serif;color:{INK};max-width:1000px;margin:40px auto;padding:0 20px;line-height:1.55;-webkit-print-color-adjust:exact;print-color-adjust:exact}}
.cover{{border-bottom:1px solid {BORDER};margin-bottom:38px;padding-bottom:34px}}
.brand-mono{{font-family:Georgia,"Times New Roman",serif;font-size:60px;font-weight:600;line-height:1;letter-spacing:-.03em;color:{INK}}}
.brand-acc{{color:{ACCENT}}}
.brand-name{{font-family:"Sora","Segoe UI",sans-serif;font-size:19px;font-weight:600;letter-spacing:.16em;color:{INK};margin:10px 0 0}}
.brand-eyebrow{{font-size:11px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:{MUTED};margin:10px 0 0}}
.draft{{border:1px solid #8A5E12;border-radius:999px;color:#8A5E12;display:inline-block;font-size:10px;font-weight:600;letter-spacing:.12em;margin:26px 0 0;padding:6px 10px;text-transform:uppercase}}
.cover-asset{{font-family:"Sora","Segoe UI",sans-serif;font-size:34px;font-weight:600;letter-spacing:-.02em;margin:18px 0 0}}
.cover-sub{{color:{MUTED};margin:6px 0 0}}
.cover-ledger{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0 26px;margin:30px 0 0}}
.cover-ledger div{{border-top:1px solid {BORDER};padding-top:9px}}
.cover-ledger dt{{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:{MUTED}}}
.cover-ledger dd{{margin:5px 0 0;font-variant-numeric:tabular-nums;font-weight:600}}
.cover-note{{border-left:2px solid {ACCENT};font-size:13px;margin:26px 0 0;padding:2px 0 2px 14px}}
.cost{{background:{BACKGROUND};border:1px solid {BORDER};border-radius:12px;margin:18px 0;padding:18px 20px}}
.cost-label{{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:{MUTED};margin:0}}
.cost-value{{font-family:"Sora","Segoe UI",sans-serif;font-size:30px;font-weight:600;letter-spacing:-.02em;margin:6px 0 0;font-variant-numeric:tabular-nums}}
.cost-mid{{color:{MUTED};margin:4px 0 0}}
h2{{font-family:"Sora","Segoe UI",sans-serif;font-size:20px;letter-spacing:-.01em;border-top:1px solid {BORDER};padding-top:18px;margin:38px 0 10px}}
.note{{border-left:2px solid {ACCENT};background:{BACKGROUND};padding:12px 14px}}
.gallery{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}
img{{width:100%;height:220px;object-fit:contain;background:{BACKGROUND}}}
figure{{margin:0}}figcaption{{font-size:12px;color:{MUTED};padding:8px 0}}
table{{border-collapse:collapse;width:100%;margin:18px 0;font-variant-numeric:tabular-nums}}
th,td{{border-bottom:1px solid {BORDER};text-align:left;padding:10px;vertical-align:top}}
th{{background:{BACKGROUND};font-size:11px;letter-spacing:.08em;text-transform:uppercase}}
small{{color:{MUTED}}}
@media print{{
  @page{{size:A4;margin:16mm}}
  body{{margin:0;padding:0;max-width:none;font-size:11pt}}
  .cover{{border-bottom:0;break-after:page;margin-bottom:0;min-height:245mm;padding-top:6mm}}
  .brand-mono{{font-size:54px}}
  h2{{break-after:avoid}}
  figure,tr,.cost{{break-inside:avoid}}
  thead{{display:table-header-group}}
  .gallery{{display:grid;grid-template-columns:1fr 1fr}}
  .brand-acc{{color:{INK}}}
}}
</style></head><body>
<section class="cover">
<header class="brand">
<div class="brand-mono" aria-hidden="true">V<span class="brand-acc">A</span>V</div>
<h1 class="brand-name">Valor A la Vista</h1>
<p class="brand-eyebrow">Informe automático de inspección visual</p>
</header>
<p class="draft">{draft}</p>
<p class="cover-asset">Costo aproximado de mejora del terreno</p>
<p class="cover-sub">{e(data.get('tipo_activo', 'Activo'))}</p>
<dl class="cover-ledger">
<div><dt>Fecha</dt><dd>{e(data.get('fecha_local', ''))}</dd></div>
<div><dt>Área declarada</dt><dd>{e(area_text)}</dd></div>
<div><dt>Ubicación</dt><dd>{e(location)}</dd></div>
<div><dt>Partidas con precio</dt><dd>{len(partidas)}</dd></div>
<div><dt>{e(total_label)}</dt><dd>{e(total_text)}</dd></div>
<div><dt>Condición aparente</dt><dd>{e(score_text)}</dd></div>
</dl>
<p class="cover-note"><strong>{draft}.</strong> {e(data.get('limitacion', ''))} {e(data.get('advertencia', ''))}</p>
</section>
<h2>Resumen</h2>
<div class="cost">
<p class="cost-label">Costo aproximado de mejora · {'rango completo' if complete else 'total parcial'}</p>
<p class="cost-value">{e(total_text)}</p>
<p class="cost-mid">Valor central: {e(central_text)}. Cobertura de criterios: {e(coverage_text)}.</p>
</div>
<p><strong>Resultado visual:</strong> {e(data.get('clasificacion', 'Sin datos'))}. El costo no es el valor del activo.</p>
<p><strong>IA:</strong> {ai_html}<br>
<strong>SHA-256 del contenido JSON:</strong> {e(data.get('sha256_contenido', 'No calculado'))} (no es firma digital).</p>
<h2>Partidas y precios de referencia</h2>
<p>Partidas m²: costo = superficie afectada declarada × precio por m². No es el valor de venta del terreno. Otras unidades se mantienen cuando corresponda.</p>
<table><thead><tr><th>Intervención</th><th>Cantidad</th><th>Precio unitario</th><th>Subtotal</th><th>Fuente y fecha</th></tr></thead>
<tbody>{partida_rows or '<tr><td colspan="5">Sin partidas con precio referencial.</td></tr>'}</tbody></table>
<ul>{missing_html or ("<li>Sin partidas propuestas; no demuestra ausencia de daños.</li>" if not partidas else "<li>Las partidas incluidas tienen cantidad y precio registrados.</li>")}{limitations}</ul>
<h2>Supuestos</h2><ul>{assumptions}</ul>
<h2>Tarifas unitarias publicadas</h2><ul>{rates_html or '<li>Sin tarifas unitarias disponibles.</li>'}</ul>
<h2>Referencias web no verificadas independientemente</h2>{web_html}
<h2>Hallazgos observados</h2><table><thead><tr><th>Criterio</th><th>Estado</th><th>Evidencia</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Lesiones específicas propuestas</h2><table><thead><tr><th>ID y foto</th><th>Elemento y ubicación</th><th>Daño aparente</th><th>Severidad y certeza</th><th>Evidencia y acción orientativa</th></tr></thead><tbody>{damage_rows or '<tr><td colspan="5">Sin detalle de lesiones; no demuestra ausencia de daños.</td></tr>'}</tbody></table>
<h2>Trazabilidad por fotografía</h2><ul>{photo_evidence or '<li>Sin detalle por foto.</li>'}</ul>
<h2>YOLO local (opcional)</h2><p>Regiones propuestas, no verificadas ni incluidas automáticamente en el presupuesto. No detectar regiones no demuestra ausencia de daños.</p><p>SHA-256 de pesos: {e(yolo.get('sha256_pesos', 'No ejecutado'))}. Umbral: {e(yolo.get('umbral', 'No aplica'))}.</p><ul>{yolo_rows or '<li>Sin regiones YOLO registradas.</li>'}</ul>
<h2>Posibles mejoras orientativas</h2><ul>{improvements or "<li>Sin mejoras observables propuestas.</li>"}</ul>
<h2>Materiales aparentes observados por IA</h2><ul>{materials_html or "<li>Sin elementos identificados o no se ejecutó IA.</li>"}</ul>
<h2>Fotografías de respaldo</h2><div class="gallery">{''.join(pics) or '<p>Sin fotografías.</p>'}</div>
<p><small>Los hashes corresponden a los archivos originales y no prueban su origen. El HTML contiene copias redimensionadas.</small></p>
</body></html>"""
