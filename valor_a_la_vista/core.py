"""Inspección visual y mejoras de terrenos; no calcula el valor de venta del suelo.

No constituye un sistema de tasación regulatoria ni de aprobación crediticia.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping

# Pesos y umbrales ilustrativos, NO calibrados con datos de una entidad financiera.
MAX_PHOTOS = 16
MAX_PHOTO_BYTES = 12 * 1024 * 1024
MAX_TOTAL_PHOTO_BYTES = 64 * 1024 * 1024
ASSETS = {
    "terreno": {
        "nombre": "Terreno · costo de mejora",
        "fotos": [
            ("Panorámica", "Vista general, sin personas."),
            ("Acceso", "Entrada o vía de acceso."),
            ("Superficie", "Pendiente y superficie."),
            ("Límites", "Cercos visibles; no prueban linderos."),
        ],
        "criterios": {
            "acceso": ("Acceso visible", 25, "Condiciones observables de la entrada"),
            "superficie": ("Superficie aparente", 25, "Obstáculos, maleza, desniveles visibles"),
            "drenaje": ("Indicios de acumulación de agua", 25, "Charcos o erosión visibles, no estudio del suelo"),
            "limites": ("Delimitación visible", 25, "Cercos o referencias observables; no verificación legal"),
        },
    },
}


# Catálogo de indicios visuales, no diagnósticos ni precios. El modelo solo elige
# códigos; unidades, intervenciones y precauciones se definen aquí.
@dataclass(frozen=True)
class DamageType:
    nombre: str
    criterios: tuple[str, ...]
    unidad: str
    intervencion: str
    trabajos: tuple[str, ...]
    revision_tecnica: bool = False


DAMAGE_CATALOG = {
    "terreno": {
        "maleza": DamageType("Maleza visible", ("superficie",), "m2", "Evaluar limpieza autorizada del área afectada.", ("limpieza",)),
        "residuos": DamageType("Residuos u obstáculos visibles", ("superficie",), "m2", "Revisar tipo de residuos antes de cotizar limpieza y disposición autorizada.", ("limpieza",)),
        "agua_acumulada": DamageType("Acumulación visible de agua", ("drenaje",), "m2", "Solicitar evaluación de causa y drenaje antes de definir obras.", ("diagnostico", "tratamiento"), True),
        "erosion_visible": DamageType("Erosión aparente", ("drenaje", "superficie"), "m2", "Solicitar evaluación de terreno; no inferir capacidad portante ni estabilidad.", ("diagnostico", "tratamiento"), True),
        "acceso_deteriorado": DamageType("Acceso visiblemente deteriorado", ("acceso",), "m2", "Inspeccionar acceso y autorizaciones antes de definir mejoramiento.", ("diagnostico", "resane")),
        "cerco_danado": DamageType("Cerco visible dañado", ("limites",), "ml", "Revisar reparación del cerco; su presencia no acredita linderos ni propiedad.", ("diagnostico", "reposicion")),
        "desnivel_visible": DamageType("Desnivel u obstáculo en superficie", ("superficie",), "m2", "Solicitar medición y evaluación antes de cotizar nivelación.", ("diagnostico", "tratamiento"), True),
    },
}
for _asset, _catalog in DAMAGE_CATALOG.items():
    _catalog["otro_visible"] = DamageType("Otro deterioro visible sin clasificar", tuple(ASSETS[_asset]["criterios"]), "servicio",
                                          "Requiere identificación y revisión profesional antes de presupuestar.", ("diagnostico",), True)

SEVERITY_LABELS = {"leve": "Leve", "moderada": "Moderada", "alta": "Alta", "no_determinable": "No determinable"}
CERTAINTY_LABELS = {"visible": "Indicio visible", "posible": "Posible; requiere comprobación"}
WORK_STAGES = {"diagnostico": "Diagnóstico", "limpieza": "Limpieza", "preparacion": "Retiro y preparación",
               "resane": "Resane o reparación", "tratamiento": "Tratamiento", "acabado": "Acabado",
               "reposicion": "Reposición"}
UNIT_LABELS = {"m2": "m²", "ml": "metro lineal", "unidad": "unidad", "servicio": "servicio"}


def validate_damage_findings(asset, items):
    """Valida detalle visual, sin aceptar cantidades o precios generados por IA."""
    if not isinstance(items, list):
        return [], ["Falta una lista válida de daños detallados; no demuestra ausencia de daños."], False
    clean, warnings, seen = [], [], set()
    if len(items) > 24:
        warnings.append("Respuesta excede 24 daños por foto; requiere revisión y otras tomas.")
    for item in items[:24]:
        if not isinstance(item, dict):
            warnings.append("Detalle de daño inválido descartado.")
            continue
        kind, criterion = item.get("tipo"), item.get("criterio")
        severity, certainty = item.get("severidad"), item.get("certeza")
        if not all(isinstance(v, str) for v in (kind, criterion, severity, certainty)):
            warnings.append("Daño con etiquetas inválidas descartado.")
            continue
        spec = DAMAGE_CATALOG[asset].get(kind)
        if not spec or criterion not in spec.criterios or severity not in SEVERITY_LABELS or certainty not in CERTAINTY_LABELS:
            warnings.append("Tipo de daño, criterio o severidad no admitidos para este activo.")
            continue
        texts = {key: item.get(key) for key in ("elemento", "ubicacion", "evidencia")}
        if any(not isinstance(v, str) or not v.strip() or v.strip().lower() == "no_visible" for v in texts.values()):
            warnings.append(f"Daño sin elemento, ubicación o evidencia: {kind}.")
            continue
        if len(texts["evidencia"].strip()) < 8 or re.search(
            r"\b(?:sin|no (?:hay|se observan?|se aprecian?))\s+(?:evidencia de\s+)?"
            r"(?:daños?|fisuras?|humedad|salitre|moho|desprendimientos?|corrosión|oxidación|maleza|residuos)",
            texts["evidencia"], re.I,
        ):
            warnings.append(f"Evidencia insuficiente o negativa para el daño {kind}; requiere revisión.")
            continue
        texts = {key: value.strip()[:320 if key == "evidencia" else 120] for key, value in texts.items()}
        identity = (kind, criterion, texts["elemento"].casefold(), texts["ubicacion"].casefold())
        if identity in seen:
            warnings.append(f"Detalle repetido en la misma foto descartado: {kind}.")
            continue
        seen.add(identity)
        clean.append({"tipo": kind, "criterio": criterion, **texts, "severidad": severity, "certeza": certainty,
                      "dano": spec.nombre, "unidad_sugerida": spec.unidad,
                      "intervencion_sugerida": spec.intervencion, "trabajos_sugeridos": list(spec.trabajos),
                      "requiere_revision": spec.revision_tecnica or certainty == "posible" or severity in {"alta", "no_determinable"}})
    return clean, warnings, not warnings


# Sugerencias orientativas, no diagnóstico ni presupuesto de un profesional.
IMPROVEMENTS = {
    "terreno": {
        "acceso": "Inspeccionar condiciones del acceso y su regularidad legal.",
        "superficie": "Evaluar limpieza, nivelación y factibilidad con un técnico.",
        "drenaje": "Solicitar evaluación de drenaje y, cuando corresponda, de suelos.",
        "limites": "Contrastar cercos visibles con planos y documentos catastrales.",
    },
}


def recommend_improvements(asset: str, statuses: Mapping[str, str]) -> list[dict[str, str]]:
    """Devuelve posibles acciones únicamente donde hay deterioro observable."""
    out = []
    for code, (title, _, _) in ASSETS[asset]["criterios"].items():
        state = statuses.get(code)
        if state in {"regular", "malo"}:
            out.append({
                "criterio": title,
                "prioridad": "Alta" if state == "malo" else "Media",
                "accion": IMPROVEMENTS[asset][code],
            })
    return out

STATUS_LABELS = {
    "no_visible": "No verificable",
    "bueno": "Bueno",
    "regular": "Regular",
    "malo": "Deteriorado",
}
# Puntuación ilustrativa por criterio.
STATUS_POINTS = {"bueno": 100.0, "regular": 60.0, "malo": 20.0}


@dataclass(frozen=True)
class Evaluation:
    score: float | None
    coverage: float
    category: str
    verified_criteria: int
    total_criteria: int


def evaluate(asset: str, statuses: Mapping[str, str]) -> Evaluation:
    """Evalúa solo criterios observados; no asigna puntaje a 'no_visible'."""
    criteria = ASSETS[asset]["criterios"]
    observed_weight = 0
    weighted_points = 0.0
    verified = 0
    for code, (_, weight, _) in criteria.items():
        state = statuses.get(code, "no_visible")
        if state in STATUS_POINTS:
            observed_weight += weight
            weighted_points += weight * STATUS_POINTS[state]
            verified += 1
    coverage = observed_weight / sum(c[1] for c in criteria.values()) * 100
    score = round(weighted_points / observed_weight, 1) if observed_weight else None
    if score is None:
        category = "Sin datos suficientes"
    elif coverage < 100:
        category = "Incompleto — requiere revisión"
    elif score >= 85:
        category = "Muy buen estado aparente"
    elif score >= 70:
        category = "Buen estado aparente"
    elif score >= 50:
        category = "Requiere mejoras aparentes"
    else:
        category = "Deterioro aparente relevante"
    return Evaluation(score, round(coverage, 1), category, verified, len(criteria))


def reference_amount(
    evaluation: Evaluation,
    photo_count: int,
    reviewed: bool,
    declared_value: float,
    requested: float,
    improvement_budget: float,
    simulated_ratio: float = 0.4,
) -> tuple[float | None, str]:
    """Ejemplo pedagógico, no límite real de una caja ni recomendación de crédito."""
    if not reviewed:
        return None, "Falta la revisión humana."
    if photo_count < 3:
        return None, "Se requieren al menos 3 fotos."
    if evaluation.score is None or evaluation.coverage < 100:
        return None, "Faltan criterios por verificar."
    if not all(math.isfinite(value) for value in (declared_value, requested, improvement_budget, simulated_ratio)):
        return None, "Importes o ratio no válidos."
    if min(declared_value, requested, improvement_budget) <= 0:
        return None, "Completa valor, crédito y presupuesto."
    if not 0 < simulated_ratio <= 1:
        return None, "Ratio no válido."
    if evaluation.score < 50:
        return None, "Deterioro aparente: requiere inspección presencial."
    amount = min(
        float(declared_value) * simulated_ratio * (evaluation.score / 100),
        float(requested),
        float(improvement_budget),
    )
    return round(amount, 2), "Solo simulación con parámetros ilustrativos; no es una oferta crediticia."


def combine_findings(
    asset: str, photo_findings: list[list[dict]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Conserva la condición visual más desfavorable, a revisar por persona."""
    ranks = {"malo": 0, "regular": 1, "bueno": 2}
    statuses = {key: "no_visible" for key in ASSETS[asset]["criterios"]}
    evidence = {key: "" for key in ASSETS[asset]["criterios"]}
    for findings in photo_findings:
        for item in findings:
            key = item.get("criterio")
            state = item.get("estado")
            if key not in statuses or state not in ranks:
                continue
            previous = statuses[key]
            if previous == "no_visible" or ranks[state] < ranks[previous]:
                statuses[key] = state
                evidence[key] = str(item.get("evidencia", ""))[:320]
    return statuses, evidence
