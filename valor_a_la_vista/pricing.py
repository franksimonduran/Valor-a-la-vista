"""Presupuestos por lesión y referencias oficiales; conserva la tabla local legada.

Ni los precios ni las cantidades se inventan con IA. Las agrupaciones declaradas
evitan presupuestar por foto y se comprueban solapamientos de zona y trabajo.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import urlsplit

from .core import ASSETS, MAX_PHOTOS, WORK_STAGES, UNIT_LABELS, validate_damage_findings

DEFAULT_TABLE_PATH = Path(__file__).resolve().parent.parent / "data" / "precios_referenciales.csv"
PRICE_COLUMNS = (
    "activo", "criterio", "intervencion", "unidad", "factor",
    "precio_min", "precio_max", "moneda", "fuente", "fecha", "nota",
)
DAMAGED = {"regular", "malo"}


def is_official_source_url(value) -> bool:
    """Comprueba procedencia de la URL, no el contenido ni vigencia del documento."""
    if not isinstance(value, str) or len(value) > 2000 or "\\" in value or any(c.isspace() for c in value):
        return False
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower()
        if len(host) > 253 or not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in host.split(".")):
            return False
        return (parsed.scheme == "https" and (host == "gob.pe" or host.endswith(".gob.pe"))
                and not parsed.username and not parsed.password and parsed.port in (None, 443))
    except ValueError:
        return False


@dataclass(frozen=True)
class PriceRow:
    activo: str
    criterio: str
    intervencion: str
    unidad: str
    factor: float
    precio_min: float
    precio_max: float
    moneda: str
    fuente: str
    fecha: str
    nota: str = ""


@dataclass(frozen=True)
class BudgetItem:
    criterio: str
    titulo: str
    intervencion: str
    unidad: str
    cantidad: float
    precio_min: float
    precio_max: float
    subtotal_min: float
    subtotal_max: float
    fuente: str
    fecha: str
    nota: str = ""
    validacion: str = "Referencia local sin verificación independiente"
    url_fuente: str = ""
    documento: str = ""
    vigencia: str = ""
    tipo_referencia: str = ""
    id_partida: str = ""
    danos_incluidos: tuple[str, ...] = ()
    zona_afectada: str = ""
    trabajos: tuple[str, ...] = ()
    revision_tecnica_declarada: bool = False


@dataclass(frozen=True)
class MissingItem:
    criterio: str
    titulo: str
    motivo: str


@dataclass(frozen=True)
class RepairBudget:
    items: tuple[BudgetItem, ...] = field(default_factory=tuple)
    missing: tuple[MissingItem, ...] = field(default_factory=tuple)
    total_min: float | None = None
    total_max: float | None = None
    total_central: float | None = None
    complete: bool = False
    area: float | None = None
    damaged: int = 0
    unobserved: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def partial(self) -> bool:
        return not self.complete


def _clean(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _row_error(raw: Mapping[str, str], allowed_units=None) -> str | None:
    activo = _clean(raw.get("activo"))
    criterio = _clean(raw.get("criterio"))
    if activo not in ASSETS:
        return "activo desconocido"
    if criterio not in ASSETS[activo]["criterios"]:
        return "criterio no pertenece al activo"
    if not _clean(raw.get("intervencion")):
        return "falta la intervención"
    if len(_clean(raw.get("intervencion"))) > 160:
        return "intervención demasiado larga"
    unidad = _clean(raw.get("unidad"))
    if unidad not in (allowed_units or {"m2", "servicio"}):
        return "unidad no admitida"
    factor = _number(raw.get("factor"))
    if factor is None or factor < 0:
        return "factor inválido"
    if unidad == "servicio" and factor != 0:
        return "un servicio debe tener factor 0"
    if unidad == "m2" and factor <= 0:
        return "una partida por m² necesita factor mayor que 0"
    price_min = _number(raw.get("precio_min"))
    price_max = _number(raw.get("precio_max"))
    if price_min is None or price_max is None or price_min <= 0 or price_max <= 0:
        return "precio ausente o inválido"
    if price_max < price_min:
        return "precio máximo menor que el mínimo"
    if _clean(raw.get("moneda")) != "PEN":
        return "moneda distinta de PEN"
    if not _clean(raw.get("fuente")):
        return "falta la fuente"
    if len(_clean(raw.get("fuente"))) > 200:
        return "fuente demasiado larga"
    try:
        date.fromisoformat(_clean(raw.get("fecha")))
    except ValueError:
        return "fecha inválida"
    if len(_clean(raw.get("nota"))) > 300:
        return "nota demasiado larga"
    return None


def load_price_table(path: str | Path | None = None) -> tuple[list[PriceRow], list[str]]:
    """Carga y valida la tabla. Devuelve las filas usables y las advertencias."""
    source = Path(path) if path else DEFAULT_TABLE_PATH
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(name.strip() for name in (reader.fieldnames or ()))
        if not all(column in header for column in PRICE_COLUMNS):
            raise ValueError("La tabla de precios no tiene todas las columnas requeridas.")
        rows: list[PriceRow] = []
        warnings: list[str] = []
        seen: set[tuple[str, str]] = set()
        for line, raw in enumerate(reader, start=2):
            error = _row_error(raw)
            if error:
                warnings.append(f"Fila {line} descartada: {error}.")
                continue
            key = (_clean(raw["activo"]), _clean(raw["criterio"]))
            if key in seen:
                warnings.append(f"Fila {line} descartada: partida duplicada para {key[1]}.")
                continue
            seen.add(key)
            rows.append(PriceRow(
                activo=key[0],
                criterio=key[1],
                intervencion=_clean(raw["intervencion"]),
                unidad=_clean(raw["unidad"]),
                factor=float(_clean(raw["factor"])),
                precio_min=float(_clean(raw["precio_min"])),
                precio_max=float(_clean(raw["precio_max"])),
                moneda=_clean(raw["moneda"]),
                fuente=_clean(raw["fuente"]),
                fecha=_clean(raw["fecha"]),
                nota=_clean(raw.get("nota")),
            ))
    return rows, warnings


def build_repair_budget(asset: str, statuses: Mapping[str, str], area,
                        rows: Iterable[PriceRow], *, missing_reason="Sin precio referencial en la tabla.") -> RepairBudget:
    """Cruza el deterioro observado con la tabla y calcula rangos por partida."""
    if asset not in ASSETS:
        raise ValueError("Activo desconocido.")
    declared_area = _number(area)
    if declared_area is not None and declared_area <= 0:
        declared_area = None
    by_criterion = {row.criterio: row for row in rows if row.activo == asset}
    items: list[BudgetItem] = []
    missing: list[MissingItem] = []
    criteria = ASSETS[asset]["criterios"]
    damaged = 0
    for code, (title, _, _) in criteria.items():
        if statuses.get(code) not in DAMAGED:
            continue
        damaged += 1
        row = by_criterion.get(code)
        if row is None:
            missing.append(MissingItem(code, title, missing_reason))
            continue
        if row.unidad == "m2":
            if declared_area is None:
                missing.append(MissingItem(code, title, "Falta el área declarada para una partida por m²."))
                continue
            quantity = round(declared_area * row.factor, 1)
        else:
            quantity = 1.0
        if quantity <= 0:
            missing.append(MissingItem(code, title, "La cantidad calculada no es positiva."))
            continue
        items.append(BudgetItem(
            criterio=code,
            titulo=title,
            intervencion=row.intervencion,
            unidad=row.unidad,
            cantidad=quantity,
            precio_min=row.precio_min,
            precio_max=row.precio_max,
            subtotal_min=round(quantity * row.precio_min, 2),
            subtotal_max=round(quantity * row.precio_max, 2),
            fuente=row.fuente,
            fecha=row.fecha,
            nota=row.nota,
        ))
    if items:
        total_min = round(sum(item.subtotal_min for item in items), 2)
        total_max = round(sum(item.subtotal_max for item in items), 2)
        total_central = round((total_min + total_max) / 2, 2)
    elif damaged:
        total_min = total_max = total_central = None
    else:
        total_min = total_max = total_central = 0.0
    return RepairBudget(
        items=tuple(items),
        missing=tuple(missing),
        total_min=total_min,
        total_max=total_max,
        total_central=total_central,
        complete=not missing,
        area=declared_area,
        damaged=damaged,
    )


def budget_to_dict(budget: RepairBudget) -> dict:
    """Representación serializable para el informe JSON."""
    return {
        "moneda": "PEN",
        "area_declarada_m2": budget.area,
        "partidas": [asdict(item) for item in budget.items],
        "sin_precio_referencial": [asdict(item) for item in budget.missing],
        "total_min_soles": budget.total_min,
        "total_max_soles": budget.total_max,
        "total_central_soles": budget.total_central,
        "completo": budget.complete,
        "criterios_no_verificables": list(budget.unobserved),
        "limitaciones": list(budget.limitations),
    }


def build_declared_budget(asset: str, statuses: Mapping[str, str], area, quotes: Mapping,
                          *, photos_complete: bool = False, allowed_units=None) -> RepairBudget:
    """Calcula con documentos de procedencia oficial y cantidades declaradas.

    Nunca usa el área total del activo ni los factores ilustrativos de la tabla.
    El usuario contrasta contenido, vigencia y aplicabilidad; la URL por sí sola
    no demuestra que el documento respalde ese importe ni que sea una tarifa oficial.
    """
    if asset not in ASSETS:
        raise ValueError("Activo desconocido.")
    items, missing = [], []
    unobserved = []
    damaged = 0
    cent = Decimal("0.01")
    for code, (title, _, _) in ASSETS[asset]["criterios"].items():
        state = statuses.get(code, "no_visible")
        if state not in {"bueno", "regular", "malo"}:
            unobserved.append(title)
        if state not in DAMAGED:
            continue
        damaged += 1
        quote = quotes.get(code) or {}
        reason = None
        if quote.get("confirmado") is not True:
            reason = "Falta contrastar y confirmar la referencia oficial de esta partida."
        elif not is_official_source_url(quote.get("url_fuente")):
            reason = "Falta un enlace HTTPS directo a una fuente oficial peruana (.gob.pe)."
        elif quote.get("tipo_referencia") != "costo_reparacion":
            reason = "Un valor de edificación, arancel, índice o presupuesto global no equivale a costo de reparación."
        elif not _clean(quote.get("documento")) or len(_clean(quote.get("documento"))) > 300:
            reason = "Identifica el documento oficial y la página, tabla o partida que respalda el precio."
        elif not _clean(quote.get("vigencia")) or len(_clean(quote.get("vigencia"))) > 200:
            reason = "Falta indicar el período y ámbito geográfico de aplicación."
        else:
            raw = {"activo": asset, "criterio": code, "intervencion": quote.get("intervencion"),
                   "unidad": quote.get("unidad"), "factor": 1 if quote.get("unidad") == "m2" else 0,
                   "precio_min": quote.get("precio_min"), "precio_max": quote.get("precio_max"),
                   "moneda": "PEN", "fuente": quote.get("fuente"), "fecha": quote.get("fecha"),
                   "nota": quote.get("nota")}
            reason = _row_error(raw, allowed_units)
            if not reason and not _clean(quote.get("nota")):
                reason = "Falta indicar el alcance: materiales, mano de obra y exclusiones."
        if not reason:
            try:
                quantity = Decimal(str(quote.get("cantidad")))
                low = Decimal(str(quote.get("precio_min")))
                high = Decimal(str(quote.get("precio_max")))
                if not all(v.is_finite() and 0 < v <= 1_000_000 for v in (quantity, low, high)):
                    raise ValueError("Cantidad o precio fuera de rango (mayor que 0 y hasta 1 000 000).")
                if date.fromisoformat(_clean(quote.get("fecha"))) > date.today():
                    raise ValueError("La cotización no puede tener fecha futura.")
                if quantity.as_tuple().exponent < -3 or low.as_tuple().exponent < -2 or high.as_tuple().exponent < -2:
                    raise ValueError("Usa hasta 3 decimales en cantidad y 2 en precios.")
            except (InvalidOperation, ValueError, TypeError) as exc:
                reason = str(exc) or "Cantidad inválida."
        if reason:
            missing.append(MissingItem(code, title, reason))
            continue
        items.append(BudgetItem(
            criterio=code, titulo=title, intervencion=_clean(quote["intervencion"]),
            unidad=quote["unidad"], cantidad=float(quantity), precio_min=float(low), precio_max=float(high),
            subtotal_min=float((quantity * low).quantize(cent, rounding=ROUND_HALF_UP)),
            subtotal_max=float((quantity * high).quantize(cent, rounding=ROUND_HALF_UP)),
            fuente=_clean(quote["fuente"]), fecha=_clean(quote["fecha"]), nota=_clean(quote["nota"]),
            validacion="Procedencia de URL oficial; contenido, vigencia y cantidad contrastados según declaración del usuario, no verificados por el sistema",
            url_fuente=quote["url_fuente"], documento=_clean(quote["documento"]),
            vigencia=_clean(quote["vigencia"]), tipo_referencia="costo_reparacion",
        ))
    limitations = []
    if unobserved:
        limitations.append("Hay criterios no verificables; no se descartan daños fuera de lo observado.")
    if not photos_complete:
        limitations.append("Faltan tomas guiadas o alguna foto no pudo procesarse.")
    if items:
        low_total = sum((Decimal(str(item.subtotal_min)) for item in items), Decimal(0))
        high_total = sum((Decimal(str(item.subtotal_max)) for item in items), Decimal(0))
        totals = (float(low_total), float(high_total), float(((low_total + high_total) / 2).quantize(cent, rounding=ROUND_HALF_UP)))
    elif damaged or unobserved or not photos_complete:
        totals = (None, None, None)
    else:
        totals = (0.0, 0.0, 0.0)
    declared_area = _number(area)
    return RepairBudget(items=tuple(items), missing=tuple(missing), total_min=totals[0],
                        total_max=totals[1], total_central=totals[2],
                        complete=not missing and not limitations, area=declared_area if declared_area and declared_area > 0 else None,
                        damaged=damaged, unobserved=tuple(unobserved), limitations=tuple(limitations))


def build_damage_budget(asset, statuses, area, damages, quotes, *, photos_complete=False, detail_complete=False):
    """Una partida por grupo declarado de lesiones, nunca por número de fotos.

    Agrupar no suma áreas. Se exige una cantidad para la unión de la zona afectada.
    Dos partidas con igual zona y una fase compartida se bloquean para no duplicar
    preparación/acabados. Zonas mal identificadas no pueden detectarse por código.
    """
    if asset not in ASSETS:
        raise ValueError("Activo desconocido.")
    if not isinstance(damages, list) or len(damages) > 24 * MAX_PHOTOS or not isinstance(quotes, Mapping):
        raise ValueError("Lista de daños o referencias inválida.")
    known, observations = {}, set()
    for damage in damages:
        clean, _, valid = validate_damage_findings(asset, [damage])
        ident = damage.get("id") if isinstance(damage, dict) else None
        if not valid or not isinstance(ident, str) or not ident or len(ident) > 80 or ident in known:
            raise ValueError("Detalle de daños inválido o identificador repetido.")
        observation = (damage.get("foto", ""), damage.get("sha256", ""), clean[0]["tipo"], clean[0]["criterio"],
                       clean[0]["elemento"].casefold(), clean[0]["ubicacion"].casefold())
        if observation in observations:
            raise ValueError("Observación repetida en la misma foto.")
        observations.add(observation)
        known[ident] = {**clean[0], "id": ident}
    groups, errors = {}, {}
    for ident in known:
        current, visited = ident, set()
        while True:
            if current in visited or current not in known:
                errors[ident] = "Agrupación circular o daño de destino inexistente."
                break
            visited.add(current)
            quote = quotes.get(current) or {}
            target = quote.get("incluido_en", "") if isinstance(quote, Mapping) else None
            if not isinstance(target, str):
                errors[ident] = "Agrupación inválida."
                break
            if not target:
                groups.setdefault(current, []).append(ident)
                break
            current = target
    candidates = {}
    for ident, members in groups.items():
        quote = quotes.get(ident) or {}
        if not isinstance(quote, Mapping):
            errors[ident] = "Referencia de reparación inválida."
            continue
        zone = _clean(quote.get("zona_afectada"))
        stages = quote.get("trabajos", [])
        if not zone or len(zone) > 120:
            errors[ident] = "Identifica la zona física para evitar contar el mismo daño en varias fotos."
            continue
        if not isinstance(stages, (list, tuple)) or not stages or any(not isinstance(s, str) or s not in WORK_STAGES for s in stages):
            errors[ident] = "Declara las fases de trabajo incluidas en el precio."
            continue
        candidates[ident] = (" ".join(zone.casefold().split()), set(stages))
        if any(known[member]["requiere_revision"] for member in members) and quote.get("revision_tecnica_confirmada") is not True:
            errors[ident] = "Requiere evaluación técnica previa declarada antes de presupuestar esta lesión."
    owners = {}
    for ident, (zone, stages) in candidates.items():
        for stage in stages:
            key = (zone, stage)
            if key in owners:
                reason = "Posible doble cobro en la misma zona y fase; agrupa los daños en una sola partida o corrige el alcance."
                errors[ident] = reason
                errors[owners[key]] = reason
            owners[key] = ident
    items, missing = [], []
    for ident, members in groups.items():
        damage = known[ident]
        title = damage["dano"] + " · " + damage["elemento"]
        if ident in errors:
            missing.append(MissingItem(ident, title, errors[ident]))
            continue
        quote = quotes.get(ident) or {}
        observed = {code: "bueno" for code in ASSETS[asset]["criterios"]}
        observed[damage["criterio"]] = "regular"
        validated = build_declared_budget(asset, observed, area, {damage["criterio"]: quote},
                                          photos_complete=True, allowed_units=set(UNIT_LABELS))
        if validated.missing:
            missing.append(MissingItem(ident, title, validated.missing[0].motivo))
            continue
        items.append(replace(validated.items[0], titulo=title, id_partida=ident, danos_incluidos=tuple(members),
                             zona_afectada=_clean(quote["zona_afectada"]), trabajos=tuple(sorted(set(quote["trabajos"]))),
                             revision_tecnica_declarada=quote.get("revision_tecnica_confirmada") is True))
    for ident, reason in errors.items():
        if ident not in groups:
            missing.append(MissingItem(ident, known[ident]["dano"], reason))
    covered = {damage["criterio"] for damage in known.values()}
    for code, (title, _, _) in ASSETS[asset]["criterios"].items():
        if statuses.get(code) in DAMAGED and code not in covered:
            missing.append(MissingItem(code, title, "El criterio tiene deterioro pero falta identificar la lesión concreta; repite una toma de detalle."))
    unobserved = tuple(title for code, (title, _, _) in ASSETS[asset]["criterios"].items()
                       if statuses.get(code) not in {"bueno", "regular", "malo"})
    limitations = []
    if unobserved:
        limitations.append("Hay criterios no verificables; no se descartan daños fuera de lo observado.")
    if not photos_complete:
        limitations.append("Faltan tomas guiadas o alguna foto no pudo procesarse.")
    if not detail_complete:
        limitations.append("El detalle de lesiones está incompleto o contiene propuestas descartadas; requiere nuevas tomas o revisión.")
    if items:
        low = sum((Decimal(str(item.subtotal_min)) for item in items), Decimal(0))
        high = sum((Decimal(str(item.subtotal_max)) for item in items), Decimal(0))
        totals = (float(low), float(high), float(((low + high) / 2).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)))
    elif missing or limitations or known:
        totals = (None, None, None)
    else:
        totals = (0.0, 0.0, 0.0)
    declared_area = _number(area)
    return RepairBudget(items=tuple(items), missing=tuple(missing), total_min=totals[0], total_max=totals[1],
                        total_central=totals[2], complete=not missing and not limitations,
                        area=declared_area if declared_area and declared_area > 0 else None,
                        damaged=len(known) or sum(statuses.get(c) in DAMAGED for c in ASSETS[asset]["criterios"]),
                        unobserved=unobserved, limitations=tuple(limitations))
