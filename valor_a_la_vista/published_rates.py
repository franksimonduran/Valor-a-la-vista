"""Ejemplo legado de tarifa comercial; no se usa en la interfaz de fuentes oficiales."""
from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from html.parser import HTMLParser

import requests

SOURCE = "https://homesolution.net/pe/about/preciosreferencia/pintor"
SERVICE = "Reparación de paredes por metro cuadrado"


class _TableRows(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag == "td" and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
            self.cell = None


def parse_wall_rate(html: str) -> dict:
    """Acepta únicamente la fila y la aclaración de alcance conocidas."""
    plain = re.sub(r"<[^>]+>", " ", html)
    if not re.search(r"solo mano de obra\.\s*no se incluye costo de materiales", plain, re.I):
        raise ValueError("La fuente no confirma el alcance de mano de obra.")
    parser = _TableRows()
    parser.feed(html)
    rows = [row for row in parser.rows if len(row) == 3 and row[0].casefold() == SERVICE.casefold()]
    if len(rows) != 1:
        raise ValueError("No se encontró una única tarifa de reparación de paredes.")
    values = []
    for text in rows[0][1:]:
        match = re.fullmatch(r"s/\s*(\d{1,6}(?:\.\d{1,2})?)", text, re.I)
        if not match:
            raise ValueError("Precio o moneda no reconocidos.")
        try:
            value = Decimal(match.group(1))
        except InvalidOperation as ex:
            raise ValueError("Precio inválido.") from ex
        values.append(value)
    if values[0] <= 0 or values[1] < values[0]:
        raise ValueError("Rango de precios inválido.")
    return {
        "intervencion": SERVICE,
        "precio_min": str(values[0]),
        "precio_max": str(values[1]),
        "moneda": "PEN",
        "unidad": "m²",
        "alcance": "Solo mano de obra; no incluye materiales. La tarifa es orientativa y puede variar por localidad y complejidad.",
        "fuente": "Home Solution Perú",
        "url": SOURCE,
        "consultado": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fecha_publicacion": None,
    }


def fetch_wall_rate() -> dict:
    """Consulta una sola URL fija; no sigue redirecciones ni ejecuta código web."""
    try:
        with requests.get(SOURCE, timeout=(5, 15), allow_redirects=False, stream=True) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("La fuente redirigió o no está disponible.")
            chunks = []
            size = 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 2_000_000:
                    raise ValueError("La respuesta de la fuente excede el límite.")
                chunks.append(chunk)
            return parse_wall_rate(b"".join(chunks).decode("utf-8", errors="replace"))
    except (requests.RequestException, ValueError) as ex:
        raise RuntimeError("No se pudo comprobar la tarifa de reparación de paredes.") from ex


def calculate_wall_labour(rate: dict, affected_area) -> tuple[Decimal, Decimal]:
    """Calcula mano de obra con un área afectada conocida, no el área del inmueble."""
    try:
        area = Decimal(str(affected_area))
        low = Decimal(rate["precio_min"])
        high = Decimal(rate["precio_max"])
        if not all(number.is_finite() and number > 0 for number in (area, low, high)):
            raise ValueError("El área y los precios deben ser positivos y finitos.")
        if low > high or rate.get("moneda") != "PEN" or rate.get("unidad") != "m²":
            raise ValueError("Unidad, moneda o rango incompatible.")
        cent = Decimal("0.01")
        return ((area * low).quantize(cent, rounding=ROUND_HALF_UP),
                (area * high).quantize(cent, rounding=ROUND_HALF_UP))
    except (InvalidOperation, KeyError, TypeError) as ex:
        raise ValueError("Faltan datos válidos para calcular la mano de obra.") from ex
