"""Presupuesto referencial con datos ingresados por el revisor."""
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


def calculate_budget(items):
    """Valida partidas; un total parcial nunca se presenta como presupuesto completo."""
    if len(items) > 100:
        raise ValueError("El presupuesto admite como máximo 100 partidas.")
    rows = []
    total = Decimal("0")
    complete = bool(items)
    for item in items:
        text = {}
        for key in ("descripcion", "unidad", "fuente", "fecha"):
            value = item.get(key, "")
            if not isinstance(value, str) or len(value) > 300:
                raise ValueError(f"Campo {key} inválido.")
            text[key] = value.strip()
        if not text["descripcion"]:
            raise ValueError("Cada partida necesita una descripción.")
        if text["fecha"]:
            try:
                date.fromisoformat(text["fecha"])
            except ValueError as exc:
                raise ValueError("La fecha debe tener formato AAAA-MM-DD y ser válida.") from exc
        numbers = []
        for key in ("cantidad", "precio_unitario"):
            value = item.get(key)
            if value is None or value == "":
                numbers.append(None)
                continue
            try:
                number = Decimal(str(value))
            except (InvalidOperation, ValueError) as exc:
                raise ValueError(f"{key} debe ser un número válido.") from exc
            if not number.is_finite() or number <= 0 or number > Decimal("100000000"):
                raise ValueError(f"{key} debe ser positivo, finito y como máximo 100000000.")
            numbers.append(number)
        quantity, price = numbers
        subtotal = None
        if quantity is not None and price is not None:
            subtotal = (quantity * price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            total += subtotal
        row_complete = bool(text["unidad"] and text["fuente"] and text["fecha"] and subtotal is not None)
        complete = complete and row_complete
        rows.append({**text, "cantidad": float(quantity) if quantity is not None else None,
                     "precio_unitario": float(price) if price is not None else None,
                     "subtotal": float(subtotal) if subtotal is not None else None,
                     "completa": row_complete})
    return {"partidas": rows, "total_conocido": float(total), "completo": complete,
            "total": float(total) if complete else None}
