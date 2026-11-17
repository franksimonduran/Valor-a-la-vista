import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from valor_a_la_vista.core import ASSETS
from valor_a_la_vista.inspection import analyze_photos
from valor_a_la_vista.evaluation import LABELS, calculate_metrics
from valor_a_la_vista.vision import PROMPT_VERSION, SCHEMA_VERSION, openai_config


def validate_manifest(entries, base):
    if not isinstance(entries, list) or not entries:
        raise ValueError("El manifiesto debe ser una lista no vacía.")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Cada entrada debe ser un objeto.")
        ident, asset, shot, image, labels = (entry.get(k) for k in ("id", "activo", "toma", "imagen", "etiquetas"))
        if not isinstance(ident, str) or not ident.strip() or ident in seen:
            raise ValueError("Cada imagen necesita un id único y no vacío.")
        seen.add(ident)
        if asset not in ASSETS or shot not in [name for name, _ in ASSETS[asset]["fotos"]]:
            raise ValueError(f"{ident}: activo o toma inválidos.")
        if not isinstance(image, str) or not image or not (base / image).is_file():
            raise ValueError(f"{ident}: archivo de imagen inexistente.")
        if not isinstance(labels, dict) or not labels:
            raise ValueError(f"{ident}: se requiere al menos una etiqueta.")
        if any(code not in ASSETS[asset]["criterios"] or status not in LABELS for code, status in labels.items()):
            raise ValueError(f"{ident}: criterio o estado inválido.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    api_key, configured_model, base_url = openai_config()
    parser.add_argument("--model", default=configured_model)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(entries, args.manifest.parent)
    if not api_key:
        parser.error("Falta OPENAI_API_KEY; las imágenes se enviarán a OpenAI.")
    print("Las imágenes del manifiesto se enviarán a OpenAI y puede generar cargos.")
    records, lesions, cache = [], [], {}
    for entry in entries:
        try:
            result = analyze_photos(entry["activo"], {entry["toma"]: (args.manifest.parent / entry["imagen"]).read_bytes()},
                                    model=args.model, api_key=api_key, cache=cache, base_url=base_url)
            states = result["statuses"]
            error = "; ".join(result["fallos"]) or None
            lesions.append({"id": entry["id"], "danos": result["danos"], "detalle_completo": result["detalle_completo"],
                            "advertencias": result["advertencias"]})
        except Exception as ex:
            states, error = {}, str(ex)
        for criterion, truth in entry["etiquetas"].items():
            records.append({"id": entry["id"], "asset": entry["activo"], "criterio": criterion,
                            "referencia": truth, "prediccion": states.get(criterion, "no_visible"),
                            "error": error})
        print(f"Evaluada: {entry['id']}" + (f" (error: {error})" if error else ""))
    output = {"fecha_utc": datetime.now(timezone.utc).isoformat(), "modelo": args.model,
              "version_prompt": PROMPT_VERSION, "version_esquema": SCHEMA_VERSION,
              "errores": sum(bool(r["error"]) for r in records), "registros": records,
              "metricas": calculate_metrics(records), "lesiones_por_imagen": lesions,
              "limitacion": "Métricas descriptivas sobre el conjunto proporcionado; no validan uso financiero."}
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Informe guardado: {args.output}")
    return 1 if output["errores"] else 0


if __name__ == "__main__":
    sys.exit(main())
