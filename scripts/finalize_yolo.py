"""Activa explícitamente el piloto evaluado y comprueba su inferencia local."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from valor_a_la_vista import bootstrap
from valor_a_la_vista.model_release import prepare_release, publish_release, ACTIVE_WEIGHTS, file_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--activate-experimental", action="store_true")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    source, card = prepare_release(run)
    if not args.activate_experimental:
        parser.error("Para activar explícitamente este modelo usa --activate-experimental.")
    # Prueba el checkpoint antes de publicar. No se transmite ninguna imagen.
    from valor_a_la_vista.detector import load_terrain_detector, detect_regions, annotated_image
    model, digest = load_terrain_detector(source)
    if digest != card["weights_sha256"]:
        raise ValueError("El checkpoint cambió antes de la prueba de inferencia.")
    manifest_path = run / "dataset_snapshot" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(row for row in manifest["records"] if row["split"] == "test")
    image = (run / "dataset_snapshot" / row["image_path"]).resolve()
    if not image.is_relative_to(run / "dataset_snapshot") or file_sha256(image) != row["sha256"]:
        raise ValueError("La imagen de comprobación no coincide con el experimento.")
    raw = image.read_bytes()
    regions = detect_regions(raw, model)
    smoke = {"status": "ok", "image_id": row["id"], "image_sha256": hashlib.sha256(raw).hexdigest(),
             "regions_proposed": len(regions), "regions": regions, "weights_sha256": digest,
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "note": "Comprobación funcional, no validación visual ni prueba independiente de calidad."}
    # Estos artefactos pertenecen al experimento; no reemplazan imágenes ni etiquetas.
    if not (run / "inference_smoke.json").exists():
        (run / "inference_smoke.json").write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")
        annotated_image(raw, regions).save(run / "inference_example.jpg", quality=90)
    card = publish_release(run)
    report_path = run / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["published_in_app"] = True
    report["active_weights"] = str(ACTIVE_WEIGHTS)
    report["activated_utc"] = card["activated_utc"]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fixed = card["test_fixed_threshold"]
    summary = (f"# YOLO11n · residuos · experimental\n\n"
        f"Entrenamiento completado: {card['completed_epochs']} épocas. Activado: {card['activated_utc']}.\n\n"
        f"- mAP50 en test: {card['test']['mAP50']:.2%}\n"
        f"- mAP50–95 en test: {card['test']['mAP50_95']:.2%}\n"
        f"- Precisión a confianza 0,35 / IoU 0,50: {fixed['precision']:.2%}\n"
        f"- Recall a confianza 0,35 / IoU 0,50: {fixed['recall']:.2%}\n\n"
        "Solo residuos; no maleza, agua, erosión ni todos los daños del terreno. No validado para uso real.\n\n"
        "## Uso en la app\n\nAñade una foto o prepara fotogramas de un video, abre **Detección local con YOLO**, "
        "acepta el aviso experimental y pulsa **Detectar regiones con YOLO**. No necesita clave de OpenAI para la detección local.\n\n"
        "Las cajas y confianzas no son m², diagnóstico ni partidas de presupuesto. Revisa siempre los resultados.\n\n"
        "## Limitaciones\n\n" + "\n".join("- " + text for text in card["limitations"]) +
        f"\n\nSHA-256 de pesos: `{card['weights_sha256']}`.\n\n{card['license']}.\n")
    if not (run / "RESULTADO.md").exists():
        (run / "RESULTADO.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": "activated_experimental", "weights": str(ACTIVE_WEIGHTS), "model_id": card["model_id"],
                      "test": card["test"], "test_fixed_threshold": fixed, "smoke": smoke["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
