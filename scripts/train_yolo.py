"""Entrena y evalúa YOLO local en el piloto TACO auditado, sin publicar en la app."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from valor_a_la_vista import bootstrap  # noqa: E402
from valor_a_la_vista.dataset import audit_dataset, dataset_yaml
from valor_a_la_vista.training import matching_counts, count_metrics

DATA = ROOT / ".data"


def local_file(value: str, suffixes: set[str]) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(DATA.resolve()) or not path.is_file() or path.suffix.lower() not in suffixes:
        raise argparse.ArgumentTypeError("Se requiere un archivo local existente dentro de .data.")
    return path


def standard_metrics(result):
    return {"precision_resumen_ultralytics": float(result.box.mp), "recall_resumen_ultralytics": float(result.box.mr),
            "mAP50": float(result.box.map50), "mAP50_95": float(result.box.map),
            "conf_minima_curva": .001,
            "nota": "AP se calcula sobre la curva; precisión/recall de este resumen no son necesariamente el punto de confianza 0.35."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=lambda value: local_file(value, {".yaml", ".yml"}))
    parser.add_argument("--weights", required=True, type=lambda value: local_file(value, {".pt"}))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--name", default="residuos_yolo11n_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--allow-unreviewed-pilot", action="store_true", help="Acepta solo un experimento, no una validación de uso real.")
    args = parser.parse_args(argv)
    if not (1 <= args.epochs <= 100 and 128 <= args.imgsz <= 640 and 1 <= args.batch <= 8
            and 1 <= args.threads <= 4 and 0 <= args.freeze <= 22 and 1 <= args.patience <= 30):
        parser.error("Parámetros fuera de los límites de este entrenamiento piloto.")
    if not args.name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in args.name):
        parser.error("Nombre de experimento inválido.")
    run = DATA / "runs" / args.name
    if run.exists():
        parser.error("Se conserva el experimento existente; elige otro --name.")
    audited = audit_dataset(args.dataset.parent)
    if audited["review_status"] != "revisado" and not args.allow_unreviewed_pilot:
        parser.error("El piloto está pendiente de revisión. Para un experimento explícito usa --allow-unreviewed-pilot.")
    manifest_path = args.dataset.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    started = datetime.now(timezone.utc).isoformat()
    run.mkdir(parents=True, exist_ok=False)
    # Ultralytics escribe labels/*.cache. Trabaja en una copia para conservar
    # intactas las imágenes, etiquetas y la auditoría del lote descargado.
    snapshot = run / "dataset_snapshot"
    snapshot.mkdir()
    for folder in ("images", "labels"):
        shutil.copytree(args.dataset.parent / folder, snapshot / folder)
    for filename in ("manifest.json", "LICENSE-DATASET.txt", "README.md"):
        source = args.dataset.parent / filename
        if source.is_file():
            shutil.copyfile(source, snapshot / filename)
    training_dataset = snapshot / "dataset.yaml"
    training_dataset.write_text(dataset_yaml(snapshot), encoding="utf-8")
    audit_dataset(snapshot)
    report = {"status": "iniciado", "started_utc": started, "class_names": {"0": "residuos"}, "validated_for_real_use": False,
              "published_in_app": False, "dataset": str(args.dataset), "dataset_audit": audited,
              "training_snapshot": str(training_dataset),
              "sha256_dataset_manifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "base_weights": str(args.weights), "sha256_base_weights": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "parameters": vars(args) | {"dataset": str(args.dataset), "weights": str(args.weights)},
              "limitations": ["Solo residuos, no otros daños del terreno.", "Sin negativos revisados; puede haber falsas alarmas.",
                              "Anotaciones públicas aún sin revisión completa local ni permisos individuales verificados.",
                              "Test separado por lote, no validación independiente garantizada por terreno físico ni en Perú.",
                              "El modelo no mide m² ni calcula precios; no es diagnóstico ni garantía financiera."]}
    def save_report():
        (run / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_report()
    # CPU y sin telemetría/sincronización. No hay llamadas a OpenAI ni a entrenamiento cloud.
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ.pop("ULTRALYTICS_API_KEY", None)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["COMET_MODE"] = "DISABLED"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.chdir(DATA)
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO, settings
        from ultralytics.utils.callbacks import platform as platform_callbacks
        from ultralytics.utils.events import events
        torch.set_num_threads(args.threads)
        settings.update({"datasets_dir": str(DATA / "datasets"), "weights_dir": str(DATA / "models"),
                         "runs_dir": str(DATA / "runs"), "sync": False,
                         "wandb": False, "comet": False, "mlflow": False, "clearml": False,
                         "tensorboard": False, "dvc": False, "raytune": False})
        # La plataforma puede activarse con una clave guardada aunque sync=False.
        # Deshabilita sus callbacks solo en este proceso, sin borrar claves del usuario.
        platform_callbacks.callbacks.clear()
        events.enabled = False
        report["versions"] = {"torch": torch.__version__, "ultralytics": ultralytics.__version__}
        report["hardware"] = {"cuda_available": torch.cuda.is_available(), "cpu_threads": args.threads}
        save_report()
        model = YOLO(str(args.weights), task="detect")
        model.train(data=str(training_dataset), epochs=args.epochs, device=args.device, imgsz=args.imgsz, batch=args.batch,
                    workers=0, cache=False, amp=False, freeze=args.freeze, patience=args.patience,
                    seed=42, deterministic=True, optimizer="AdamW", lr0=.001, cos_lr=True,
                    mosaic=.5, close_mosaic=5, project=str(run), name="train", exist_ok=False,
                    plots=False, save=True, val=True, verbose=False)
        best = run / "train" / "weights" / "best.pt"
        if not best.is_file():
            raise RuntimeError("El entrenamiento no guardó best.pt.")
        detector = YOLO(str(best), task="detect")
        if detector.names != {0: "residuos"}:
            raise RuntimeError("Clases finales incompatibles con el piloto.")
        report["checkpoint"] = str(best)
        report["sha256_checkpoint"] = hashlib.sha256(best.read_bytes()).hexdigest()
        with (run / "train" / "results.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        report["completed_epochs"] = len(rows)
        report["status"] = "entrenado_evaluando"
        save_report()
        for split in ("val", "test"):
            result = detector.val(data=str(training_dataset), split=split, device=args.device, imgsz=args.imgsz,
                                  batch=args.batch, workers=0, plots=False, conf=.001, verbose=False,
                                  project=str(run), name="evaluation_" + split, exist_ok=False)
            report[split] = standard_metrics(result)
            save_report()
        counts = {"tp": 0, "fp": 0, "fn": 0}
        fixed_rows = []
        for row in manifest["records"]:
            if row["split"] != "test":
                continue
            image = snapshot / row["image_path"]
            output = detector.predict(source=str(image), conf=.35, iou=.7, imgsz=args.imgsz, device=args.device,
                                      max_det=100, save=False, verbose=False)[0]
            predictions = [[*box.xyxy[0].tolist(), float(box.conf[0])] for box in output.boxes]
            targets = []
            for line in (snapshot / row["label_path"]).read_text(encoding="utf-8").splitlines():
                _, cx, cy, w, h = map(float, line.split())
                targets.append([(cx-w/2)*row["width"], (cy-h/2)*row["height"],
                                (cx+w/2)*row["width"], (cy+h/2)*row["height"]])
            matched = matching_counts(predictions, targets)
            fixed_rows.append({"image_id": row["id"], **matched, "predictions": predictions})
            for key in counts:
                counts[key] += matched[key]
        report["test_at_conf_035_iou_050"] = {**count_metrics(counts), "confidence": .35, "matching_iou": .5,
                                             "nms_iou": .7, "max_det": 100, "images": len(fixed_rows)}
        (run / "test_predictions.json").write_text(json.dumps(fixed_rows, indent=2), encoding="utf-8")
        report["status"] = "completado_experimental"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_report()
        print("TRAINING_COMPLETE " + json.dumps({"run": str(run), "epochs": report["completed_epochs"],
              "best": str(best), "test": report["test"], "fixed_threshold": report["test_at_conf_035_iou_050"]}), flush=True)
    except Exception as exc:
        report["status"] = "fallido"
        report["error"] = str(exc)
        save_report()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
