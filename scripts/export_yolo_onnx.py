"""Exporta el checkpoint LOCAL verificado a ONNX para Render Free.

Ejecutar SOLO en un equipo de confianza con el checkpoint entrenado por el
propietario. Cargar un archivo .pt de terceros puede ejecutar código.
No altera el modelo original ni publica automáticamente ningún archivo.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from valor_a_la_vista.model_release import read_active_card

PT = ROOT / ".data" / "models" / "terrenos" / "best.pt"
DEST = ROOT / "assets" / "models" / "terrenos"


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    card = read_active_card(PT)
    if card["class_names"] != {"0": "residuos"} or card["validated_for_real_use"] is not False:
        raise ValueError("Solo se permite publicar el piloto experimental de residuos.")
    output = DEST / "best.onnx"
    metadata = DEST / "model_card.json"
    if output.exists() or metadata.exists():
        raise FileExistsError("El modelo de despliegue ya existe. No se sobrescribe; respalda y revisa primero.")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("Instala las dependencias locales: pip install -r requirements-yolo.txt") from None
    import cv2
    import numpy as np

    print("Exportando un checkpoint LOCAL con SHA-256 validado...", flush=True)
    model = YOLO(str(PT))
    classes = model.names
    classes = {int(k): str(v) for k, v in classes.items()}
    if classes != {0: "residuos"}:
        raise ValueError("Las clases de los pesos no coinciden con el piloto de residuos.")

    # Salida cruda YOLO11n de una clase. NMS se calcula en OpenCV en la web.
    exported = Path(model.export(format="onnx", imgsz=416, batch=1,
                                 opset=12, simplify=False, dynamic=False,
                                 nms=False, device="cpu"))
    if not exported.is_file() or not 0 < exported.stat().st_size <= 50 * 1024 * 1024:
        raise RuntimeError("La exportación ONNX no generó un archivo del tamaño esperado.")

    # Una prueba real de inferencia, sin enviar fotos ni llamar a OpenAI.
    try:
        net = cv2.dnn.readNetFromONNX(str(exported))
        net.setInput(np.zeros((1, 3, 416, 416), dtype=np.float32))
        prediction = np.asarray(net.forward())
        if (prediction.ndim != 3 or prediction.shape[0] != 1
                or 5 not in prediction.shape[1:]
                or not np.all(np.isfinite(prediction))):
            raise ValueError("La salida ONNX no corresponde al detector de residuos.")
    except (cv2.error, ValueError) as exc:
        raise RuntimeError("OpenCV no pudo ejecutar el ONNX. Conserva los pesos originales; revisa exportación y opset.") from exc

    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(exported, output)
    card["onnx_export"] = {
        "format": "yolo11_raw_xywh",
        "sha256": file_hash(output),
        "imgsz": 416,
        "opset": 12,
        "backend": "opencv_dnn",
        "smoke_output_shape": list(prediction.shape),
    }
    metadata.write_text(json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    from valor_a_la_vista.onnx_detector import read_onnx_card
    read_onnx_card(output, metadata)
    print(json.dumps({
        "status": "exported_verified_experimental",
        "model_id": card["model_id"],
        "onnx_size_mb": round(output.stat().st_size / (1024 ** 2), 2),
        "onnx_sha256": card["onnx_export"]["sha256"],
        "output_shape": list(prediction.shape),
        "files_to_publish": [str(output.relative_to(ROOT)), str(metadata.relative_to(ROOT))],
    }, ensure_ascii=False, indent=2))
    print("No es una validación de precisión con terrenos reales.", flush=True)


if __name__ == "__main__":
    main()
