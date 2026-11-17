"""Descarga explícita de pesos base oficiales YOLO11n; no entrena ni usa la API."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / ".data" / "models" / "base" / "yolo11n.pt"
URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
EXPECTED_SIZE = 5613764
HOSTS = {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}


def main():
    metadata = TARGET.with_suffix(".source.json")
    if TARGET.exists() or metadata.exists():
        if TARGET.is_file() and metadata.is_file():
            record = json.loads(metadata.read_text(encoding="utf-8"))
            digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()
            if record.get("source_url") == URL and record.get("sha256") == digest:
                print("Pesos base ya disponibles y hash local correcto:", TARGET)
                return 0
        raise RuntimeError("Se conservan los archivos existentes: revisar pesos y procedencia antes de continuar.")
    url = URL
    raw = None
    with requests.Session() as session:
        session.trust_env = False
        for _ in range(5):
            parsed = urlsplit(url)
            if parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.username or parsed.password:
                raise RuntimeError("Redirección fuera del repositorio oficial permitido.")
            with session.get(url, stream=True, timeout=(15, 90), allow_redirects=False) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    url = urljoin(url, response.headers["Location"])
                    continue
                response.raise_for_status()
                chunks, size = [], 0
                for chunk in response.iter_content(64 * 1024):
                    size += len(chunk)
                    if size > EXPECTED_SIZE:
                        raise RuntimeError("Tamaño distinto del asset oficial seleccionado.")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                break
    if raw is None or len(raw) != EXPECTED_SIZE or not raw.startswith(b"PK"):
        raise RuntimeError("Pesos incompletos o formato no esperado.")
    record = {"source_url": URL, "release": "v8.3.0", "asset_id": 196201276, "size": len(raw),
              "sha256": hashlib.sha256(raw).hexdigest(), "downloaded_utc": datetime.now(timezone.utc).isoformat(),
              "license": "Ultralytics AGPL-3.0 / consultar licencia Enterprise para usos que la requieran",
              "license_url": "https://www.ultralytics.com/license",
              "verification": "Origen HTTPS oficial y tamaño del asset; GitHub no publica digest para este asset antiguo. SHA-256 local registrado, no firma independiente.",
              "purpose": "Base COCO genérica para transferencia, no detector entrenado de residuos de terrenos"}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with TARGET.open("xb") as handle:
        handle.write(raw)
    with metadata.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"path": str(TARGET), "bytes": len(raw), "sha256": record["sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
