"""Rutas locales de caché para los modelos del proyecto."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".data" / "cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".data" / "cache" / "torch"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".data" / "config" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".data" / "config" / "matplotlib"))
