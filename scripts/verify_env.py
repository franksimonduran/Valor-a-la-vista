"""Comprueba el entorno local sin descargar modelos."""
import importlib.util
import importlib.metadata
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from valor_a_la_vista import bootstrap  # noqa: E402

print(f"Python: {sys.executable}")
print(f"Entorno virtual: {sys.prefix}")
print(f"Entorno esperado: {root / '.venv'}")
for name in ("HF_HOME", "TORCH_HOME", "PIP_CACHE_DIR"):
    print(f"{name}: {os.environ.get(name, '(no configurado en este proceso)')}")
errors = []
for name, distribution in (("streamlit", "streamlit"), ("PIL", "Pillow"), ("numpy", "numpy"), ("requests", "requests"), ("cv2", "opencv-python")):
    try:
        __import__(name)
        print(f"{name}: {importlib.metadata.version(distribution)}")
    except ImportError:
        errors.append(name)
        print(f"{name}: NO disponible")
for name in ("torch", "transformers"):
    print(f"{name} (opcional): {'instalado' if importlib.util.find_spec(name) else 'no instalado; no necesario para la app'}")
from valor_a_la_vista.vision import openai_config
key, model, base_url = openai_config()
print(f"OpenAI: {'clave configurada' if key else 'falta clave'}; modelo: {model}; API: {base_url}")
print("No se enviaron fotos ni se hicieron llamadas a OpenAI.")
sys.exit(1 if errors else 0)
