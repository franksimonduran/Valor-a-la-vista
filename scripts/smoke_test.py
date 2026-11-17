"""Comprueba arranque HTTP y fuentes locales, sin API ni fotografías."""
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen


def main():
    root = Path(__file__).resolve().parent.parent
    temp = root / ".data" / "temp"
    temp.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(dir=temp, prefix="smoke-") as folder:
        with (Path(folder) / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", str(root / "app.py"),
                 "--server.address", "127.0.0.1", "--server.port", str(port),
                 "--server.headless", "true", "--server.fileWatcherType", "none"],
                cwd=root, stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                base = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Streamlit termino antes de arrancar.")
                    try:
                        with urlopen(base + "/_stcore/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (URLError, TimeoutError):
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError("Streamlit no respondio en 30 segundos.")
                    time.sleep(.2)
                for path in ("/", "/app/static/fonts/inter-400.woff2", "/app/static/fonts/sora-600.woff2"):
                    with urlopen(base + path, timeout=5) as response:
                        if response.status != 200 or not response.read():
                            raise RuntimeError("Recurso no disponible: " + path)
                print("OK: servidor local, pagina y fuentes. Sin llamadas a OpenAI.")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
