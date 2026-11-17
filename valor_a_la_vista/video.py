"""Muestreo local y acotado de videos; no transmite video ni audio a la API."""
from __future__ import annotations

import hashlib
import io
import math
import tempfile
from pathlib import Path

from PIL import Image

MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_VIDEO_SECONDS = 60
MAX_VIDEO_FRAMES = 12
MAX_VIDEO_PIXELS = 3840 * 2160
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".webm"}
TEMP_ROOT = Path(__file__).resolve().parent.parent / ".data" / "tmp"


def timestamp(seconds):
    """Tiempo aproximado basado en índice/fps del archivo, no reloj certificado."""
    milliseconds = round(seconds * 1000)
    minutes, remainder = divmod(milliseconds, 60_000)
    return f"{minutes:02d}:{remainder / 1000:06.3f}"


def extract_video_frames(raw, filename, *, max_frames=8):
    if not raw or len(raw) > MAX_VIDEO_BYTES:
        raise ValueError("Video vacío o demasiado grande; máximo 100 MB.")
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1][:160]
    suffix = Path(name).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise ValueError("Usa MP4, MOV, AVI o WebM.")
    # No aceptar listas de reproducción/URLs disfrazadas con extensión de video.
    avi = raw[:4] == b"RIFF" and raw[8:12] == b"AVI "
    webm = raw[:4] == b"\x1a\x45\xdf\xa3"
    isobmff = raw[4:8] in {b"ftyp", b"moov", b"mdat", b"wide", b"free"}
    if not ((suffix == ".avi" and avi) or (suffix == ".webm" and webm) or (suffix in {".mp4", ".mov"} and isobmff)):
        raise ValueError("El contenido no corresponde a un contenedor de video admitido; no se aceptan enlaces ni listas de reproducción.")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 1 <= max_frames <= MAX_VIDEO_FRAMES:
        raise ValueError("Selecciona entre 1 y 12 fotogramas.")
    try:
        import cv2
    except ImportError:
        raise RuntimeError("Falta OpenCV. Instala las dependencias de requirements.txt para analizar videos.") from None
    digest = hashlib.sha256(raw).hexdigest()
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    frames, seen, notices = [], set(), []
    with tempfile.TemporaryDirectory(prefix="video-", dir=TEMP_ROOT) as temporary:
        path = Path(temporary) / ("entrada" + suffix)
        path.write_bytes(raw)
        capture = None
        try:
            capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            if not capture.isOpened():
                raise ValueError("No se pudo abrir el video. Prueba MP4 con códec H.264.")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (not all(math.isfinite(v) and v > 0 for v in (fps, count_value, width, height))
                    or fps > 240 or count_value > 14400 or count_value != int(count_value)):
                raise ValueError("Duración, fps o dimensiones del video no utilizables.")
            count = int(count_value)
            duration = count / fps
            if duration > MAX_VIDEO_SECONDS + .001:
                raise ValueError("El video debe durar como máximo 60 segundos; recórtalo antes de subirlo.")
            if width * height > MAX_VIDEO_PIXELS:
                raise ValueError("El video supera resolución 4K; reduce su resolución.")
            samples = min(max_frames, count)
            indices = sorted({round(i * (count - 1) / max(1, samples - 1)) for i in range(samples)})
            for index in indices:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    notices.append(f"No se pudo buscar el fotograma {index}.")
                    continue
                ok, frame = capture.read()
                if not ok or frame is None:
                    notices.append(f"No se pudo leer el fotograma {index}.")
                    continue
                if frame.shape[0] * frame.shape[1] > MAX_VIDEO_PIXELS:
                    raise ValueError("Un fotograma supera resolución 4K.")
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((1280, 1280))
                output = io.BytesIO()
                image.save(output, "JPEG", quality=80, optimize=True)
                encoded = output.getvalue()
                fingerprint = hashlib.sha256(encoded).hexdigest()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                seconds = round(index / fps, 3)
                frames.append({"nombre": f"Video {digest[:8]} · {timestamp(seconds)}.jpg", "raw": encoded,
                    "origen": {"tipo": "video", "video_nombre": name, "video_sha256": digest,
                               "tiempo_s": seconds, "tiempo_aproximado": timestamp(seconds), "indice_frame": index,
                               "fps": fps, "duracion_s": round(duration, 3), "audio_analizado": False}})
        except cv2.error:
            raise ValueError("El video no pudo decodificarse; prueba MP4 H.264 de hasta 60 segundos.") from None
        finally:
            if capture is not None:
                capture.release()
    if not frames:
        raise ValueError("No hay fotogramas legibles en el video.")
    notices.append("Muestreo de fotogramas: puede omitir daños entre capturas. Los tiempos son aproximados; no se analiza audio.")
    return frames, notices
