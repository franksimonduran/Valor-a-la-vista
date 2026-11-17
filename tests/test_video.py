import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import cv2
import numpy as np

from valor_a_la_vista.video import extract_video_frames, timestamp
from valor_a_la_vista.gallery import add_video_frames, inspection_inputs, media_origins
from valor_a_la_vista.inspection import analyze_photos, photo_fingerprint


def make_video(directory, *, fps=10, count=20, static=False):
    path = Path(directory) / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (96, 64))
    if not writer.isOpened():
        raise RuntimeError("Códec de pruebas no disponible")
    try:
        for i in range(count):
            writer.write(np.full((64, 96, 3), (20, 90, 80 if static else i * 9 % 255), dtype=np.uint8))
    finally:
        writer.release()
    return path.read_bytes()


class VideoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.temp_root = self.directory / "extract"
        patcher = patch("valor_a_la_vista.video.TEMP_ROOT", self.temp_root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_timestamp(self):
        self.assertEqual(timestamp(12.345), "00:12.345")

    def test_real_synthetic_video_sampled_locally(self):
        raw = make_video(self.directory)
        with patch("requests.post", side_effect=AssertionError("Sin API")):
            frames, notes = extract_video_frames(raw, "video.avi", max_frames=4)
        self.assertEqual(len(frames), 4)
        self.assertEqual([f["origen"]["indice_frame"] for f in frames], [0, 6, 13, 19])
        self.assertEqual(frames[-1]["origen"]["tiempo_s"], 1.9)
        self.assertEqual(frames[0]["origen"]["video_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertFalse(frames[0]["origen"]["audio_analizado"])
        self.assertTrue(notes)
        self.assertEqual(list(self.temp_root.iterdir()), [])

    def test_one_frame_already_usable(self):
        frames, _ = extract_video_frames(make_video(self.directory), "video.avi", max_frames=1)
        self.assertEqual(len(frames), 1)
        gallery, notes = add_video_frames({}, frames)
        self.assertFalse(notes)
        photos, categories = inspection_inputs(gallery)
        origins = media_origins(gallery)
        damage = {"tipo": "residuos", "criterio": "superficie", "elemento": "Suelo", "ubicacion": "Centro",
                  "evidencia": "Residuos visibles sobre la superficie", "severidad": "moderada", "certeza": "visible"}
        inspector = Mock(return_value={"hallazgos": [{"criterio": "superficie", "estado": "bueno", "evidencia": "Suelo visible en la toma"}], "danos": [damage]})
        result = analyze_photos("terreno", photos, api_key="test", model="gpt-test", inspector=inspector, photo_categories=categories, origins=origins)
        self.assertEqual(len(result["danos"]), 1)
        self.assertEqual(result["danos"][0]["origen"]["tiempo_s"], 0)
        self.assertEqual(result["metadata"]["origenes_por_foto"], origins)
        self.assertFalse(result["fotos_completas"])
        self.assertTrue(any("Video analizado" in w for w in result["advertencias"]))

    def test_static_video_deduplicated(self):
        frames, _ = extract_video_frames(make_video(self.directory, static=True), "video.avi", max_frames=8)
        self.assertEqual(len(frames), 1)

    def test_rejects_long_video(self):
        with self.assertRaisesRegex(ValueError, "60 segundos"):
            extract_video_frames(make_video(self.directory, fps=1, count=61), "video.avi")
        self.assertEqual(list(self.temp_root.iterdir()), [])

    def test_rejects_corrupt_video_and_cleans_temp(self):
        with self.assertRaises(ValueError):
            extract_video_frames(b"\x00\x00\x00\x18ftypisom" + b"broken" * 10, "video.mp4")
        self.assertEqual(list(self.temp_root.iterdir()), [])

    def test_playlist_disguised_as_video_rejected_before_decoder(self):
        with patch("cv2.VideoCapture") as capture, self.assertRaisesRegex(ValueError, "listas de reproducción"):
            extract_video_frames(b"#EXTM3U\nhttps://example.com/private-video", "fake.mp4")
        capture.assert_not_called()

    def test_rejects_empty_wrong_extension_and_limits(self):
        for raw, name, count in ((b"", "a.mp4", 4), (b"a", "a.exe", 4), (b"a", "a.mp4", 13),
                                  (b"a", "a.mp4", 0), (b"a", "a.mp4", True)):
            with self.subTest(name=name, count=count), self.assertRaises(ValueError):
                extract_video_frames(raw, name, max_frames=count)
        with patch("valor_a_la_vista.video.MAX_VIDEO_BYTES", 1), self.assertRaises(ValueError):
            extract_video_frames(b"xx", "a.mp4")

    def test_origin_changes_signature_but_not_measurements(self):
        photos = {"Frame": b"test"}
        first = photo_fingerprint(photos, origins={"Frame": {"tiempo_s": 0}})
        second = photo_fingerprint(photos, origins={"Frame": {"tiempo_s": 1}})
        self.assertNotEqual(first, second)

    def test_total_gallery_limit_also_applies_to_frames(self):
        frames, _ = extract_video_frames(make_video(self.directory), "video.avi", max_frames=4)
        with patch("valor_a_la_vista.gallery.MAX_PHOTOS", 2):
            gallery, notes = add_video_frames({}, frames)
        self.assertEqual(len(gallery), 2)
        self.assertTrue(notes)

    def test_duplicate_video_does_not_readd_frames(self):
        frames, _ = extract_video_frames(make_video(self.directory), "video.avi", max_frames=2)
        first, _ = add_video_frames({}, frames)
        second, notes = add_video_frames(first, frames)
        self.assertEqual(len(first), len(second))
        self.assertTrue(notes)
