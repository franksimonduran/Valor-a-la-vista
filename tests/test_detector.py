import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from valor_a_la_vista.detector import load_terrain_detector, detect_regions, annotated_image, detector_status


class DetectorTests(unittest.TestCase):
    def image(self):
        out = io.BytesIO()
        Image.new("RGB", (100, 100), "green").save(out, "PNG")
        return out.getvalue()

    def model(self, coords=None, confidence=.8, kind="maleza"):
        box = types.SimpleNamespace(cls=np.array([0]), conf=np.array([confidence]), xyxy=np.array([coords or [10, 20, 80, 90]]))
        return types.SimpleNamespace(names={0: kind}, predict=Mock(return_value=[types.SimpleNamespace(boxes=[box])]))

    def test_missing_weights_do_not_load_or_download(self):
        factory = Mock()
        with tempfile.TemporaryDirectory() as temp, patch("valor_a_la_vista.detector.DATA_ROOT", Path(temp)):
            with self.assertRaises(ValueError):
                load_terrain_detector(Path(temp) / "missing.pt", factory=factory)
        factory.assert_not_called()

    def test_requires_local_trusted_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.pt"
            path.write_bytes(b"synthetic-only")
            with patch("valor_a_la_vista.detector.DATA_ROOT", Path(temp)):
                factory = Mock(return_value=self.model())
                model, digest = load_terrain_detector(path, factory=factory)
                self.assertEqual(len(digest), 64)
                self.assertEqual(model.names, {0: "maleza"})
            with patch("valor_a_la_vista.detector.DATA_ROOT", Path(temp) / "other"), self.assertRaises(ValueError):
                load_terrain_detector(path, factory=factory)

    def test_generic_yolo_classes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.pt"
            path.write_bytes(b"synthetic")
            with patch("valor_a_la_vista.detector.DATA_ROOT", Path(temp)), self.assertRaisesRegex(ValueError, "genérico"):
                load_terrain_detector(path, factory=Mock(return_value=self.model(kind="person")))

    def test_detection_returns_normalized_boxes_not_area_or_budget(self):
        model = self.model()
        regions = detect_regions(self.image(), model)
        self.assertEqual(regions[0]["caja_normalizada_xyxy"], [.1, .2, .8, .9])
        self.assertNotIn("area", regions[0])
        self.assertNotIn("precio", regions[0])
        self.assertFalse(model.predict.call_args.kwargs["save"])
        self.assertEqual(model.predict.call_args.kwargs["device"], "cpu")
        self.assertEqual(model.predict.call_args.kwargs["imgsz"], 416)
        self.assertEqual(model.predict.call_args.kwargs["iou"], .7)
        self.assertEqual(model.predict.call_args.kwargs["max_det"], 100)

    def test_bad_boxes_and_confidences_not_accepted(self):
        for coords, confidence in (([80, 20, 10, 90], .8), ([-1, 20, 80, 90], .8),
                                  ([10, 20, 101, 90], .8), ([10, 20, 80, 90], float("nan")),
                                  ([10, 20, 80, 90], .1)):
            with self.subTest(coords=coords, confidence=confidence):
                self.assertEqual(detect_regions(self.image(), self.model(coords, confidence)), [])

    def test_render_does_not_change_original(self):
        raw = self.image()
        rendered = annotated_image(raw, detect_regions(raw, self.model()))
        self.assertEqual(rendered.size, (100, 100))
        self.assertNotEqual(rendered.tobytes(), Image.open(io.BytesIO(raw)).tobytes())

    def test_startup_status_without_weights(self):
        with tempfile.TemporaryDirectory() as temp, patch("valor_a_la_vista.detector.DEFAULT_WEIGHTS", Path(temp) / "absent.pt"):
            ready, note = detector_status()
        self.assertFalse(ready)
        self.assertIn("no activado", note)

    def test_missing_card_prevents_active_model_loading(self):
        with tempfile.TemporaryDirectory() as temp, patch("valor_a_la_vista.detector.DEFAULT_WEIGHTS", Path(temp) / "best.pt"):
            (Path(temp) / "best.pt").write_bytes(b"unknown file")
            factory = Mock()
            with self.assertRaises(ValueError):
                load_terrain_detector(factory=factory)
            factory.assert_not_called()
