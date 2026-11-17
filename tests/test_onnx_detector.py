"""Pruebas sin pesos reales ni llamadas de red para la inferencia ONNX."""
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from valor_a_la_vista import onnx_detector
from valor_a_la_vista.detector import detector_status, load_terrain_detector, detect_regions


class ONNXDetectorTests(unittest.TestCase):
    def fake_card(self, digest):
        return {
            "schema_version": 1, "model_id": "residuos_piloto", "architecture": "YOLO11n",
            "status": "experimental", "class_names": {"0": "residuos"},
            "validated_for_real_use": False, "completed_epochs": 21,
            "weights_sha256": "a" * 64,
            "test": {"mAP50": .568, "mAP50_95": .376},
            "test_fixed_threshold": {"precision": .816, "recall": .505, "f1": .624},
            "inference": {"imgsz": 416, "confidence": .35, "nms_iou": .7, "max_det": 100},
            "limitations": ["Solo residuos; no validado en terrenos reales."],
            "onnx_export": {"format": "yolo11_raw_xywh", "sha256": digest,
                            "imgsz": 416, "opset": 12, "backend": "opencv_dnn"},
        }

    def test_model_card_requires_matching_onnx_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best.onnx"
            card_path = Path(folder) / "model_card.json"
            path.write_bytes("modelo sintético, no ejecutable".encode("utf-8"))
            card_path.write_text(json.dumps(self.fake_card(hashlib.sha256(path.read_bytes()).hexdigest())),
                                 encoding="utf-8")
            self.assertEqual(onnx_detector.read_onnx_card(path, card_path)["completed_epochs"], 21)
            path.write_bytes(b"alterado")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                onnx_detector.read_onnx_card(path, card_path)

    def test_bad_model_card_never_marks_onnx_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder) / "best.onnx"
            weights.write_bytes(b"fake")
            with patch.object(onnx_detector, "ONNX_PATH", weights), \
                 patch.object(onnx_detector, "ONNX_CARD", Path(folder) / "missing.json"):
                ready, msg = detector_status()
                self.assertFalse(ready)
                self.assertIn("inválido", msg)

    def test_onnx_loader_is_selected_without_using_torch(self):
        fake = onnx_detector.OpenCVYoloDetector(Mock(), Path("best.onnx"))
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder) / "best.onnx"
            weights.write_bytes(b"fake")
            with patch.object(onnx_detector, "ONNX_PATH", weights), \
                 patch.object(onnx_detector, "load_onnx_detector", return_value=(fake, "b" * 64)) as loader:
                actual, digest = load_terrain_detector()
                self.assertIs(actual, fake)
                self.assertEqual(digest, "b" * 64)
                loader.assert_called_once()

    def test_letterbox_and_nms_on_single_residue_detection(self):
        image = Image.new("RGB", (400, 200), "#c7ab82")
        out = io.BytesIO()
        image.save(out, format="PNG")
        # RGB image 400x200 -> letterbox 416x416, pad top 104.
        # Detección dentro de la imagen con centro (208,208), dimensiones 166.4.
        net = Mock()
        net.forward.return_value = np.array(
            [[[208.0], [208.0], [166.4], [166.4], [0.82]]], dtype=np.float32)
        model = onnx_detector.OpenCVYoloDetector(net, Path("test.onnx"))
        regions = detect_regions(out.getvalue(), model)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["tipo"], "residuos")
        self.assertEqual(regions[0]["confianza_detector"], .82)
        for actual, expected in zip(regions[0]["caja_normalizada_xyxy"], (.3, .1, .7, .9)):
            self.assertAlmostEqual(actual, expected, places=4)
        self.assertNotIn("m2", regions[0])
        self.assertEqual(net.setInput.call_count, 1)
        self.assertEqual(net.forward.call_count, 1)

    def test_incompatible_output_and_low_confidence(self):
        out = io.BytesIO()
        Image.new("RGB", (320, 240), "green").save(out, "PNG")
        net = Mock()
        model = onnx_detector.OpenCVYoloDetector(net, Path("test.onnx"))
        net.forward.return_value = np.zeros((1, 6, 4), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "ONNX"):
            detect_regions(out.getvalue(), model)
        net.forward.return_value = np.array(
            [[[208.], [208.], [100.], [100.], [0.1]]], dtype=np.float32)
        self.assertEqual(detect_regions(out.getvalue(), model), [])
