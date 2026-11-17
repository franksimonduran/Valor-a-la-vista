import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from valor_a_la_vista import model_release as release
from valor_a_la_vista.training import count_metrics


def completed_fixture(root):
    run = root / "runs" / "synthetic_experiment"
    weights = run / "train" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"synthetic testing fixture only, not actual model")
    fixed = {**count_metrics({"tp": 48, "fp": 0, "fn": 48}), "confidence": .35, "matching_iou": .5,
             "nms_iou": .7, "max_det": 100, "images": 48}
    report = {"status": "completado_experimental", "class_names": {"0": "residuos"}, "completed_epochs": 3,
              "checkpoint": str(weights), "sha256_checkpoint": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "sha256_dataset_manifest": "a"*64, "dataset_audit": {"splits": {"train": 204, "val": 48, "test": 48}},
              "parameters": {"imgsz": 416}, "test": {"mAP50": .5, "mAP50_95": .3},
              "test_at_conf_035_iou_050": fixed, "limitations": ["Solo residuos, datos de piloto sin negativos revisados."]}
    (run / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
    predictions = [{"image_id": str(i), "tp": 1, "fp": 0, "fn": 1} for i in range(48)]
    (run / "test_predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    return run, report, weights


class ModelReleaseTests(unittest.TestCase):
    def test_publishes_only_completed_tested_artifact_and_hash(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(release, "DATA_ROOT", Path(temp)):
            root = Path(temp)
            run, report, source = completed_fixture(root)
            target = root / "models" / "terrenos" / "best.pt"
            card = release.publish_release(run, target=target)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertEqual(card["status"], "experimental")
            self.assertFalse(card["validated_for_real_use"])
            self.assertEqual(card["class_names"], {"0": "residuos"})
            self.assertEqual(release.read_active_card(target), card)

    def test_publish_same_model_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(release, "DATA_ROOT", Path(temp)):
            root = Path(temp)
            run, _, _ = completed_fixture(root)
            target = root / "models" / "terrenos" / "best.pt"
            first = release.publish_release(run, target=target)
            self.assertEqual(first, release.publish_release(run, target=target))

    def test_never_overwrites_unfamiliar_active_weights(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(release, "DATA_ROOT", Path(temp)):
            root = Path(temp)
            run, _, _ = completed_fixture(root)
            target = root / "models" / "terrenos" / "best.pt"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"user work")
            with self.assertRaises(ValueError):
                release.publish_release(run, target=target)
            self.assertEqual(target.read_bytes(), b"user work")

    def test_in_progress_or_failed_training_cannot_be_published(self):
        for status in ("iniciado", "fallido", "entrenado_evaluando"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp, patch.object(release, "DATA_ROOT", Path(temp)):
                run, report, _ = completed_fixture(Path(temp))
                report["status"] = status
                (run / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaises(ValueError):
                    release.prepare_release(run)

    def test_changed_checkpoint_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(release, "DATA_ROOT", Path(temp)):
            run, _, source = completed_fixture(Path(temp))
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                release.prepare_release(run)

    def test_missing_or_partial_test_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(release, "DATA_ROOT", Path(temp)):
            run, _, _ = completed_fixture(Path(temp))
            (run / "test_predictions.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompletas"):
                release.prepare_release(run)

    def test_inconsistent_metrics_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(release, "DATA_ROOT", Path(temp)):
            run, report, _ = completed_fixture(Path(temp))
            report["test_at_conf_035_iou_050"]["precision"] = .9
            (run / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "métricas"):
                release.prepare_release(run)

    def test_app_card_rejects_unsupported_classes_and_fake_real_validation(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(release, "DATA_ROOT", Path(temp)):
            run, _, _ = completed_fixture(Path(temp))
            _, card = release.prepare_release(run)
            for change in ({"class_names": {"0": "person"}}, {"validated_for_real_use": True}, {"test": {"mAP50": float("nan")}},
                           {"inference": {"imgsz": 640}}, {"completed_epochs": 0}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    release.validate_card({**card, **change})

    def test_tampered_active_weights_disable_loading(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(release, "DATA_ROOT", Path(temp)):
            root = Path(temp)
            run, _, _ = completed_fixture(root)
            target = root / "models" / "terrenos" / "best.pt"
            release.publish_release(run, target=target)
            target.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                release.read_active_card(target)
