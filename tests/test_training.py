import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from valor_a_la_vista.training import box_iou, matching_counts, count_metrics

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("train_yolo_script", ROOT / "scripts" / "train_yolo.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class TrainingMetricsTests(unittest.TestCase):
    def test_iou_identical_and_disjoint(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)

    def test_duplicate_predictions_are_not_two_true_positives(self):
        counts = matching_counts([[0, 0, 10, 10, .9], [0, 0, 10, 10, .8]], [[0, 0, 10, 10]])
        self.assertEqual(counts, {"tp": 1, "fp": 1, "fn": 0})

    def test_one_prediction_cannot_match_two_targets(self):
        counts = matching_counts([[0, 0, 10, 10, .9]], [[0, 0, 10, 10], [1, 1, 11, 11]])
        self.assertEqual(counts, {"tp": 1, "fp": 0, "fn": 1})

    def test_confidence_threshold_held_fixed(self):
        self.assertEqual(matching_counts([[0, 0, 10, 10, .2]], [[0, 0, 10, 10]]), {"tp": 0, "fp": 0, "fn": 1})

    def test_false_positives_on_empty_image(self):
        self.assertEqual(matching_counts([[0, 0, 10, 10, .8]], []), {"tp": 0, "fp": 1, "fn": 0})

    def test_precision_recall_f1(self):
        result = count_metrics({"tp": 3, "fp": 1, "fn": 2})
        self.assertEqual(result["precision"], .75)
        self.assertEqual(result["recall"], .6)
        self.assertAlmostEqual(result["f1"], 2/3)

    def test_zero_counts_are_not_perfect_metrics(self):
        result = count_metrics({"tp": 0, "fp": 0, "fn": 0})
        self.assertEqual((result["precision"], result["recall"], result["f1"]), (0, 0, 0))

    def test_invalid_prediction_or_threshold(self):
        with self.assertRaises(ValueError):
            matching_counts([[0, 0, 10, 10, float("nan")]], [])
        with self.assertRaises(ValueError):
            matching_counts([], [], confidence=0)


class TrainingSafetyTests(unittest.TestCase):
    def test_unknown_weights_outside_data_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "untrusted.pt"
            path.write_bytes(b"not weights")
            with self.assertRaises(Exception):
                trainer.local_file(str(path), {".pt"})

    def test_experiment_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(trainer, "DATA", Path(temp)):
            root = Path(temp)
            (root / "runs" / "existing").mkdir(parents=True)
            dataset, weights = root / "dataset.yaml", root / "model.pt"
            dataset.write_text("", encoding="utf-8")
            weights.write_bytes(b"synthetic")
            with patch.object(trainer, "audit_dataset") as audit, self.assertRaises(SystemExit):
                trainer.main(["--dataset", str(dataset), "--weights", str(weights), "--name", "existing"])
            audit.assert_not_called()

    def test_unreviewed_dataset_needs_explicit_pilot_option(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(trainer, "DATA", Path(temp)):
            root = Path(temp)
            dataset, weights = root / "dataset.yaml", root / "model.pt"
            dataset.write_text("", encoding="utf-8")
            weights.write_bytes(b"synthetic")
            with patch.object(trainer, "audit_dataset", return_value={"review_status": "pendiente"}), self.assertRaises(SystemExit):
                trainer.main(["--dataset", str(dataset), "--weights", str(weights)])
            self.assertFalse((root / "runs").exists())
