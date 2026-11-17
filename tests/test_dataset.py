import hashlib
import importlib.util
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from PIL import Image

from valor_a_la_vista.dataset import (outdoor_candidates, coco_boxes_to_yolo, normalize_training_image,
    split_by_group, audit_records, audit_dataset, dataset_yaml)

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("prepare_taco_dataset", ROOT / "scripts" / "prepare_taco_dataset.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class DatasetTests(unittest.TestCase):
    def image(self, color="green", size=(100, 100)):
        output = io.BytesIO()
        Image.new("RGB", size, color).save(output, "JPEG")
        return output.getvalue()

    def coco(self):
        return {"categories": [{"id": 0, "name": "Bottle"}],
                "images": [{"id": i, "file_name": f"batch_{i}/a.jpg"} for i in range(4)],
                "annotations": [{"id": i, "image_id": i, "category_id": 0} for i in range(4)],
                "scene_annotations": [{"image_id": 0, "background_ids": [3]}, {"image_id": 1, "background_ids": [5]},
                                      {"image_id": 2, "background_ids": [1, 3]}, {"image_id": 3, "background_ids": [2]}]}

    def box(self, **fields):
        return {"category_id": 6, "bbox": [10, 20, 30, 40], "iscrowd": 0, **fields}

    def test_only_dirt_vegetation_without_indoor(self):
        groups = outdoor_candidates(self.coco())
        self.assertEqual(set(groups), {"batch_0", "batch_1"})

    def test_scene_union_conservatively_excludes_indoor(self):
        coco = self.coco()
        coco["scene_annotations"].append({"image_id": 0, "background_ids": [1]})
        self.assertNotIn("batch_0", outdoor_candidates(coco))

    def test_bbox_conversion_merges_to_residuos(self):
        label = coco_boxes_to_yolo([self.box()], 100, 100, category_ids={6})
        self.assertEqual(label, "0 0.25000000 0.40000000 0.30000000 0.40000000\n")

    def test_duplicate_boxes_not_counted_twice(self):
        label = coco_boxes_to_yolo([self.box(), self.box()], 100, 100, category_ids={6})
        self.assertEqual(len(label.splitlines()), 1)

    def test_invalid_boxes_are_not_silently_dropped_or_negatives(self):
        for changed in ({"bbox": [-2, 0, 10, 10]}, {"bbox": [0, 0, 0, 10]}, {"bbox": [0, 0, 200, 10]},
                        {"bbox": [0, 0, float("nan"), 10]}, {"bbox": [0, 0, True, 10]}, {"bbox": []},
                        {"iscrowd": 1}, {"category_id": 999}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                coco_boxes_to_yolo([self.box(**changed)], 100, 100, category_ids={6})
        with self.assertRaises(ValueError):
            coco_boxes_to_yolo([], 100, 100, category_ids={6})

    def test_invalid_dimensions(self):
        for width in (0, -1, True, float("inf")):
            with self.assertRaises(ValueError):
                coco_boxes_to_yolo([self.box()], width, 100, category_ids={6})

    def test_normalization_preserves_aspect_and_limits_resolution(self):
        raw = self.image(size=(1200, 800))
        encoded, width, height, digest, phash = normalize_training_image(raw, 1200, 800)
        self.assertLessEqual(max(width, height), 640)
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), digest)
        self.assertIsInstance(phash, int)

    def test_wrong_aspect_is_not_stretched_to_fit_annotations(self):
        with self.assertRaisesRegex(ValueError, "Proporción"):
            normalize_training_image(self.image(size=(100, 50)), 100, 100)

    def test_exif_rotation_requires_review(self):
        output = io.BytesIO()
        image = Image.new("RGB", (100, 100))
        exif = Image.Exif()
        exif[274] = 6
        image.save(output, "JPEG", exif=exif)
        with self.assertRaisesRegex(ValueError, "Orientación"):
            normalize_training_image(output.getvalue(), 100, 100)

    def records(self):
        hashes = ["0000000000000000", "ffffffffffffffff", "aaaaaaaaaaaaaaaa"]
        return [{"id": str(i), "group": f"batch_{i}", "sha256": str(i), "dhash": hashes[i]} for i in range(3)]

    def test_groups_not_split_and_seed_reproducible(self):
        records = [{**row, "id": row["id"] + str(i)} for row in self.records() for i in range(3)]
        first, second = split_by_group(records), split_by_group(records)
        self.assertEqual(first, second)
        for group in {row["group"] for row in records}:
            self.assertEqual(len({row["split"] for row in first if row["group"] == group}), 1)
        self.assertEqual({row["split"] for row in first}, {"train", "val", "test"})

    def test_too_few_groups_cannot_claim_independent_splits(self):
        with self.assertRaises(ValueError):
            split_by_group(self.records()[:2])

    def test_audit_rejects_exact_near_duplicates_and_group_leakage(self):
        good = split_by_group(self.records())
        self.assertEqual(sum(audit_records(good).values()), 3)
        for changed in ({"sha256": good[0]["sha256"]}, {"dhash": good[0]["dhash"]}, {"group": good[0]["group"]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                audit_records([good[0], {**good[1], **changed}, good[2]])

    def test_dataset_yaml_is_local_one_class_no_download_command(self):
        text = dataset_yaml(ROOT / ".data" / "datasets" / "example")
        self.assertIn("0: residuos", text)
        self.assertNotIn("download:", text)
        self.assertIn("train: images/train", text)


class DownloadSafetyTests(unittest.TestCase):
    def test_untrusted_or_credential_urls_never_connect(self):
        for url in ("http://img.tacodataset.org/a.jpg", "https://img.tacodataset.org.evil.test/a.jpg",
                    "https://user@img.tacodataset.org/a.jpg", "https://img.tacodataset.org/a.jpg?secret=1"):
            with patch.object(prepare.requests, "Session") as session, self.assertRaises(ValueError):
                prepare.download(url)
            session.assert_not_called()

    def test_redirects_not_followed_and_no_app_key(self):
        response = Mock(status_code=302)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        session = Mock()
        session.get.return_value = response
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        with patch.object(prepare.requests, "Session", return_value=session), self.assertRaisesRegex(RuntimeError, "302"):
            prepare.download("https://img.tacodataset.org/a.jpg")
        self.assertFalse(session.get.call_args.kwargs["allow_redirects"])
        self.assertFalse(session.trust_env)
        self.assertNotIn("Authorization", session.get.call_args.kwargs["headers"])

    def test_save_new_preserves_existing_different_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.txt"
            prepare.save_new(path, b"original")
            prepare.save_new(path, b"original")
            with self.assertRaises(ValueError):
                prepare.save_new(path, b"modified")
            self.assertEqual(path.read_bytes(), b"original")

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(prepare, "DATASETS", Path(temp)):
            output = Path(temp) / "existing"
            output.mkdir()
            with patch.object(prepare, "download") as download, self.assertRaises(SystemExit):
                prepare.main(["--output", str(output)])
            download.assert_not_called()


class PublishedDatasetAuditTests(unittest.TestCase):
    def create(self, root):
        records = []
        labels = "0 0.5 0.5 0.2 0.2\n".encode()
        for i in range(3):
            output = io.BytesIO()
            Image.frombytes("RGB", (32, 32), random.Random(i).randbytes(32*32*3)).save(output, "JPEG")
            raw, width, height, digest, phash = normalize_training_image(output.getvalue(), 32, 32)
            records.append({"id": f"synthetic_{i}", "group": f"batch_{i}", "sha256": digest,
                            "dhash": f"{phash:016x}", "width": width, "height": height,
                            "boxes": 1, "sha256_labels": hashlib.sha256(labels).hexdigest(), "raw": raw})
        records = split_by_group(records)
        for row in records:
            row["image_path"] = f"images/{row['split']}/{row['id']}.jpg"
            row["label_path"] = f"labels/{row['split']}/{row['id']}.txt"
            for field, content in (("image_path", row.pop("raw")), ("label_path", labels)):
                path = root / row[field]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        manifest = {"records": records, "count": 3, "class_names": {"0": "residuos"},
                    "splits": audit_records(records), "review_status": "pendiente", "trained": False}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "dataset.yaml").write_text(dataset_yaml(root), encoding="utf-8")
        return manifest

    def test_full_audit_accepts_valid_pairs_without_network(self):
        with tempfile.TemporaryDirectory() as temp, patch("requests.get", side_effect=AssertionError("Sin red")):
            root = Path(temp)
            self.create(root)
            result = audit_dataset(root)
            self.assertEqual((result["images"], result["boxes"]), (3, 3))
            self.assertEqual(result["review_status"], "pendiente")
            self.assertFalse(result["trained"])

    def test_changed_images_or_labels_rejected(self):
        for field in ("image_path", "label_path"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest = self.create(root)
                (root / manifest["records"][0][field]).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "hash"):
                    audit_dataset(root)

    def test_extra_unpaired_files_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create(root)
            (root / "labels" / "extra.txt").write_text("0 .5 .5 .2 .2", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sin pareja"):
                audit_dataset(root)

    def test_wrong_class_rejected_even_if_hash_updated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.create(root)
            row = manifest["records"][0]
            raw = b"1 .5 .5 .2 .2\n"
            (root / row["label_path"]).write_bytes(raw)
            row["sha256_labels"] = hashlib.sha256(raw).hexdigest()
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Etiqueta"):
                audit_dataset(root)

    def test_yaml_does_not_silently_change_to_remote_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create(root)
            (root / "dataset.yaml").write_text("download: https://example.com/archive.zip", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "YAML"):
                audit_dataset(root)

    def test_publisher_end_to_end_with_mocked_downloads_and_license_filter(self):
        images, annotations, scenes = [], [], []
        content = {}
        for i in range(30):
            images.append({"id": i, "file_name": f"batch_{i%5}/image_{i}.jpg", "width": 32, "height": 32,
                           "license": None, "flickr_640_url": f"https://img.tacodataset.org/thumb/{i}_640.jpg",
                           "flickr_url": f"https://img.tacodataset.org/img/{i}.jpg"})
            annotations.append({"id": i, "image_id": i, "category_id": 6, "bbox": [4, 4, 12, 12], "iscrowd": 0})
            scenes.append({"image_id": i, "background_ids": [3]})
            output = io.BytesIO()
            Image.frombytes("RGB", (32, 32), random.Random(i).randbytes(32*32*3)).save(output, "JPEG")
            content[i] = output.getvalue()
        images.append({**images[0], "id": 999, "license": "ODBL (c) OpenLitterMap & Contributors"})
        coco = {"images": images, "annotations": annotations, "scene_annotations": scenes,
                "categories": [{"id": 6, "name": "Glass bottle"}]}
        def fetch(url, path, **kwargs):
            if url == prepare.TACO_ANNOTATIONS_URL:
                return json.dumps(coco).encode()
            if url == prepare.TACO_RECORD_URL:
                return json.dumps({"metadata": {"license": {"id": "cc-by-4.0"}}}).encode()
            ident = int(path.stem.rsplit("_", 1)[1])
            self.assertNotEqual(ident, 999)
            return content[ident]
        with tempfile.TemporaryDirectory() as temp, patch.object(prepare, "DATASETS", Path(temp)), \
             patch.object(prepare, "fetch_cached", side_effect=fetch), \
             patch.object(prepare, "download", side_effect=AssertionError("Sin red")):
            root = Path(temp) / "pilot"
            self.assertEqual(prepare.main(["--count", "30", "--output", str(root)]), 0)
            audited = audit_dataset(root)
            self.assertEqual(audited["images"], 30)
            self.assertFalse(audited["trained"])
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_images_excluded_by_policy"][0]["source_image_id"], 999)
            self.assertTrue((root / "revision.html").is_file())
            self.assertTrue((root / "LICENSE-DATASET.txt").is_file())
