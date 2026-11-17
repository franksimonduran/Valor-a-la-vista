import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("weight_downloader", ROOT / "scripts" / "download_yolo_weights.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class WeightDownloadTests(unittest.TestCase):
    def test_existing_unknown_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "base.pt"
            target.write_bytes(b"user weights")
            with patch.object(downloader, "TARGET", target), patch.object(downloader.requests, "Session") as session:
                with self.assertRaises(RuntimeError):
                    downloader.main()
                session.assert_not_called()
                self.assertEqual(target.read_bytes(), b"user weights")

    def test_known_local_source_and_hash_can_be_reused_without_network(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "base.pt"
            raw = b"synthetic trusted fixture only"
            target.write_bytes(raw)
            target.with_suffix(".source.json").write_text(json.dumps({"source_url": downloader.URL,
                "sha256": hashlib.sha256(raw).hexdigest()}), encoding="utf-8")
            with patch.object(downloader, "TARGET", target), patch.object(downloader.requests, "Session") as session:
                self.assertEqual(downloader.main(), 0)
                session.assert_not_called()

    def test_redirect_to_unknown_domain_is_not_followed(self):
        response = Mock(status_code=302, headers={"Location": "https://evil.test/model.pt"})
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        session = Mock()
        session.get.return_value = response
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as temp, patch.object(downloader, "TARGET", Path(temp)/"base.pt"), \
             patch.object(downloader.requests, "Session", return_value=session):
            with self.assertRaisesRegex(RuntimeError, "Redirección"):
                downloader.main()
            self.assertEqual(session.get.call_count, 1)
            self.assertFalse(session.get.call_args.kwargs["allow_redirects"])
            self.assertFalse(session.trust_env)
