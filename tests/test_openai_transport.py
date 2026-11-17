import os
import types
import unittest
from unittest.mock import Mock, patch

import requests

from valor_a_la_vista.vision import DEFAULT_API_BASE, openai_config, post_openai, validate_api_base


class ConfigTests(unittest.TestCase):
    def config(self, **fields):
        local = types.SimpleNamespace(OPENAI_API_KEY="test", OPENAI_MODEL="gpt-4.1", API_BASE_URL=DEFAULT_API_BASE)
        for name, value in fields.items():
            setattr(local, name, value)
        return types.SimpleNamespace(config=local)

    def test_environment_precedence(self):
        with patch.dict("sys.modules", {"config": self.config()}), patch.dict(os.environ, {"OPENAI_API_KEY": "env-test", "OPENAI_MODEL": "gpt-test"}, clear=True):
            self.assertEqual(openai_config(), ("env-test", "gpt-test", DEFAULT_API_BASE))

    def test_empty_model_uses_default(self):
        with patch.dict("sys.modules", {"config": self.config(OPENAI_MODEL="")}), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(openai_config()[1], "gpt-6.1-terra")

    def test_placeholder_key_is_not_usable(self):
        with patch.dict("sys.modules", {"config": self.config(OPENAI_API_KEY="TU_CLAVE")}), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(openai_config()[0], "")

    def test_credential_is_not_a_model(self):
        with patch.dict("sys.modules", {"config": self.config(OPENAI_MODEL="sk-no-mostrar")}), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as caught:
                openai_config()
        self.assertNotIn("sk-no-mostrar", str(caught.exception))

    def test_only_native_api_allowed(self):
        self.assertEqual(validate_api_base(DEFAULT_API_BASE + "/"), DEFAULT_API_BASE)
        for url in ("https://platform.openai.com/api-keys", "https://api.openai.com.evil.test/v1", "http://api.openai.com/v1",
                    "https://user@api.openai.com/v1", "https://api.openai.com/v1?key=secret", "https://googleapis.com/v1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_api_base(url)


class TransportTests(unittest.TestCase):
    @patch("valor_a_la_vista.vision.requests.post")
    def test_authorization_header_not_query_or_payload(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = {"ok": True}
        self.assertEqual(post_openai("responses", {"input": "test"}, api_key="secret"), {"ok": True})
        args, kwargs = post.call_args
        self.assertNotIn("secret", args[0])
        self.assertNotIn("secret", str(kwargs["json"]))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret")
        self.assertFalse(kwargs["allow_redirects"])

    @patch("valor_a_la_vista.vision.requests.post")
    def test_redirect_never_followed(self, post):
        post.return_value.status_code = 302
        with self.assertRaisesRegex(RuntimeError, "redirigió"):
            post_openai("responses", {}, api_key="secret")
        self.assertEqual(post.call_count, 1)

    @patch("valor_a_la_vista.vision.requests.post")
    def test_untrusted_endpoint_never_gets_key(self, post):
        with self.assertRaises(ValueError):
            post_openai("responses", {}, api_key="secret", base_url="https://evil.test")
        post.assert_not_called()

    @patch("valor_a_la_vista.vision.requests.post")
    def test_missing_key_never_connects(self, post):
        with self.assertRaises(RuntimeError):
            post_openai("responses", {}, api_key="")
        post.assert_not_called()

    @patch("valor_a_la_vista.vision.time.sleep")
    @patch("valor_a_la_vista.vision.requests.post")
    def test_unknown_429_is_not_retried(self, post, sleep):
        response = Mock(status_code=429)
        response.json.return_value = {"error": {"code": "unknown"}}
        post.return_value = response
        with self.assertRaisesRegex(RuntimeError, "429"):
            post_openai("responses", {}, api_key="secret")
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    @patch("valor_a_la_vista.vision.time.sleep")
    @patch("valor_a_la_vista.vision.requests.post")
    def test_quota_429_is_not_retried(self, post, sleep):
        response = Mock(status_code=429)
        response.json.return_value = {"error": {"code": "insufficient_quota"}}
        post.return_value = response
        with self.assertRaisesRegex(RuntimeError, "cuota o facturación") as caught:
            post_openai("responses", {}, api_key="secret")
        self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    @patch("valor_a_la_vista.vision.time.sleep")
    @patch("valor_a_la_vista.vision.requests.post")
    def test_explicit_rate_limit_retries_bounded(self, post, sleep):
        response = Mock(status_code=429, headers={"Retry-After": "1.5"})
        response.json.return_value = {"error": {"code": "rate_limit_exceeded"}}
        post.return_value = response
        with self.assertRaisesRegex(RuntimeError, "temporalmente"):
            post_openai("responses", {}, api_key="secret")
        self.assertEqual(post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_any_call(1.5)

    def test_network_errors_are_sanitized(self):
        for exc in (requests.Timeout("secret"), requests.ConnectionError("secret")):
            with patch("valor_a_la_vista.vision.requests.post", side_effect=exc), self.assertRaises(RuntimeError) as caught:
                post_openai("responses", {}, api_key="secret")
            self.assertNotIn("secret", str(caught.exception))
            self.assertTrue(caught.exception.__suppress_context__)

    @patch("valor_a_la_vista.vision.requests.post")
    def test_invalid_json_sanitized(self, post):
        post.return_value.status_code = 200
        post.return_value.json.side_effect = ValueError("secret")
        with self.assertRaisesRegex(RuntimeError, "JSON válido") as caught:
            post_openai("responses", {}, api_key="secret")
        self.assertNotIn("secret", str(caught.exception))
