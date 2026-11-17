import json
import unittest
from unittest.mock import patch

from valor_a_la_vista.online_prices import search_repair_references


class SearchTests(unittest.TestCase):
    def candidate(self, **fields):
        return {"dano_id": "d1", "intervencion": "Limpieza simulada", "unidad": "m2", "precio_min": 14, "precio_max": 17,
                "fuente": "Ejemplo sintético", "url_fuente": "https://www.gob.pe/documento-sintetico-no-real",
                "documento": "Prueba; página 1", "fecha": "2026-11-15", "vigencia": "Simulado Lima", "nota": "Solo mano de obra", **fields}

    def body(self, candidates=None, citations=True):
        rows = [self.candidate()] if candidates is None else candidates
        return {"status": "completed", "output": [{"type": "web_search_call"}, {"type": "message", "content": [{
            "type": "output_text", "text": json.dumps({"resumen": "Simulado", "candidatos": rows}),
            "annotations": [{"type": "url_citation", "url": row["url_fuente"], "title": "Simulado"} for row in rows] if citations else []}]}]}

    def search(self):
        return search_repair_references("terreno", {"superficie": "regular"}, {}, "Lima", api_key="test", model="gpt-test",
                                        damages=[{"id": "d1", "tipo": "maleza", "evidencia": "Vegetación visible"}])

    def test_search_sources_attributed_never_confirmed(self):
        with patch("valor_a_la_vista.online_prices.post_openai", return_value=self.body()) as post:
            result = self.search()
        self.assertEqual(result["fuentes"][0]["url"], self.candidate()["url_fuente"])
        self.assertFalse(result["candidatos"][0]["confirmado"])
        payload = post.call_args.args[1]
        self.assertEqual(payload["tools"][0]["type"], "web_search")
        self.assertIn("maleza", payload["input"])
        self.assertNotIn("image", payload)
        self.assertFalse(payload["store"])

    def test_commercial_sources_discarded(self):
        with patch("valor_a_la_vista.online_prices.post_openai", return_value=self.body([self.candidate(url_fuente="https://example.com")])):
            result = self.search()
        self.assertEqual(result["candidatos"], [])
        self.assertEqual(result["fuentes"], [])

    def test_missing_citations_cannot_supply_prices(self):
        with patch("valor_a_la_vista.online_prices.post_openai", return_value=self.body(citations=False)):
            self.assertEqual(self.search()["candidatos"], [])

    def test_discards_unsupported_candidate_keeps_cited_official(self):
        body = self.body([self.candidate(), self.candidate(url_fuente="https://example.com")])
        with patch("valor_a_la_vista.online_prices.post_openai", return_value=body):
            result = self.search()
        self.assertEqual(len(result["candidatos"]), 1)
        self.assertNotIn("example.com", json.dumps(result))

    def test_actual_web_search_is_required(self):
        body = self.body()
        body["output"].pop(0)
        with patch("valor_a_la_vista.online_prices.post_openai", return_value=body), self.assertRaises(RuntimeError):
            self.search()

    def test_incomplete_response_rejected(self):
        body = self.body()
        body["status"] = "incomplete"
        with patch("valor_a_la_vista.online_prices.post_openai", return_value=body), self.assertRaises(RuntimeError):
            self.search()

    def test_invalid_candidate_prices_and_metadata_discarded(self):
        for fields in ({"precio_min": -1}, {"precio_min": True}, {"precio_min": float("nan")}, {"precio_min": 14.001},
                       {"precio_max": 1}, {"unidad": "hectareas"}, {"fecha": "2999-01-01"}, {"documento": ""},
                       {"dano_id": "desconocido"}):
            with self.subTest(fields=fields), patch("valor_a_la_vista.online_prices.post_openai", return_value=self.body([self.candidate(**fields)])):
                self.assertEqual(self.search()["candidatos"], [])

    def test_no_damage_does_not_search(self):
        with patch("valor_a_la_vista.online_prices.post_openai") as post:
            result = search_repair_references("terreno", {"superficie": "bueno"}, {}, "", api_key="test", model="gpt-test")
            self.assertEqual(result["candidatos"], [])
            post.assert_not_called()
