import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

from valor_a_la_vista.vision import image_quality_warnings, inspect_image, perceptual_hash, prepare_image, parse_model_json, validate_image


def response_body(value):
    return {"status": "completed", "output": [{"type": "message", "status": "completed", "content": [
        {"type": "output_text", "text": json.dumps(value)}]}]}


class VisionTests(unittest.TestCase):
    def test_quality_and_perceptual_hash(self):
        photo = self.make_image()
        self.assertEqual(perceptual_hash(photo), perceptual_hash(photo))
        self.assertTrue(any("nitidez" in warning for warning in image_quality_warnings(photo)))

    def test_json_wrapped_in_text(self):
        result = parse_model_json('Resultado: {"hallazgos": []} fin', {"superficie"})
        self.assertEqual(result["hallazgos"], [])

    def test_no_visible_is_not_evidence_or_material(self):
        value = {"hallazgos": [{"criterio": "superficie", "estado": "regular", "evidencia": "no_visible"}],
                 "materiales_visibles": [{"elemento": "cerco", "material_aparente": "no_visible",
                                           "evidencia": "no_visible"}]}
        result = parse_model_json(json.dumps(value), {"superficie"})
        self.assertEqual(result["hallazgos"], [])
        self.assertEqual(result["materiales_visibles"], [])

    def make_image(self):
        image = Image.new("RGB", (1500, 900), (115, 145, 160))
        out = io.BytesIO()
        image.save(out, "PNG")
        return out.getvalue()

    def test_prepare_jpeg_resizes(self):
        encoded = prepare_image(self.make_image())
        img = Image.open(io.BytesIO(encoded))
        self.assertEqual(img.format, "JPEG")
        self.assertLessEqual(max(img.size), 1280)

    def test_material_parsing(self):
        result = parse_model_json(json.dumps({
            "hallazgos": [], "observacion_general": "Cerco visible",
            "materiales_visibles": [{"elemento": "Cerco", "material_aparente": "Metal",
                                     "evidencia": "Ondas en superficie"}],
        }), {"limites"})
        self.assertEqual(result["materiales_visibles"][0]["material_aparente"], "Metal")

    def test_duplicate_and_missing_criteria_warn(self):
        item = {"criterio": "superficie", "estado": "bueno", "evidencia": "Suelo visible sin daños"}
        result = parse_model_json(json.dumps({"hallazgos": [item, item]}), {"superficie", "limites"})
        self.assertEqual(len(result["hallazgos"]), 1)
        self.assertEqual(len(result["advertencias"]), 2)

    def test_visible_element_without_damage_is_valid(self):
        item = {"criterio": "superficie", "estado": "bueno", "evidencia": "Suelo visible, no se observan daños"}
        result = parse_model_json(json.dumps({"hallazgos": [item]}), {"superficie"})
        self.assertEqual(len(result["hallazgos"]), 1)

    def test_invisible_element_is_not_assessed(self):
        item = {"criterio": "acceso", "estado": "regular", "evidencia": "no se observa acceso visible"}
        result = parse_model_json(json.dumps({"hallazgos": [item]}), {"acceso"})
        self.assertEqual(result["hallazgos"], [])

    @patch("valor_a_la_vista.vision.requests.post")
    def test_openai_request_and_key_header(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = response_body({"hallazgos": [], "danos": [], "materiales_visibles": [], "observacion_general": ""})
        result = inspect_image(self.make_image(), "terreno", "Superficie", model="gpt-test",
                               api_key="clave-prueba")
        self.assertEqual(result["hallazgos"], [])
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.openai.com/v1/responses")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer clave-prueba")
        self.assertFalse(kwargs["allow_redirects"])
        schema = kwargs["json"]["text"]["format"]
        self.assertTrue(schema["strict"])
        self.assertIn("danos", schema["schema"]["required"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertTrue(kwargs["json"]["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertNotIn("temperature", kwargs["json"])
        self.assertNotIn("max_completion_tokens", kwargs["json"])
        self.assertFalse(kwargs["json"]["store"])

    def test_missing_key(self):
        with self.assertRaisesRegex(RuntimeError, "Falta OPENAI_API_KEY"):
            inspect_image(self.make_image(), "terreno", "Superficie", api_key="")

    def test_rejects_corrupt_or_unsupported_images(self):
        image = io.BytesIO()
        Image.new("RGB", (32, 32)).save(image, format="GIF")
        for raw in (b"", b"archivo corrupto", image.getvalue()):
            with self.subTest(size=len(raw)), self.assertRaises(ValueError):
                validate_image(raw)

    def test_negative_damage_evidence_cannot_support_deterioration(self):
        value = {"hallazgos": [{"criterio": "superficie", "estado": "regular", "evidencia": "No hay evidencia de daños visibles"}]}
        result = parse_model_json(json.dumps(value), {"superficie"})
        self.assertEqual(result["hallazgos"], [])
        self.assertTrue(any("contradictorios" in warning for warning in result["advertencias"]))

    def test_non_string_labels_are_discarded(self):
        value = {"hallazgos": [{"criterio": [], "estado": {}, "evidencia": "Dato inválido"}]}
        self.assertEqual(parse_model_json(json.dumps(value), {"superficie"})["hallazgos"], [])

    def test_multiple_lesions_in_one_terrain_are_not_reduced_to_a_status(self):
        common = {"elemento": "Suelo", "ubicacion": "Zona inferior", "severidad": "moderada", "certeza": "visible"}
        value = {"hallazgos": [], "danos": [
            {**common, "tipo": "agua_acumulada", "criterio": "drenaje", "evidencia": "Charcos visibles en la zona central"},
            {**common, "tipo": "residuos", "criterio": "superficie", "evidencia": "Bolsas y residuos visibles sobre el suelo"}]}
        result = parse_model_json(json.dumps(value), {"superficie", "drenaje"}, asset="terreno")
        self.assertEqual(len(result["danos"]), 2)
        self.assertTrue(result["detalle_danos_valido"])

    def test_wrong_asset_damage_and_negative_evidence_are_rejected(self):
        raw = {"tipo": "cable_expuesto", "criterio": "superficie", "elemento": "Suelo", "ubicacion": "Zona inferior",
               "evidencia": "Deterioro visible en la superficie", "severidad": "leve", "certeza": "visible"}
        for changed in ({}, {"tipo": "maleza", "evidencia": "No se observa maleza"},
                        {"tipo": "maleza", "ubicacion": ""}):
            with self.subTest(changed=changed):
                result = parse_model_json(json.dumps({"hallazgos": [], "danos": [{**raw, **changed}]}), {"superficie"}, asset="terreno")
                self.assertEqual(result["danos"], [])
                self.assertFalse(result["detalle_danos_valido"])

    def test_missing_damage_detail_is_not_a_valid_empty_inspection(self):
        result = parse_model_json('{"hallazgos": []}', {"superficie"}, asset="terreno")
        self.assertFalse(result["detalle_danos_valido"])
        self.assertTrue(result["advertencias"])

    def test_unknown_criterion_and_state_discarded(self):
        value = {"hallazgos": [{"criterio": "precio", "estado": "malo", "evidencia": "Valdrá 500 soles"},
                               {"criterio": "superficie", "estado": "excelente", "evidencia": "Suelo visible"}]}
        self.assertEqual(parse_model_json(json.dumps(value), {"superficie"})["hallazgos"], [])

    @patch("valor_a_la_vista.vision.requests.post")
    def test_refusal_or_truncation_does_not_become_inspection(self, post):
        post.return_value.status_code = 200
        for status, content in (("incomplete", [{"type": "output_text", "text": '{"hallazgos": []}'}]),
                                ("completed", [{"type": "refusal", "refusal": "No permitido"}])):
            post.return_value.json.return_value = {"status": status, "output": [{"type": "message", "content": content}]}
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                inspect_image(self.make_image(), "terreno", "Superficie", api_key="test")

    @patch("valor_a_la_vista.vision.requests.post")
    def test_strict_schema_has_no_open_objects(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = response_body({"hallazgos": [], "danos": []})
        inspect_image(self.make_image(), "terreno", "Superficie", api_key="test")
        schema = post.call_args.kwargs["json"]["text"]["format"]["schema"]
        def check(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertFalse(node["additionalProperties"])
                    self.assertEqual(set(node["required"]), set(node["properties"]))
                for child in node.values():
                    check(child)
            elif isinstance(node, list):
                for child in node:
                    check(child)
        check(schema)

    @patch("valor_a_la_vista.vision.requests.post")
    def test_reasoning_text_is_not_parsed_as_findings(self, post):
        post.return_value.status_code = 200
        body = response_body({"hallazgos": [], "danos": []})
        body["output"].insert(0, {"type": "reasoning", "summary": [{"type": "summary_text", "text": "No es una respuesta JSON"}]})
        post.return_value.json.return_value = body
        self.assertEqual(inspect_image(self.make_image(), "terreno", "Superficie", api_key="test")["hallazgos"], [])

    @patch("valor_a_la_vista.vision.requests.post")
    def test_empty_response_fails_closed(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = {"status": "completed", "output": []}
        with self.assertRaises(RuntimeError):
            inspect_image(self.make_image(), "terreno", "Superficie", api_key="test")


if __name__ == "__main__":
    unittest.main()
