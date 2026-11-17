"""Interacción de terrenos con imágenes sintéticas; toda red está bloqueada."""
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from valor_a_la_vista.core import ASSETS
from valor_a_la_vista.gallery import add_photos
from valor_a_la_vista.vision import DEFAULT_API_BASE


class AppTests(unittest.TestCase):
    def setUp(self):
        for target, kwargs in (("requests.post", {"side_effect": AssertionError("Red bloqueada")}),
                               ("requests.get", {"side_effect": AssertionError("Red bloqueada")}),
                               ("valor_a_la_vista.vision.openai_config", {"return_value": ("test", "gpt-test", DEFAULT_API_BASE)})):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def app(self):
        return AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30).run()

    def seeded_app(self, count=4):
        app = self.app()
        uploads = []
        for i in range(count):
            out = io.BytesIO()
            Image.new("RGB", (640, 480), (i * 35, 100, 150)).save(out, format="PNG")
            uploads.append((f"foto-{i}.png", out.getvalue()))
        gallery, _ = add_photos({}, uploads)
        for i, item in enumerate(gallery.values()):
            item["categoria"] = ASSETS["terreno"]["fotos"][i][0] if i < 4 else "Detalle adicional"
        app.session_state["gallery_terreno"] = gallery
        return app.run()

    def inspect_photo(self, raw, asset, shot_name, **kwargs):
        self.assertEqual(asset, "terreno")
        return {"hallazgos": [{"criterio": code, "estado": "bueno", "evidencia": "Elemento visible en esta toma"}
                               for code in ASSETS[asset]["criterios"]],
                "danos": [{"tipo": "maleza", "criterio": "superficie", "elemento": "Suelo",
                           "ubicacion": "Zona central", "evidencia": "Vegetación densa visible en la zona central",
                           "severidad": "moderada", "certeza": "visible"}] if shot_name == "Superficie" else [],
                "materiales_visibles": [], "advertencias": [], "observacion_general": "Ejemplo sintético"}

    def test_only_terrain_starts_without_network(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(set(ASSETS), {"terreno"})
        self.assertTrue(app.button(key="analyze").disabled)
        self.assertFalse(any(item.key == "asset" for item in app.radio))

    def test_reset_clears_terrain_not_unrelated_state(self):
        app = self.seeded_app()
        app.number_input(key="area_terreno").set_value(80).run()
        app.text_input(key="location_terreno").set_value("Lima").run()
        app.session_state["unrelated"] = {"conservar": True}
        next(b for b in app.button if b.label == "Reiniciar terreno").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.number_input(key="area_terreno").value, 0)
        self.assertEqual(app.text_input(key="location_terreno").value, "")
        self.assertEqual(app.session_state["gallery_terreno"], {})
        self.assertEqual(app.session_state["unrelated"], {"conservar": True})
        self.assertTrue(app.button(key="analyze").disabled)

    def test_five_images_can_be_removed_and_gallery_persists(self):
        app = self.seeded_app(5)
        self.assertFalse(app.exception)
        app.text_input(key="location_terreno").set_value("Cusco").run()
        self.assertEqual(len(app.session_state["gallery_terreno"]), 5)
        next(b for b in app.button if b.label == "Quitar fotografía").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["gallery_terreno"]), 4)

    def test_uploads_require_explicit_add_and_do_not_analyze(self):
        out = io.BytesIO()
        Image.new("RGB", (640, 480), "green").save(out, format="PNG")
        out.name = "nueva.png"
        with patch("streamlit.file_uploader", return_value=[out]), patch("valor_a_la_vista.inspection.inspect_image") as inspector:
            app = self.app()
            self.assertEqual(app.session_state["gallery_terreno"], {})
            next(b for b in app.button if b.label == "Añadir imágenes al terreno").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state["gallery_terreno"]), 1)
            inspector.assert_not_called()
            next(b for b in app.button if b.label == "Añadir imágenes al terreno").click().run()
            self.assertEqual(len(app.session_state["gallery_terreno"]), 1)

    def test_single_photo_with_no_guided_views_can_be_analyzed(self):
        def inspect_first(raw, asset, shot_name, **kwargs):
            return self.inspect_photo(raw, asset, "Superficie", **kwargs)
        with patch("valor_a_la_vista.inspection.inspect_image", side_effect=inspect_first) as inspector:
            app = self.seeded_app(1)
            next(item for item in app.selectbox if item.key.startswith("category_terreno_")).set_value("Detalle adicional").run()
            self.assertFalse(app.button(key="analyze").disabled)
            app.button(key="analyze").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(inspector.call_count, 1)
            self.assertEqual(len(app.session_state["analysis_terreno"]["analizadas"]), 1)
            self.assertFalse(app.session_state["analysis_terreno"]["fotos_completas"])
            data = app.session_state["report_data_terreno"]["data"]
            self.assertEqual(data["danos_detectados"][0]["tipo"], "maleza")
            self.assertTrue(data["mejoras_sugeridas"])
            self.assertFalse(data["costo_reparacion"]["completo"])

    def test_video_preparation_is_local_and_analysis_explicit(self):
        out = io.BytesIO()
        Image.new("RGB", (640, 480), "brown").save(out, "PNG")
        origin = {"tipo": "video", "video_nombre": "terreno.mp4", "video_sha256": "a" * 64,
                  "tiempo_s": 2, "tiempo_aproximado": "00:02.000", "indice_frame": 60, "fps": 30,
                  "duracion_s": 3, "audio_analizado": False}
        frame = {"nombre": "Video · 00:02.000.jpg", "raw": out.getvalue(), "origen": origin}
        video = io.BytesIO(b"video-sintetico")
        video.name = "terreno.mp4"
        def upload(*args, **kwargs):
            return video if kwargs.get("key", "").startswith("video_") else []
        with patch("streamlit.file_uploader", side_effect=upload), \
             patch("valor_a_la_vista.video.extract_video_frames", return_value=([frame], ["Muestreo simulado"])) as extract, \
             patch("valor_a_la_vista.inspection.inspect_image", side_effect=self.inspect_photo) as inspector:
            app = self.app()
            app.radio(key="source_terreno").set_value("Video").run()
            extract.assert_not_called()
            app.button(key="prepare_video").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(len(app.session_state["gallery_terreno"]), 1)
            inspector.assert_not_called()
            next(item for item in app.selectbox if item.key.startswith("category_terreno_")).set_value("Superficie").run()
            app.button(key="analyze").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(inspector.call_count, 1)
            data = app.session_state["report_data_terreno"]["data"]
            self.assertEqual(data["danos_detectados"][0]["origen"]["tiempo_s"], 2)
            self.assertIsNone(data["costo_reparacion"]["total_min_soles"])

    def test_photo_budget_and_report_flow_without_real_api(self):
        with patch("valor_a_la_vista.inspection.inspect_image", side_effect=self.inspect_photo) as inspector:
            app = self.seeded_app(5)
            app.text_input(key="location_terreno").set_value("Lima").run()
            app.button(key="analyze").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(inspector.call_count, 5)
            app.button(key="analyze").click().run()
            self.assertEqual(inspector.call_count, 5)
            self.assertEqual(len(app.session_state["analysis_terreno"]["danos"]), 1)
            for suffix, value in (("_zone", "Sector central"), ("_source", "Prueba sintética"),
                                  ("_url", "https://www.gob.pe/documento-sintetico-no-real"),
                                  ("_document", "Documento simulado, partida 1"), ("_validity", "Escenario simulado Lima"),
                                  ("_scope", "Solo mano de obra")):
                next(item for item in app.text_input if item.key.endswith(suffix)).set_value(value)
            for suffix, value in (("_quantity", 2.5), ("_low", 14.), ("_high", 17.)):
                next(item for item in app.number_input if item.key.endswith(suffix)).set_value(value)
            next(item for item in app.selectbox if item.key.endswith("_type")).set_value("costo_reparacion")
            next(item for item in app.checkbox if item.key.endswith("_confirmed")).set_value(True)
            next(b for b in app.button if b.label == "Calcular presupuesto").click().run()
            self.assertFalse(app.exception)
            data = app.session_state["report_data_terreno"]["data"]
            self.assertEqual((data["costo_reparacion"]["total_min_soles"], data["costo_reparacion"]["total_max_soles"]), (35, 42.5))
            self.assertTrue(data["costo_reparacion"]["completo"])
            self.assertEqual(data["danos_detectados"][0]["tipo"], "maleza")
            self.assertEqual(data["costo_reparacion"]["partidas"][0]["danos_incluidos"], (data["danos_detectados"][0]["id"],))
            app.run()
            self.assertEqual(app.session_state["report_data_terreno"]["data"]["sha256_contenido"], data["sha256_contenido"])
            app.text_input(key="location_terreno").set_value("Cusco").run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state["report_data_terreno"]["data"]["costo_reparacion"]["total_min_soles"])

    def test_damages_without_prices_reach_report(self):
        with patch("valor_a_la_vista.inspection.inspect_image", side_effect=self.inspect_photo):
            app = self.seeded_app()
            app.button(key="analyze").click().run()
            self.assertFalse(app.exception)
            data = app.session_state["report_data_terreno"]["data"]
            self.assertEqual(data["danos_detectados"][0]["tipo"], "maleza")
            self.assertIsNone(data["costo_reparacion"]["total_min_soles"])

    def test_removing_photo_invalidates_visible_analysis(self):
        with patch("valor_a_la_vista.inspection.inspect_image", side_effect=self.inspect_photo):
            app = self.seeded_app()
            app.button(key="analyze").click().run()
            next(b for b in app.button if b.label == "Quitar fotografía").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.dataframe)

    def test_bad_api_configuration_is_visible_without_network(self):
        with patch("valor_a_la_vista.vision.openai_config", side_effect=ValueError("API_BASE_URL inválida")):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)

    def test_search_prefill_is_explicit_unconfirmed_and_preserves_edits(self):
        def search(asset, statuses, evidence, location, **kwargs):
            return {"texto": "Candidato simulado", "consultado": "2026-11-15", "fuentes": [], "candidatos": [{
                "dano_id": kwargs["damages"][0]["id"], "intervencion": "Limpieza simulada", "unidad": "m2",
                "precio_min": 14, "precio_max": 17, "fuente": "Entidad sintética", "url_fuente": "https://www.gob.pe/documento-sintetico-no-real",
                "documento": "Simulado, partida 1", "fecha": "2026-11-15", "vigencia": "Simulado Lima", "nota": "Solo mano de obra"}]}

        with patch("valor_a_la_vista.inspection.inspect_image", side_effect=self.inspect_photo), \
             patch("valor_a_la_vista.online_prices.search_repair_references", side_effect=search) as lookup:
            app = self.seeded_app()
            app.text_input(key="location_terreno").set_value("Lima").run()
            app.button(key="analyze").click().run()
            lookup.assert_not_called()
            app.button(key="search_prices").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(lookup.call_count, 1)
            self.assertEqual(next(x for x in app.number_input if x.key.endswith("_low")).value, 0)
            next(b for b in app.button if b.label == "Preparar referencias candidatas para revisión").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(next(x for x in app.number_input if x.key.endswith("_low")).value, 14)
            self.assertFalse(next(x for x in app.checkbox if x.key.endswith("_confirmed")).value)
            self.assertEqual(next(x for x in app.number_input if x.key.endswith("_quantity")).value, 0)
            self.assertIsNone(app.session_state["report_data_terreno"]["data"]["costo_reparacion"]["total_min_soles"])
            next(x for x in app.number_input if x.key.endswith("_low")).set_value(15)
            next(b for b in app.button if b.label == "Calcular presupuesto").click().run()
            next(b for b in app.button if b.label == "Preparar referencias candidatas para revisión").click().run()
            self.assertEqual(next(x for x in app.number_input if x.key.endswith("_low")).value, 15)
            self.assertEqual(lookup.call_count, 1)

    def test_local_yolo_requires_ack_and_works_without_openai_key(self):
        card = {"weights_sha256": "b"*64, "completed_epochs": 3, "model_id": "synthetic_test",
                "status": "experimental", "class_names": {"0": "residuos"},
                "test": {"mAP50": .5}, "test_fixed_threshold": {"precision": .7, "recall": .5}}
        region = {"tipo": "residuos", "etiqueta": "Residuos visibles", "confianza_detector": .8,
                  "caja_normalizada_xyxy": [.1, .1, .5, .5]}
        with patch("valor_a_la_vista.detector.detector_card", return_value=card), \
             patch("valor_a_la_vista.detector.detector_status", return_value=(True, "Modelo sintético de prueba")), \
             patch("valor_a_la_vista.detector.load_terrain_detector", return_value=(object(), card["weights_sha256"])) as load, \
             patch("valor_a_la_vista.detector.detect_regions", return_value=[region]), \
             patch("valor_a_la_vista.vision.openai_config", return_value=("", "gpt-test", DEFAULT_API_BASE)), \
             patch("valor_a_la_vista.inspection.inspect_image") as inspect:
            app = self.seeded_app(1)
            self.assertFalse(app.exception)
            self.assertTrue(app.button(key="analyze").disabled)
            self.assertTrue(app.button(key="detect_yolo").disabled)
            next(item for item in app.checkbox if item.key.startswith("yolo_ack_")).check().run()
            self.assertFalse(app.button(key="detect_yolo").disabled)
            app.button(key="detect_yolo").click().run()
            self.assertFalse(app.exception)
            inspect.assert_not_called()
            self.assertEqual(load.call_count, 1)
            result = app.session_state["yolo_terreno"]
            self.assertEqual(result["resultados"][0]["regiones"][0]["tipo"], "residuos")
            self.assertFalse(result["validado"])
            card["weights_sha256"] = "c"*64
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(app.button(key="detect_yolo").disabled)
            self.assertFalse(app.dataframe, "No deben mostrarse detecciones del modelo anterior")
