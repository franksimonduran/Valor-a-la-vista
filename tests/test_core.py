import json
import unittest
from unittest.mock import Mock

from valor_a_la_vista.core import (ASSETS, DAMAGE_CATALOG, UNIT_LABELS, WORK_STAGES,
    combine_findings, evaluate, reference_amount, recommend_improvements, validate_damage_findings)
from valor_a_la_vista.pricing import RepairBudget
from valor_a_la_vista.report import build_report_data, html_report, report_digest
from valor_a_la_vista.inspection import analyze_photos, analysis_is_current, analysis_message, photo_fingerprint


class CoreTests(unittest.TestCase):
    def test_only_terrain(self):
        self.assertEqual(set(ASSETS), {"terreno"})
        self.assertEqual(set(DAMAGE_CATALOG), {"terreno"})

    def test_weights_add_up(self):
        self.assertEqual(sum(weight for _, weight, _ in ASSETS["terreno"]["criterios"].values()), 100)

    def test_not_visible_not_scored(self):
        result = evaluate("terreno", {"superficie": "bueno", "acceso": "no_visible"})
        self.assertEqual((result.score, result.coverage), (100, 25))
        self.assertIn("Incompleto", result.category)

    def test_all_unobserved(self):
        self.assertIsNone(evaluate("terreno", {}).score)

    def test_score_ponderado(self):
        states = {code: "regular" for code in ASSETS["terreno"]["criterios"]}
        self.assertEqual((evaluate("terreno", states).score, evaluate("terreno", states).coverage), (60, 100))

    def test_legacy_credit_guards_and_formula(self):
        good = evaluate("terreno", {code: "regular" for code in ASSETS["terreno"]["criterios"]})
        self.assertEqual(reference_amount(good, 4, True, 100000, 20000, 15000)[0], 15000)
        for score, photos, reviewed, value, requested, budget, ratio in (
            (good, 4, False, 100, 100, 100, .4), (good, 2, True, 100, 100, 100, .4),
            (evaluate("terreno", {"acceso": "bueno"}), 4, True, 100, 100, 100, .4),
            (good, 4, True, 100, 100, 100, 0), (good, 4, True, 100, 100, 100, 1.1),
            (good, 4, True, float("nan"), 100, 100, .4), (good, 4, True, 0, 100, 100, .4),
            (evaluate("terreno", {c: "malo" for c in ASSETS["terreno"]["criterios"]}), 4, True, 100, 100, 100, .4)):
            with self.subTest(score=score, photos=photos, ratio=ratio):
                self.assertIsNone(reference_amount(score, photos, reviewed, value, requested, budget, ratio)[0])

    def test_actions_only_when_damaged(self):
        actions = recommend_improvements("terreno", {"acceso": "regular", "superficie": "bueno", "drenaje": "malo"})
        self.assertEqual([a["prioridad"] for a in actions], ["Media", "Alta"])

    def test_conservative_combination(self):
        states, evidence = combine_findings("terreno", [
            [{"criterio": "superficie", "estado": "bueno", "evidencia": "Superficie uniforme"}],
            [{"criterio": "superficie", "estado": "malo", "evidencia": "Residuos visibles"}]])
        self.assertEqual((states["superficie"], evidence["superficie"]), ("malo", "Residuos visibles"))

    def test_automatic_proposal_and_digest(self):
        result = evaluate("terreno", {"superficie": "malo"})
        data = build_report_data("terreno", {"superficie": "malo"}, {}, result, RepairBudget(), 0, "Lima", [],
                                 proposal={"superficie": {"estado": "malo"}})
        self.assertEqual(data["propuesta_ia"]["superficie"]["estado"], "malo")
        self.assertFalse(data["revision_humana"])
        self.assertIn("sin revisión humana", data["etiqueta"])
        digest = report_digest(data)
        data["sha256_contenido"] = digest
        self.assertEqual(digest, report_digest(data))
        data["ubicacion_declarada"] = "Cusco"
        self.assertNotEqual(digest, report_digest(data))

    def test_report_includes_materials_and_escapes_evidence(self):
        result = evaluate("terreno", {"superficie": "bueno"})
        data = build_report_data("terreno", {"superficie": "bueno"}, {"superficie": "<script>x</script>"},
                                 result, RepairBudget(), 0, "", [], materials=[{"elemento": "Cerco", "material_aparente": "Metal"}])
        html = html_report(data, {})
        self.assertIn("Metal", html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


class InspectionTests(unittest.TestCase):
    def response(self, state="bueno", damage=False):
        return {"hallazgos": [{"criterio": "superficie", "estado": state, "evidencia": "Vegetación visible sobre el suelo"}],
                "materiales_visibles": [], "observacion_general": "Prueba", "advertencias": [],
                "danos": [{"tipo": "maleza", "criterio": "superficie", "elemento": "Suelo", "ubicacion": "Centro",
                           "evidencia": "Vegetación densa visible en el suelo", "severidad": "moderada", "certeza": "visible",
                           "area": 100, "precio": 50}] if damage else []}

    def analyze(self, photos=None, **kwargs):
        return analyze_photos("terreno", photos or {"Superficie": b"foto"}, api_key="test", model="gpt-test", **kwargs)

    def test_unchanged_photos_are_reused_and_dates_preserved(self):
        cache, inspector = {}, Mock(return_value=self.response())
        first = self.analyze(cache=cache, inspector=inspector)
        second = self.analyze(cache=cache, inspector=inspector)
        self.assertEqual(inspector.call_count, 1)
        self.assertEqual(first["metadata"]["evidencias_por_foto"], second["metadata"]["evidencias_por_foto"])
        self.assertTrue(analysis_is_current(second, photo_fingerprint({"Superficie": b"foto"}), "gpt-test"))
        self.assertFalse(analysis_is_current(second, first["signature"], "otro"))
        self.assertFalse(analysis_is_current(second, photo_fingerprint({"Superficie": b"otra"}), "gpt-test"))
        self.assertFalse(analysis_is_current(second, first["signature"], "gpt-test", "https://other.test"))

    def test_force_and_changed_photo_trigger_new_calls(self):
        cache, inspector = {}, Mock(return_value=self.response())
        for photos, force in (({"Superficie": b"foto"}, False), ({"Superficie": b"foto"}, True), ({"Superficie": b"otra"}, False)):
            self.analyze(photos, cache=cache, force=force, inspector=inspector)
        self.assertEqual(inspector.call_count, 3)

    def test_category_changes_invalidate_cache(self):
        cache, inspector = {}, Mock(return_value=self.response())
        first = self.analyze(cache=cache, inspector=inspector, photo_categories={"Superficie": "Panorámica"})
        second = self.analyze(cache=cache, inspector=inspector, photo_categories={"Superficie": "Superficie"})
        self.assertNotEqual(first["signature"], second["signature"])
        self.assertEqual(inspector.call_count, 2)

    def test_failures_are_retried_without_losing_successful_photos(self):
        cache = {}
        inspector = Mock(side_effect=[self.response(), RuntimeError("Sin conexión"), self.response()])
        first = self.analyze({"Superficie": b"uno", "Acceso": b"dos"}, cache=cache, inspector=inspector)
        second = self.analyze({"Superficie": b"uno", "Acceso": b"dos"}, cache=cache, inspector=inspector)
        self.assertEqual(len(first["fallos"]), 1)
        self.assertEqual(len(second["analizadas"]), 2)
        self.assertEqual(inspector.call_count, 3)

    def test_unverifiable_is_not_complete(self):
        analysis = self.analyze(inspector=Mock(return_value=self.response("no_visible")))
        kind, message = analysis_message(analysis, evaluate("terreno", analysis["statuses"]), 4)
        self.assertEqual(kind, "warning")
        self.assertIn("ningún criterio", message)

    def test_conflicting_findings_keep_per_photo_trace(self):
        analysis = self.analyze({"Superficie": b"uno", "Acceso": b"dos"},
                                inspector=Mock(side_effect=[self.response(), self.response("malo")]))
        self.assertEqual(analysis["statuses"]["superficie"], "malo")
        self.assertEqual(len(analysis["metadata"]["evidencias_por_foto"]), 2)
        self.assertTrue(any("Discrepancias" in text for text in analysis["advertencias"]))

    def test_visible_damage_overrides_good_summary_without_measures(self):
        analysis = self.analyze(inspector=Mock(return_value=self.response(damage=True)))
        self.assertEqual(analysis["statuses"]["superficie"], "regular")
        self.assertTrue(analysis["detalle_completo"])
        self.assertEqual(analysis["danos"][0]["foto"], "Superficie")
        self.assertNotIn("area", analysis["danos"][0])
        self.assertNotIn("precio", analysis["danos"][0])
        self.assertTrue(analysis["danos"][0]["id"].startswith("d_"))

    def test_repeated_views_keep_distinct_trace_without_summing_areas(self):
        analysis = self.analyze({"Superficie": b"uno", "Acceso": b"dos"}, inspector=Mock(return_value=self.response(damage=True)))
        self.assertEqual(len({d["id"] for d in analysis["danos"]}), 2)
        self.assertTrue(any("misma zona física" in text for text in analysis["advertencias"]))

    def test_generic_deterioration_without_specific_damage_is_incomplete(self):
        analysis = self.analyze(inspector=Mock(return_value=self.response("regular")))
        self.assertFalse(analysis["detalle_completo"])
        self.assertTrue(any("sin lesión concreta" in text for text in analysis["advertencias"]))

    def test_extra_photos_do_not_replace_missing_guided_views(self):
        photos = {f"Foto {i}": str(i).encode() for i in range(8)}
        analysis = self.analyze(photos, inspector=Mock(return_value=self.response()),
                                photo_categories={name: "Superficie" for name in photos})
        self.assertEqual(analysis["categorias_cubiertas"], ["Superficie"])
        self.assertFalse(analysis["fotos_completas"])

    def test_guided_coverage_with_extra_photos(self):
        photos = {name: name.encode() for name, _ in ASSETS["terreno"]["fotos"]}
        photos["Detalle"] = b"extra"
        analysis = self.analyze(photos, inspector=Mock(return_value=self.response()))
        self.assertTrue(analysis["fotos_completas"])
        self.assertEqual(len(analysis["analizadas"]), 5)

    def test_limits_and_duplicates_before_inspector(self):
        inspector = Mock()
        for photos in ({str(i): str(i).encode() for i in range(17)}, {"a": b"x", "b": b"x"}):
            with self.assertRaises(ValueError):
                self.analyze(photos, inspector=inspector)
        inspector.assert_not_called()


class DamageCatalogTests(unittest.TestCase):
    def test_catalog_criteria_units_and_acceptance(self):
        self.assertIn("otro_visible", DAMAGE_CATALOG["terreno"])
        for kind, spec in DAMAGE_CATALOG["terreno"].items():
            with self.subTest(kind=kind):
                self.assertTrue(set(spec.criterios) <= set(ASSETS["terreno"]["criterios"]))
                self.assertIn(spec.unidad, UNIT_LABELS)
                self.assertTrue(set(spec.trabajos) <= set(WORK_STAGES))
                clean, warnings, valid = validate_damage_findings("terreno", [{"tipo": kind, "criterio": spec.criterios[0],
                    "elemento": "Suelo", "ubicacion": "Centro", "evidencia": "Deterioro concreto visible en esta zona",
                    "severidad": "moderada", "certeza": "visible"}])
                self.assertTrue(valid)
                self.assertFalse(warnings)
                self.assertEqual(clean[0]["tipo"], kind)

    def test_uncertain_and_high_severity_require_prior_review(self):
        raw = {"tipo": "maleza", "criterio": "superficie", "elemento": "Suelo", "ubicacion": "Centro",
               "evidencia": "Vegetación visible sobre el suelo", "severidad": "leve", "certeza": "visible"}
        for changed in ({"certeza": "posible"}, {"severidad": "alta"}, {"severidad": "no_determinable"}):
            clean, _, valid = validate_damage_findings("terreno", [{**raw, **changed}])
            self.assertTrue(valid)
            self.assertTrue(clean[0]["requiere_revision"])
