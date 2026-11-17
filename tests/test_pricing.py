import tempfile
import unittest
from datetime import date
from pathlib import Path

from valor_a_la_vista.pricing import (PriceRow, build_repair_budget, build_declared_budget,
    build_damage_budget, budget_to_dict, load_price_table, is_official_source_url)
from valor_a_la_vista.core import ASSETS, evaluate
from valor_a_la_vista.report import build_report_data, html_report


def synthetic_rows():
    return [PriceRow("terreno", "superficie", "Limpieza simulada", "m2", 1, 3, 12, "PEN", "Prueba sintética", "2026-11-15"),
            PriceRow("terreno", "acceso", "Revisión simulada", "servicio", 0, 300, 1500, "PEN", "Prueba sintética", "2026-11-15")]


class PricingTests(unittest.TestCase):
    def test_default_catalog_has_no_unverified_prices(self):
        rows, warnings = load_price_table()
        self.assertEqual((rows, warnings), ([], []))

    def test_legacy_area_factor_and_range(self):
        budget = build_repair_budget("terreno", {"superficie": "regular"}, 80, synthetic_rows())
        self.assertTrue(budget.complete)
        self.assertEqual(budget.items[0].cantidad, 80)
        self.assertEqual((budget.total_min, budget.total_max, budget.total_central), (240, 960, 600))

    def test_legacy_service_quantity_is_one(self):
        budget = build_repair_budget("terreno", {"acceso": "malo"}, 0, synthetic_rows())
        self.assertEqual((budget.items[0].cantidad, budget.items[0].unidad), (1, "servicio"))

    def test_legacy_missing_area_has_no_total(self):
        budget = build_repair_budget("terreno", {"superficie": "regular"}, 0, synthetic_rows())
        self.assertFalse(budget.complete)
        self.assertIsNone(budget.total_min)
        self.assertTrue(any("área" in item.motivo for item in budget.missing))

    def test_legacy_missing_row_keeps_known_items(self):
        budget = build_repair_budget("terreno", {"superficie": "regular", "acceso": "malo"}, 80, synthetic_rows()[:1])
        self.assertFalse(budget.complete)
        self.assertEqual(budget.missing[0].criterio, "acceso")
        self.assertGreater(budget.total_max, budget.total_min)

    def test_legacy_no_damage_zero_total(self):
        budget = build_repair_budget("terreno", {"superficie": "bueno"}, 80, synthetic_rows())
        self.assertEqual((budget.damaged, budget.items, budget.total_min), (0, (), 0))

    def test_invalid_rows_skipped_and_valid_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "precios.csv"
            path.write_text("activo,criterio,intervencion,unidad,factor,precio_min,precio_max,moneda,fuente,fecha,nota\n"
                            "terreno,superficie,Limpieza,m2,1,10,20,PEN,,2026-11-15,sin fuente\n"
                            "terreno,acceso,Revisión,servicio,0,10,20,PEN,Sintético,2026-11-15,ok\n", encoding="utf-8")
            rows, warnings = load_price_table(path)
        self.assertEqual([row.criterio for row in rows], ["acceso"])
        self.assertTrue(any("fuente" in warning for warning in warnings))

    def test_bad_header_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "precios.csv"
            path.write_text("activo,criterio\nterreno,superficie\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_price_table(path)

    def test_budget_serialization(self):
        data = budget_to_dict(build_repair_budget("terreno", {"superficie": "regular"}, 80, synthetic_rows()))
        self.assertEqual((data["moneda"], data["area_declarada_m2"]), ("PEN", 80))
        self.assertEqual(data["partidas"][0]["criterio"], "superficie")


class DeclaredBudgetTests(unittest.TestCase):
    def setUp(self):
        self.statuses = {code: "bueno" for code in ASSETS["terreno"]["criterios"]}
        self.statuses["superficie"] = "regular"
        self.quote = {"intervencion": "Limpieza simulada", "unidad": "m2", "cantidad": "2.5", "precio_min": "14.00",
                      "precio_max": "17.00", "fuente": "Prueba sintética; no tarifa real", "fecha": date.today().isoformat(),
                      "nota": "Solo mano de obra", "confirmado": True,
                      "url_fuente": "https://www.gob.pe/documento-sintetico-no-real",
                      "documento": "Ejemplo simulado, página 1; no documento real",
                      "vigencia": "Escenario simulado para Lima", "tipo_referencia": "costo_reparacion"}

    def build(self, quote=None, statuses=None, photos_complete=True):
        return build_declared_budget("terreno", self.statuses if statuses is None else statuses, 999,
                                     {"superficie": self.quote if quote is None else quote}, photos_complete=photos_complete)

    def test_uses_affected_quantity_not_total_area(self):
        budget = self.build()
        self.assertTrue(budget.complete)
        self.assertEqual((budget.total_min, budget.total_max, budget.total_central), (35, 42.5, 38.75))
        self.assertEqual(budget.items[0].cantidad, 2.5)
        self.assertIn("declaración del usuario", budget.items[0].validacion)

    def test_requires_confirmation_source_scope_and_date(self):
        for fields in ({"confirmado": False}, {"fuente": ""}, {"nota": ""}, {"fecha": "2999-01-01"}):
            with self.subTest(fields=fields):
                budget = self.build({**self.quote, **fields})
                self.assertFalse(budget.complete)
                self.assertIsNone(budget.total_min)
                self.assertEqual(len(budget.missing), 1)

    def test_invalid_numbers_and_ranges(self):
        for fields in ({"cantidad": 0}, {"cantidad": "NaN"}, {"cantidad": "Infinity"}, {"precio_min": -1},
                       {"precio_max": 1}, {"cantidad": 1000001}, {"precio_min": "14.001"}, {"cantidad": "1.0001"}):
            with self.subTest(fields=fields):
                self.assertIsNone(self.build({**self.quote, **fields}).total_min)

    def test_decimal_half_up(self):
        self.assertEqual(self.build({**self.quote, "cantidad": "1.005", "precio_min": "1.00", "precio_max": "1.00"}).total_min, 1.01)

    def test_missing_photos_or_criteria_is_partial(self):
        for budget in (self.build(statuses={"superficie": "regular"}), self.build(photos_complete=False)):
            self.assertFalse(budget.complete)
            self.assertEqual(budget.total_min, 35)
            self.assertTrue(budget.limitations)

    def test_all_unverifiable_is_not_zero_cost(self):
        budget = self.build(statuses={})
        self.assertIsNone(budget.total_min)
        self.assertEqual(len(budget.unobserved), 4)

    def test_rejects_nonofficial_or_inapplicable_prices(self):
        for fields in ({"url_fuente": "https://example.com"}, {"documento": ""}, {"vigencia": ""},
                       *[{"tipo_referencia": kind} for kind in ("valor_edificacion", "indice", "arancel", "otro")]):
            with self.subTest(fields=fields):
                self.assertIsNone(self.build({**self.quote, **fields}).total_min)

    def test_official_urls(self):
        for url in ("https://www.gob.pe/", "https://portal.mtc.gob.pe/doc.pdf", "https://www.inei.gob.pe/"):
            self.assertTrue(is_official_source_url(url))
        for url in ("http://www.gob.pe/", "https://www.gob.pe.evil.test/", "https://www.gob.pe@evil.test/",
                    "https://www.gob.pe:8080/", "https://user@www.gob.pe/", "https://www.gob.pe/\n",
                    "https://evil.test\\.gob.pe/", "https://.gob.pe/", None):
            with self.subTest(url=url):
                self.assertFalse(is_official_source_url(url))


class DamageBudgetTests(unittest.TestCase):
    def setUp(self):
        DeclaredBudgetTests.setUp(self)
        self.quote.update({"zona_afectada": "Sector norte", "trabajos": ["limpieza"]})

    def damage(self, ident="a", kind="maleza", criterion="superficie", photo="Superficie", **fields):
        return {"id": ident, "tipo": kind, "criterio": criterion, "foto": photo, "elemento": "Suelo",
                "ubicacion": "Esquina inferior", "evidencia": "Vegetación y residuos visibles sobre el suelo",
                "severidad": "moderada", "certeza": "visible", **fields}

    def build(self, damages=None, quotes=None, statuses=None, detail_complete=True):
        return build_damage_budget("terreno", self.statuses if statuses is None else statuses, 999,
                                   [self.damage()] if damages is None else damages,
                                   {"a": self.quote} if quotes is None else quotes,
                                   photos_complete=True, detail_complete=detail_complete)

    def test_one_item_and_trace(self):
        budget = self.build()
        self.assertTrue(budget.complete)
        self.assertEqual((budget.total_min, budget.items[0].danos_incluidos, budget.items[0].zona_afectada), (35, ("a",), "Sector norte"))

    def test_same_damage_two_photos_shares_quantity(self):
        budget = self.build([self.damage(), self.damage("b", photo="Panorámica")], {"a": self.quote, "b": {"incluido_en": "a"}})
        self.assertTrue(budget.complete)
        self.assertEqual((len(budget.items), budget.total_min, budget.items[0].cantidad), (1, 35, 2.5))
        self.assertEqual(budget.items[0].danos_incluidos, ("a", "b"))

    def test_overlapping_cleaning_maleza_and_residuos_blocked(self):
        damages = [self.damage(), self.damage("b", "residuos")]
        budget = self.build(damages, {"a": self.quote, "b": self.quote})
        self.assertIsNone(budget.total_min)
        self.assertEqual(len(budget.missing), 2)
        self.assertTrue(all("doble cobro" in item.motivo for item in budget.missing))
        self.assertTrue(self.build(damages, {"a": self.quote, "b": {"incluido_en": "a"}}).complete)

    def test_distinct_zones_can_be_priced_separately(self):
        budget = self.build([self.damage(), self.damage("b", photo="Panorámica")],
                            {"a": self.quote, "b": {**self.quote, "zona_afectada": "Sector sur"}})
        self.assertTrue(budget.complete)
        self.assertEqual(budget.total_min, 70)

    def test_unknown_and_circular_groups(self):
        for quotes in ({"a": {"incluido_en": "inexistente"}}, {"a": {"incluido_en": "b"}, "b": {"incluido_en": "a"}}):
            self.assertIsNone(self.build([self.damage(), self.damage("b", photo="Panorámica")], quotes).total_min)

    def test_missing_detail_cannot_be_zero_cost(self):
        self.assertIsNone(self.build(damages=[], quotes={}).total_min)
        self.assertIsNone(self.build(damages=[], quotes={}, statuses={c: "bueno" for c in self.statuses}, detail_complete=False).total_min)

    def test_possible_damage_requires_technical_review(self):
        damage = self.damage(certeza="posible")
        self.assertIsNone(self.build([damage]).total_min)
        reviewed = self.build([damage], {"a": {**self.quote, "revision_tecnica_confirmada": True}})
        self.assertEqual(reviewed.total_min, 35)
        self.assertTrue(reviewed.items[0].revision_tecnica_declarada)

    def test_drainage_erosion_and_leveling_require_review(self):
        for kind, criterion in (("agua_acumulada", "drenaje"), ("erosion_visible", "superficie"), ("desnivel_visible", "superficie")):
            with self.subTest(kind=kind):
                self.assertIsNone(self.build([self.damage(kind=kind, criterion=criterion)]).total_min)

    def test_linear_metres_and_report_escaping(self):
        damage = self.damage(kind="cerco_danado", criterion="limites")
        budget = self.build([damage], {"a": {**self.quote, "unidad": "ml"}}, {**self.statuses, "superficie": "bueno", "limites": "regular"})
        self.assertTrue(budget.complete)
        data = build_report_data("terreno", self.statuses, {}, evaluate("terreno", self.statuses), budget, 999, "Lima", [],
                                 damages=[{**damage, "evidencia": "Cerco <script>x</script>"}])
        html = html_report(data, {})
        self.assertIn("metro lineal", html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("Lesiones específicas", html)

    def test_duplicate_observations_and_ids_rejected(self):
        for damages in ([self.damage(), self.damage()], [self.damage(), self.damage("b")]):
            with self.assertRaises(ValueError):
                self.build(damages)

    def test_more_than_four_photos_supported(self):
        damages = [self.damage(str(i), photo=f"Foto {i}") for i in range(120)]
        budget = self.build(damages, {str(i): self.quote if i == 0 else {"incluido_en": "0"} for i in range(120)})
        self.assertTrue(budget.complete)
        self.assertEqual(budget.total_min, 35)
