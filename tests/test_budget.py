import unittest

from valor_a_la_vista.budget import calculate_budget


class LegacyBudgetTests(unittest.TestCase):
    """El presupuesto manual queda como módulo legado, fuera del flujo."""

    def setUp(self):
        self.item = {"descripcion": "Reparación", "unidad": "m²", "cantidad": "2.5",
                     "precio_unitario": "10.10", "fuente": "Cotización", "fecha": "2026-11-15"}

    def test_complete_and_rounding(self):
        result = calculate_budget([self.item])
        self.assertTrue(result["completo"])
        self.assertEqual(result["total"], 25.25)

    def test_incomplete_is_not_total(self):
        item = {**self.item, "cantidad": "", "fuente": ""}
        result = calculate_budget([item])
        self.assertFalse(result["completo"])
        self.assertIsNone(result["total"])
        self.assertIsNone(result["partidas"][0]["subtotal"])
        self.assertIsNone(calculate_budget([])["total"])

    def test_invalid_values(self):
        for value in ("-1", "0", "NaN", "Infinity", "xyz"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                calculate_budget([{**self.item, "cantidad": value}])
        with self.assertRaises(ValueError):
            calculate_budget([{**self.item, "fecha": "2026-02-30"}])


if __name__ == "__main__":
    unittest.main()
