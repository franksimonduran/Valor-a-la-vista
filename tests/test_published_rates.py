import unittest
from unittest.mock import patch

from valor_a_la_vista.published_rates import parse_wall_rate, fetch_wall_rate, calculate_wall_labour


class PublishedRateTests(unittest.TestCase):
    def setUp(self):
        self.html = """<p>Solo mano de obra. No se incluye costo de materiales.</p>
        <table><tr><td>Reparación de paredes por metro cuadrado</td>
        <td>s/14.00</td><td>s/17.00</td></tr></table>"""

    def test_exact_row_and_scope(self):
        rate = parse_wall_rate(self.html)
        self.assertEqual((rate["precio_min"], rate["precio_max"]), ("14.00", "17.00"))
        self.assertEqual(rate["unidad"], "m²")
        self.assertIn("no incluye materiales", rate["alcance"])
        self.assertIsNone(rate["fecha_publicacion"])

    def test_rejects_missing_scope_wrong_currency_and_duplicates(self):
        for html in (
            self.html.replace("Solo mano de obra.", ""),
            self.html.replace("s/14.00", "USD 14.00"),
            self.html.replace("</table>", "<tr><td>Reparación de paredes por metro cuadrado</td>"
                              "<td>s/14.00</td><td>s/17.00</td></tr></table>"),
        ):
            with self.subTest(html=html), self.assertRaises(ValueError):
                parse_wall_rate(html)

    def test_no_redirects(self):
        with patch("valor_a_la_vista.published_rates.requests.get") as get:
            get.return_value.__enter__.return_value.status_code = 302
            with self.assertRaises(RuntimeError):
                fetch_wall_rate()
            self.assertFalse(get.call_args.kwargs["allow_redirects"])

    def test_labour_calculation_uses_affected_area(self):
        rate = parse_wall_rate(self.html)
        self.assertEqual(tuple(map(str, calculate_wall_labour(rate, "2.5"))), ("35.00", "42.50"))
        for invalid in (0, -1, "NaN", "Infinity", None):
            with self.subTest(area=invalid), self.assertRaises(ValueError):
                calculate_wall_labour(rate, invalid)


if __name__ == "__main__":
    unittest.main()
