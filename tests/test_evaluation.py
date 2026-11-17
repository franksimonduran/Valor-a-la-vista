import unittest

from valor_a_la_vista.evaluation import calculate_metrics


class EvaluationMetricsTests(unittest.TestCase):
    def test_metrics_include_unverifiable_and_exclude_errors(self):
        records = [
            {"asset": "terreno", "criterio": "superficie", "referencia": "bueno", "prediccion": "bueno"},
            {"asset": "terreno", "criterio": "superficie", "referencia": "malo", "prediccion": "no_visible"},
            {"asset": "terreno", "criterio": "superficie", "referencia": "malo", "prediccion": "malo", "error": "timeout"},
        ]
        result = calculate_metrics(records)["terreno/superficie"]
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["matriz"]["malo"]["no_visible"], 1)
        self.assertEqual(result["clases"]["malo"]["recall"], 0)
        self.assertEqual(result["acuerdo"], .5)

    def test_undefined_kappa(self):
        result = calculate_metrics([{"asset": "terreno", "criterio": "superficie",
                                     "referencia": "bueno", "prediccion": "bueno"}])
        self.assertIsNone(result["terreno/superficie"]["kappa"])
