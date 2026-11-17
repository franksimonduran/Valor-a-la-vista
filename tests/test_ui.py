"""Pruebas del sistema visual: tokens, contraste y componentes."""
import unittest

from valor_a_la_vista import ui
from valor_a_la_vista.pricing import BudgetItem, build_repair_budget, load_price_table


def _luminance(color: str) -> float:
    value = color.lstrip("#")
    channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(first: str, second: str) -> float:
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


class ThemeTests(unittest.TestCase):
    def test_theme_config_declares_both_modes_and_fonts(self):
        import tomllib
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        with (root / ".streamlit" / "config.toml").open("rb") as handle:
            config = tomllib.load(handle)
        self.assertIn("light", config["theme"])
        self.assertIn("dark", config["theme"])
        self.assertTrue(config["server"]["enableStaticServing"])
        for face in config["theme"].get("fontFaces", []):
            self.assertTrue(face["url"].startswith("app/static/"))
            relative = face["url"].removeprefix("app/static/")
            self.assertTrue((root / "static" / relative).is_file(), face["url"])

    def test_resolve_mode_defaults_to_light(self):
        self.assertEqual(ui.resolve_mode("dark"), "dark")
        self.assertEqual(ui.resolve_mode(None), "light")
        self.assertEqual(ui.resolve_mode("cualquier-cosa"), "light")

    def test_styles_inject_mode_variables(self):
        self.assertIn("--vav-bg:#111022", ui.styles("dark"))
        self.assertIn("--vav-bg:#F1EFF9", ui.styles())
        self.assertIn("--vav-bg:#F1EFF9", ui.styles(None))

    def test_styles_have_no_unresolved_braces(self):
        css = ui.styles("dark")
        self.assertNotIn("{{", css)
        self.assertNotIn("}}", css)

    def test_contrast_meets_wcag_aa(self):
        for mode in ("light", "dark"):
            palette = ui.tokens(mode)
            pairs = [
                (palette["ink"], palette["bg"]),
                (palette["ink"], palette["surface"]),
                (palette["muted"], palette["surface"]),
                (palette["accent"], palette["bg"]),
                (palette["alert"], palette["bg"]),
            ]
            for foreground, background in pairs:
                self.assertGreaterEqual(contrast(foreground, background), 4.5,
                                        f"{mode}: {foreground} sobre {background}")


class MaterialTests(unittest.TestCase):
    def test_shadows_only_use_elevation_tokens(self):
        css = ui.styles("dark")
        lines = [line for line in css.splitlines() if "box-shadow:" in line]
        self.assertTrue(lines)
        for line in lines:
            self.assertIn("var(--vav-elev-", line)

    def test_no_gradients_or_filters(self):
        css = ui.styles("dark")
        for token in ("gradient", "backdrop-filter", "color-mix"):
            self.assertNotIn(token, css)

    def test_translate_only_on_hover(self):
        css = ui.styles("dark")
        lines = [line for line in css.splitlines() if "translateY" in line]
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertIn(":hover", line)

    def test_palette_declares_two_elevations(self):
        for mode in ("light", "dark"):
            palette = ui.tokens(mode)
            self.assertIn("elev_1", palette)
            self.assertIn("elev_2", palette)
            self.assertNotIn("shadow", palette)

    def test_gauge_draws_with_svg_not_gradients(self):
        markup = ui.score_gauge(60, 100)
        self.assertIn("<svg", markup)
        self.assertIn("stroke-dasharray", markup)
        self.assertNotIn("gradient", markup)


class ComponentTests(unittest.TestCase):
    def test_progress_rail_marks_current_step(self):
        markup = ui.progress_rail(3)
        self.assertEqual(markup.count('aria-current="step"'), 1)
        self.assertEqual(markup.count("is-done"), 2)
        self.assertIn("is-active", markup)
        for label in ui.STEPS:
            self.assertIn(label, markup)
        self.assertNotIn("01 Fotos", markup)
        self.assertNotIn("02 Análisis", markup)

    def test_photo_title_is_not_numbered(self):
        self.assertEqual(ui.shot_title(1, "Fachada"), '<p class="vav-shot-title">Fachada</p>')

    def test_section_header_escapes_html(self):
        markup = ui.section_header(3, "<script>alert(1)</script>", "propósito")
        self.assertNotIn("<script>", markup)
        self.assertIn("&lt;script&gt;", markup)
        self.assertNotIn("03", markup)
        self.assertNotIn("<p>", ui.section_header(3, "Solo título"))

    def test_score_gauge_reports_value_and_state(self):
        markup = ui.score_gauge(72, 100)
        self.assertIn(">72<", markup)
        self.assertIn("Score visual", markup)
        self.assertIn('stroke-dasharray="326.73"', markup)
        self.assertIn('stroke-dashoffset="91.48"', markup)

    def test_score_gauge_handles_missing_score(self):
        markup = ui.score_gauge(None, 40, True)
        self.assertIn("Sin datos", markup)
        self.assertIn("is-partial", markup)
        self.assertIn('stroke-dashoffset="326.73"', markup)

    def test_score_gauge_clamps_range(self):
        self.assertIn('stroke-dashoffset="0.00"', ui.score_gauge(140, 100))
        self.assertIn('stroke-dashoffset="326.73"', ui.score_gauge(-5, 100))

    def test_status_pill_states(self):
        self.assertIn("Cargada", ui.status_pill(True))
        self.assertEqual(ui.status_pill(False), "")

    def test_sidebar_meta_is_read_only_text_and_escapes(self):
        markup = ui.sidebar_meta("Modelo OpenAI", "gpt-4.1")
        self.assertIn("vav-sidebar-meta", markup)
        self.assertIn("gpt-4.1", markup)
        self.assertNotIn("<input", markup)
        unsafe = ui.sidebar_meta("<b>Modelo</b>", "<script>alert(1)</script>")
        self.assertNotIn("<script>", unsafe)
        self.assertNotIn("<b>", unsafe)
        self.assertIn("&lt;script&gt;", unsafe)

    def test_masthead_includes_status_and_escapes(self):
        markup = ui.masthead("<script>x</script>", 2, 4, "Sin análisis")
        self.assertNotIn("<script>", markup)
        self.assertIn("2 de 4 tomas", markup)
        self.assertIn("Sin análisis", markup)
        self.assertIn("vav-masthead-status", markup)

    def test_masthead_without_asset_has_no_status(self):
        self.assertNotIn("vav-masthead-status", ui.masthead())

    def test_callout_labels_and_escape(self):
        self.assertIn("Nota", ui.callout("note", "x"))
        self.assertIn("Aviso", ui.callout("warning", "x"))
        self.assertIn("Error", ui.callout("error", "x"))
        self.assertIn("Completado", ui.callout("success", "x"))
        markup = ui.callout("error", "<script>alert(1)</script>")
        self.assertNotIn("<script>", markup)
        self.assertIn("&lt;script&gt;", markup)
        self.assertIn("vav-callout-error", markup)
        self.assertIn("vav-callout-note", ui.callout("desconocido", "x"))

    def test_score_panel_reports_ledger(self):
        markup = ui.score_panel(72, 70, 3, 4, "Estado regular", True)
        self.assertIn("vav-score", markup)
        self.assertIn("70 %", markup)
        self.assertIn("3 / 4", markup)
        self.assertIn("Estado regular", markup)
        self.assertIn("is-partial", markup)

    def test_cost_panel_formats_range_and_location(self):
        from test_pricing import synthetic_rows
        budget = build_repair_budget("terreno", {"superficie": "regular"}, 80, synthetic_rows())
        markup = ui.cost_panel(budget, "Lima")
        self.assertIn("vav-cost", markup)
        self.assertIn("S/ 240.00", markup)
        self.assertIn("S/ 960.00", markup)
        self.assertIn("S/ 600.00", markup)
        self.assertIn("Lima", markup)
        self.assertIn("80 m²", markup)

    def test_cost_panel_marks_partial_without_area(self):
        from test_pricing import synthetic_rows
        budget = build_repair_budget("terreno", {"superficie": "regular"}, 0, synthetic_rows())
        markup = ui.cost_panel(budget, "")
        self.assertIn("total parcial", markup)
        self.assertIn("Sin total calculable", markup)
        self.assertIn('class="vav-cost is-partial"', markup)

    def test_cost_table_lists_source_and_escapes(self):
        item = BudgetItem(
            criterio="superficie", titulo="Superficie", intervencion="<script>Reparar</script>",
            unidad="m2", cantidad=2.0, precio_min=10.0, precio_max=20.0,
            subtotal_min=20.0, subtotal_max=40.0, fuente="Fuente", fecha="2026-11-15",
        )
        markup = ui.cost_table([item])
        self.assertIn("vav-table", markup)
        self.assertNotIn("<script>", markup)
        self.assertIn("&lt;script&gt;", markup)
        self.assertIn("Fuente", markup)
        self.assertIn("2026-11-15", markup)


if __name__ == "__main__":
    unittest.main()
