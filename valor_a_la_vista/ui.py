"""Elementos visuales compartidos por la aplicación y el informe.

El CSS se mantiene como texto plano y los colores se inyectan como variables
CSS. Así se evita duplicar llaves en f-strings y el modo claro/oscuro se
resuelve con un solo conjunto de reglas.

Dirección visual: panel premium luminoso con acentos violeta y azul.
"""
import math
from html import escape
from .core import UNIT_LABELS, WORK_STAGES

# Tokens de referencia usados por el informe HTML (siempre en modo claro).
INK = "#0F2B33"
MUTED = "#5A6B70"
ACCENT = "#0B6E66"
BORDER = "#E2E7E4"
SURFACE = "#FFFFFF"
BACKGROUND = "#F4F6F5"

LIGHT = {
    "bg": "#F1EFF9",
    "surface": "#FFFFFF",
    "raised": "#F4F2FC",
    "line": "#E3DFF1",
    "ink": "#211B38",
    "muted": "#625B78",
    "accent": "#5931CB",
    "cinema": "#3156B9",
    "alert": "#8A5E12",
    "danger": "#B03A32",
    "ok": "#1E704C",
    "elev_1": "0 1px 2px rgba(15,43,51,.05), 0 8px 24px rgba(15,43,51,.05)",
    "elev_2": "0 2px 4px rgba(15,43,51,.06), 0 18px 40px rgba(15,43,51,.10)",
}

DARK = {
    "bg": "#111022",
    "surface": "#1C1A31",
    "raised": "#282440",
    "line": "#40395C",
    "ink": "#F6F3FF",
    "muted": "#C1B9D7",
    "accent": "#C3A9FF",
    "cinema": "#9AB6FF",
    "alert": "#FFC66D",
    "danger": "#FF918B",
    "ok": "#5DE4B8",
    "elev_1": "0 1px 2px rgba(0,0,0,.45), 0 10px 28px rgba(0,0,0,.32)",
    "elev_2": "0 2px 6px rgba(0,0,0,.50), 0 22px 56px rgba(0,0,0,.42)",
}

STEPS = ("Fotos", "Análisis", "Resultado", "Informe")
CALLOUTS = {"note": "Nota", "warning": "Aviso", "error": "Error", "success": "Completado"}

GAUGE_SIZE = 112
GAUGE_STROKE = 8
GAUGE_RADIUS = (GAUGE_SIZE - GAUGE_STROKE) / 2
GAUGE_LENGTH = 2 * math.pi * GAUGE_RADIUS


def resolve_mode(value) -> str:
    """Normaliza el tema del cliente; cualquier valor desconocido es claro."""
    return "dark" if value == "dark" else "light"


def tokens(mode="light") -> dict[str, str]:
    return DARK if resolve_mode(mode) == "dark" else LIGHT


_CSS = """<style>
    :root { color-scheme: light dark; }
    .stApp, [data-testid="stAppViewContainer"] { background: var(--vav-bg); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stDecoration"], [data-testid="stStatusWidget"], footer { display: none; }
    [data-testid="stSidebar"] { border-right: 1px solid #6744B7; }
    [data-testid="stSidebar"] .stMarkdown p { color: #EEE8FF; }
    [data-testid="stSidebar"] .vav-callout { --vav-callout: #FFD875; }
    [data-testid="stSidebar"] .vav-callout-label { color: #FFD875; }
    h1, h2, h3, h4 { font-family: "Sora", "Segoe UI Variable Display", "Segoe UI", sans-serif; letter-spacing: -.01em; }

    .vav-sidebar-brand { border-bottom: 1px solid #9479D0; margin-bottom: 18px; padding-bottom: 14px; }
    .vav-sidebar-brand .vav-monogram { color: #FFFFFF; font-family: Georgia, "Iowan Old Style", "Times New Roman", serif; font-size: 1.8rem; font-weight: 600; letter-spacing: -.03em; line-height: 1; }
    .vav-sidebar-brand .vav-accent { color: #FFD875; }
    .vav-sidebar-brand .vav-wordmark { color: #EEE8FF; font-family: "Sora", "Segoe UI", sans-serif; font-size: .72rem; letter-spacing: .18em; margin: 6px 0 0; text-transform: uppercase; }
    .vav-sidebar-label { color: var(--vav-muted); font-size: .72rem; letter-spacing: .12em; margin: 18px 0 6px; text-transform: uppercase; }
    .vav-sidebar-meta { color: #FFFFFF; font-size: .86rem; margin: 0 0 10px; overflow-wrap: anywhere; }
    .vav-sidebar-meta span { color: #E1D6FF; display: block; font-size: .72rem; letter-spacing: .08em; margin-bottom: 2px; text-transform: uppercase; }
    .vav-sidebar-foot { border-top: 1px solid #9479D0; color: #EEE8FF; font-size: .72rem; margin-top: 22px; padding-top: 12px; }

    .vav-masthead { background: #4332A6; border: 1px solid #6555C8; border-radius: 22px; box-shadow: var(--vav-elev-1); display: grid; gap: 28px; grid-template-columns: minmax(0, 1fr) auto; margin-bottom: 18px; overflow: hidden; padding: 34px 38px; position: relative; }
    .vav-masthead::before { background: #6455D0; border-radius: 50%; content: ""; height: 260px; position: absolute; right: -60px; top: -130px; width: 260px; }
    .vav-masthead::after { border: 1px solid #9388E8; border-radius: 50%; content: ""; height: 170px; opacity: .65; position: absolute; right: 110px; top: -110px; width: 170px; }
    .vav-masthead-brand { position: relative; z-index: 1; }
    .vav-masthead .vav-monogram { color: #FFFFFF; font-family: Georgia, "Iowan Old Style", "Times New Roman", serif; font-size: 2rem; font-weight: 600; letter-spacing: -.03em; line-height: 1; }
    .vav-masthead .vav-accent { color: #87E9E0; }
    .vav-masthead .vav-wordmark { color: #FFFFFF; font-family: "Sora", "Segoe UI", sans-serif; font-size: .78rem; font-weight: 600; letter-spacing: .12em; margin: 8px 0 0; }
    .vav-masthead .vav-masthead-rule { background: #8F83DB; height: 1px; margin: 14px 0; width: 56px; }
    .vav-masthead .vav-eyebrow { color: #E4E0FF; font-size: .72rem; font-weight: 600; letter-spacing: .12em; margin: 0; text-transform: uppercase; }
    .vav-masthead .vav-tagline { color: #FFFFFF; font-family: "Sora", sans-serif; font-size: clamp(1.4rem, 2.6vw, 2.2rem); font-weight: 600; letter-spacing: -.03em; line-height: 1.2; margin: 10px 0 0; max-width: 22ch; }
    .vav-masthead-status { align-self: end; background: #302380; border: 1px solid #897EE0; border-radius: 14px; padding: 16px 20px; position: relative; text-align: left; z-index: 1; }
    .vav-status-label { color: #E4E0FF; font-size: .72rem; letter-spacing: .1em; margin: 0 0 6px; text-transform: uppercase; }
    .vav-status-value { color: #FFFFFF; font-family: "Sora", "Segoe UI", sans-serif; font-size: .92rem; font-weight: 600; margin: 0; }
    .vav-status-meta { color: #E4E0FF; font-size: .82rem; margin: 6px 0 0; }

    .vav-steps { background: var(--vav-surface); border: 1px solid var(--vav-line); border-radius: 16px; display: flex; flex-wrap: wrap; gap: 6px; list-style: none; margin: 0 0 24px; padding: 8px; }
    .vav-steps li { align-items: center; border-radius: 10px; color: var(--vav-muted); display: flex; font-size: .76rem; font-weight: 600; gap: 8px; padding: 9px 13px; }
    .vav-steps li.is-done { color: var(--vav-ink); }
    .vav-steps li.is-active { background: var(--vav-raised); color: var(--vav-accent); }
    .vav-dot { background: var(--vav-line); border-radius: 50%; height: 6px; width: 6px; }
    .vav-steps li.is-done .vav-dot { background: var(--vav-muted); }
    .vav-steps li.is-active .vav-dot { background: var(--vav-accent); }

    .vav-section { margin-top: 28px; padding: 12px 0 4px; }
    .vav-section .vav-step { color: var(--vav-cinema); font-family: "Sora", "Segoe UI", sans-serif; font-size: .82rem; padding-top: 3px; }
    .vav-section h2 { color: var(--vav-ink); font-family: "Sora", "Segoe UI Variable Display", "Segoe UI", sans-serif; font-size: 1.5rem; letter-spacing: -.01em; margin: 0; }
    .vav-section p { color: var(--vav-muted); margin: 6px 0 0; }

    .vav-chip { align-items: center; color: var(--vav-muted); display: inline-flex; font-size: .72rem; font-weight: 600; gap: 6px; }
    .vav-chip::before { background: currentColor; border-radius: 50%; content: ""; height: 6px; width: 6px; }
    .vav-chip-ready { color: var(--vav-ok); }
    .vav-chip-pending { color: var(--vav-muted); }

    .vav-shot-title { color: var(--vav-ink); font-family: "Sora", "Segoe UI", sans-serif; font-size: 1rem; font-weight: 600; margin: 0; }
    .vav-shot-number { color: var(--vav-cinema); font-size: 1.3rem; font-variant-numeric: tabular-nums; margin-right: 10px; }

    .vav-gauge { height: 112px; position: relative; width: 112px; }
    .vav-gauge-ring { display: block; height: 100%; transform: rotate(-90deg); width: 100%; }
    .vav-gauge-ring circle { fill: none; stroke-width: 8; }
    .vav-gauge-track { stroke: var(--vav-line); }
    .vav-gauge-arc { stroke: var(--vav-accent); stroke-linecap: round; transition: stroke-dashoffset .4s ease; }
    .vav-gauge-center { align-content: center; display: grid; inset: 0; position: absolute; text-align: center; }
    .vav-gauge-value { color: var(--vav-ink); font-size: 1.75rem; font-variant-numeric: tabular-nums; font-weight: 600; line-height: 1; }
    .vav-gauge-unit { color: var(--vav-muted); display: block; font-size: .72rem; margin-top: 4px; }
    .vav-gauge-label { color: var(--vav-muted); font-size: .72rem; letter-spacing: .12em; margin: 10px 0 0; text-transform: uppercase; }
    .vav-gauge-label.is-partial { color: var(--vav-alert); }

    .vav-score { align-items: center; background: var(--vav-surface); border: 1px solid var(--vav-line); border-radius: 14px; box-shadow: var(--vav-elev-2); display: grid; gap: 30px; grid-template-columns: auto minmax(0, 1fr); margin: 18px 0 10px; padding: 24px 28px; }
    .vav-score-gauge { display: grid; justify-items: center; }
    .vav-score-body { display: grid; gap: 12px; }
    .vav-score-eyebrow { color: var(--vav-muted); font-size: .72rem; letter-spacing: .12em; margin: 0; text-transform: uppercase; }
    .vav-score-category { color: var(--vav-ink); font-family: "Sora", "Segoe UI", sans-serif; font-size: 1.35rem; font-weight: 600; letter-spacing: -.01em; margin: 0; }
    .vav-ledger { display: grid; gap: 10px; grid-template-columns: repeat(2, minmax(0, 1fr)); margin: 0; }
    .vav-ledger div { background: var(--vav-raised); border-radius: 12px; padding: 14px; }
    .vav-ledger dt { color: var(--vav-muted); font-size: .72rem; letter-spacing: .08em; margin: 0; text-transform: uppercase; }
    .vav-ledger dd { color: var(--vav-ink); font-size: 1.35rem; font-variant-numeric: tabular-nums; font-weight: 600; margin: 4px 0 0; }

    .vav-cost { background: var(--vav-surface); border: 1px solid var(--vav-line); border-left: 4px solid var(--vav-accent); border-radius: 20px; box-shadow: var(--vav-elev-1); display: grid; gap: 26px; grid-template-columns: minmax(0, 1.3fr) minmax(260px, 1fr); margin: 14px 0; overflow-wrap: anywhere; padding: 28px 32px; }
    .vav-cost-main { align-content: center; display: grid; gap: 12px; }
    .vav-cost.is-partial { border-left-color: var(--vav-alert); }
    .vav-cost-label { color: var(--vav-muted); font-size: .72rem; letter-spacing: .12em; margin: 0; text-transform: uppercase; }
    .vav-cost-label.is-partial { color: var(--vav-alert); }
    .vav-cost-value { color: var(--vav-ink); font-family: "Sora", "Segoe UI", sans-serif; font-size: clamp(1.7rem, 3vw, 2.7rem); font-variant-numeric: tabular-nums; font-weight: 600; letter-spacing: -.035em; margin: 6px 0 0; }
    .vav-cost-mid { color: var(--vav-muted); font-size: .9rem; margin: 0; }
    .vav-cost-note { color: var(--vav-muted); font-size: .8rem; margin: 0; }
    .vav-table-wrap { border: 1px solid var(--vav-line); border-radius: 12px; margin: 14px 0; overflow-x: auto; }
    .vav-table { border-collapse: collapse; font-size: .86rem; min-width: 680px; width: 100%; }
    .vav-table th, .vav-table td { border-bottom: 1px solid var(--vav-line); padding: 10px 12px; text-align: left; vertical-align: top; }
    .vav-table th { background: var(--vav-raised); color: var(--vav-muted); font-size: .68rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
    .vav-table tr:last-child td { border-bottom: 0; }
    .vav-table td { color: var(--vav-ink); font-variant-numeric: tabular-nums; }
    .vav-table small { color: var(--vav-muted); display: block; font-size: .74rem; margin-top: 4px; }

    .vav-callout { border-left: 2px solid var(--vav-callout, var(--vav-accent)); margin: 12px 0; padding: 2px 0 2px 14px; }
    .vav-callout-label { color: var(--vav-callout, var(--vav-accent)); display: block; font-size: .72rem; font-weight: 600; letter-spacing: .12em; margin-bottom: 4px; text-transform: uppercase; }
    .vav-callout p { color: var(--vav-ink); margin: 0; }
    .vav-callout-note { --vav-callout: var(--vav-accent); }
    .vav-callout-warning { --vav-callout: var(--vav-alert); }
    .vav-callout-error { --vav-callout: var(--vav-danger); }
    .vav-callout-success { --vav-callout: var(--vav-ok); }

    .vav-rule { border: 0; border-top: 1px solid var(--vav-line); margin: 22px 0; }

    [class*="st-key-shot_"] { background: var(--vav-surface); border: 1px solid var(--vav-line) !important; border-radius: 18px !important; box-shadow: var(--vav-elev-1); padding: 20px; transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease; }
    [class*="st-key-shot_"]:hover { border-color: var(--vav-accent) !important; box-shadow: var(--vav-elev-2); transform: translateY(-1px); }
    .st-key-declared_details { background: var(--vav-surface); border: 1px solid var(--vav-line); border-radius: 18px; box-shadow: var(--vav-elev-1); padding: 22px; }
    .vav-details-title { color: var(--vav-ink); font-family: "Sora", sans-serif; font-size: 1rem; font-weight: 600; margin: 0; }
    .vav-details-sub { color: var(--vav-muted); font-size: .8rem; margin: 4px 0 10px; }

    .stButton button, .stDownloadButton button, .stFormSubmitButton button { border-radius: 10px; font-weight: 600; min-height: 44px; transition: border-color .16s ease, box-shadow .16s ease, transform .16s ease; }
    .stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover { border-color: var(--vav-accent); }
    .st-key-analyze button { box-shadow: var(--vav-elev-1); }
    .st-key-analyze button:hover { box-shadow: var(--vav-elev-2); transform: translateY(-1px); }
    .stButton button:focus-visible, .stDownloadButton button:focus-visible, .stFormSubmitButton button:focus-visible { outline: 2px solid var(--vav-accent); outline-offset: 2px; }

    [data-testid="stFileUploaderDropzone"] { background: var(--vav-raised); border: 1px dashed var(--vav-line); border-radius: 10px; }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--vav-accent); }
    [data-testid="stMetric"] { background: var(--vav-surface); border: 1px solid var(--vav-line); border-radius: 12px; padding: 14px 16px; }
    [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    [data-testid="stProgress"] [role="progressbar"] { border-radius: 999px; overflow: hidden; }
    [data-testid="stProgress"] [role="progressbar"] > div { border-radius: 999px; }
    [data-testid="stExpander"] details { background: transparent; border: 0; border-top: 1px solid var(--vav-line); border-radius: 0; }
    [data-testid="stExpander"] summary { color: var(--vav-ink); font-weight: 600; }
    .stTextInput input, .stNumberInput input, .stTextArea textarea { background: var(--vav-surface); }

    @media (max-width: 760px) {
        .vav-masthead { gap: 18px; grid-template-columns: 1fr; padding: 24px; }
        .vav-masthead .vav-monogram { font-size: 2.4rem; }
        .vav-masthead .vav-wordmark { font-size: .86rem; letter-spacing: .12em; }
        .vav-masthead-status { padding: 14px 16px; text-align: left; }
        .vav-section { gap: 12px; grid-template-columns: 28px 1fr; }
        .vav-score { grid-template-columns: 1fr; padding: 22px; }
        .vav-score-gauge { justify-items: start; }
        .vav-ledger { grid-template-columns: 1fr; }
        .vav-cost-value { font-size: 2rem; }
        .vav-cost { grid-template-columns: 1fr; padding: 20px; }
        .vav-steps { gap: 14px; }
        .vav-steps li { font-size: .64rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation: none !important; transition: none !important; }
    }
    </style>"""


def styles(mode="light"):
    palette = tokens(mode)
    declarations = ";".join(f"--vav-{key.replace('_', '-')}:{value}" for key, value in palette.items())
    return f"<style>:root{{{declarations}}}</style>" + _CSS


def sidebar_brand():
    return """<div class="vav-sidebar-brand">
    <div class="vav-monogram" aria-hidden="true">V<span class="vav-accent">A</span>V</div>
    <p class="vav-wordmark">Valor A la Vista</p>
    </div>"""


def sidebar_label(text):
    return f'<p class="vav-sidebar-label">{escape(str(text))}</p>'


def sidebar_meta(label, value):
    """Dato fijo de la barra lateral. Se muestra, no se edita."""
    return (f'<p class="vav-sidebar-meta"><span>{escape(str(label))}</span>'
            f'{escape(str(value))}</p>')


def sidebar_footer():
    return '<p class="vav-sidebar-foot">Prototipo académico · Sin valor contractual</p>'


def masthead(asset_name="", shots=0, total=0, state=""):
    status = ""
    if asset_name:
        meta = f"{shots} de {total} tomas"
        if state:
            meta += f" · {state}"
        status = ('<div class="vav-masthead-status">'
                  '<p class="vav-status-label">Inspección</p>'
                  f'<p class="vav-status-value">{escape(str(asset_name))}</p>'
                  f'<p class="vav-status-meta">{escape(meta)}</p>'
                  '</div>')
    return f"""<header class="vav-masthead">
    <div class="vav-masthead-brand">
    <div class="vav-monogram" aria-hidden="true">V<span class="vav-accent">A</span>V</div>
    <h1 class="vav-wordmark">Valor A la Vista</h1>
    <div class="vav-masthead-rule" aria-hidden="true"></div>
    <p class="vav-eyebrow">Inspección visual</p>
    <p class="vav-tagline">Lo visible, convertido en claridad.</p>
    </div>{status}
    </header>"""


def progress_rail(step):
    items = []
    for index, label in enumerate(STEPS, start=1):
        if index < step:
            state = " is-done"
        elif index == step:
            state = " is-active"
        else:
            state = ""
        current = ' aria-current="step"' if index == step else ""
        items.append(f'<li class="vav-step-item{state}"{current}><span class="vav-dot" aria-hidden="true"></span>{escape(label)}</li>')
    return f'<ol class="vav-steps" aria-label="Progreso de la inspección">{"".join(items)}</ol>'


def section_header(number, title, purpose=""):
    note = f"<p>{escape(str(purpose))}</p>" if purpose else ""
    return (f'<section class="vav-section">'
            f'<div><h2>{escape(str(title))}</h2>{note}</div></section>')


def details_heading(asset_name):
    return ('<div class="vav-details-heading">'
            '<p class="vav-details-title">Datos de la inspección</p>'
            f'<p class="vav-details-sub">{escape(str(asset_name))} · Datos declarados</p>'
            '</div>')


def shot_title(number, name):
    return f'<p class="vav-shot-title">{escape(str(name))}</p>'


def status_pill(loaded):
    """Solo confirma una toma cargada; el estado pendiente no necesita etiqueta."""
    if not loaded:
        return ""
    return '<span class="vav-chip vav-chip-ready">Cargada</span>'


def callout(kind, text):
    """Aviso editorial: filete de 2 px, etiqueta y texto. Sin fondo tintado."""
    kind = kind if kind in CALLOUTS else "note"
    return (f'<div class="vav-callout vav-callout-{kind}">'
            f'<span class="vav-callout-label">{CALLOUTS[kind]}</span>'
            f'<p>{escape(str(text))}</p></div>')


def rule():
    return '<hr class="vav-rule">'


def score_gauge(score, coverage=0, partial=False):
    """Medidor circular sin gradientes; la cifra real queda como texto accesible."""
    if score is None:
        value = 0
        label = "Sin datos"
    else:
        value = max(0.0, min(100.0, float(score)))
        label = f"{value:g}"
    state = "parcial" if partial else "visual"
    caption_class = "vav-gauge-label is-partial" if partial else "vav-gauge-label"
    caption = f"Score {state}"
    accessible = f"Score {state}: {label}" + (" de 100" if score is not None else "")
    unit = '<span class="vav-gauge-unit">/100</span>' if score is not None else ""
    offset = GAUGE_LENGTH * (1 - value / 100)
    return (f'<div class="vav-gauge" role="img" aria-label="{escape(accessible)}">'
            f'<svg class="vav-gauge-ring" viewBox="0 0 {GAUGE_SIZE} {GAUGE_SIZE}" aria-hidden="true" focusable="false">'
            f'<circle class="vav-gauge-track" cx="{GAUGE_SIZE / 2}" cy="{GAUGE_SIZE / 2}" r="{GAUGE_RADIUS:g}"></circle>'
            f'<circle class="vav-gauge-arc" cx="{GAUGE_SIZE / 2}" cy="{GAUGE_SIZE / 2}" r="{GAUGE_RADIUS:g}" '
            f'stroke-dasharray="{GAUGE_LENGTH:.2f}" stroke-dashoffset="{offset:.2f}"></circle>'
            f'</svg>'
            f'<div class="vav-gauge-center"><span class="vav-gauge-value">{escape(label)}</span>{unit}</div>'
            f'</div><p class="{caption_class}">{escape(caption)}</p>')


def score_panel(score, coverage, verified, total, category, partial=False):
    """Panel de resultado: medidor, clasificación y ledger de cobertura."""
    return ('<section class="vav-score">'
            f'<div class="vav-score-gauge">{score_gauge(score, coverage, partial)}</div>'
            '<div class="vav-score-body">'
            '<p class="vav-score-eyebrow">Lectura preliminar</p>'
            f'<p class="vav-score-category">{escape(str(category))}</p>'
            '<dl class="vav-ledger">'
            f'<div><dt>Cobertura ponderada</dt><dd>{coverage:.0f} %</dd></div>'
            f'<div><dt>Criterios evaluados</dt><dd>{verified} / {total}</dd></div>'
            '</dl></div></section>')


def _soles(value):
    return f"S/ {value:,.2f}"


def cost_panel(budget, location=""):
    """Resultado principal: rango de reparación, nunca valor del activo."""
    if budget.total_min is None:
        value = "Sin total calculable"
        mid = "Faltan datos observables o precios para formar un total."
    elif budget.damaged == 0:
        value = _soles(0)
        mid = "Sin partidas propuestas para lo observado; no descarta daños ocultos."
    elif budget.total_min == budget.total_max:
        value = _soles(budget.total_min)
        mid = f"Valor central {_soles(budget.total_central)}."
    else:
        value = f"{_soles(budget.total_min)} – {_soles(budget.total_max)}"
        mid = f"Valor central {_soles(budget.total_central)}."
    label = "Costo aproximado de mejora · total parcial" if budget.partial else (
        "Costo aproximado de mejora · rango")
    label_class = "vav-cost-label is-partial" if budget.partial else "vav-cost-label"
    area = f"{budget.area:g} m² declarados" if budget.area else "Área no indicada"
    place = escape(str(location).strip()) if location else "Ubicación no indicada"
    panel_class = "vav-cost is-partial" if budget.partial else "vav-cost"
    return (f'<section class="{panel_class}">'
            '<div class="vav-cost-main">'
            f'<p class="{label_class}">{escape(label)}</p>'
            f'<p class="vav-cost-value">{escape(value)}</p>'
            f'<p class="vav-cost-mid">{escape(mid)}</p>'
            '<p class="vav-cost-note">Rango orientativo; cotizaciones y cantidades no verificadas por el sistema.</p>'
            '</div>'
            '<dl class="vav-ledger">'
            f'<div><dt>Partidas con precio</dt><dd>{len(budget.items)}</dd></div>'
            f'<div><dt>Sin precio referencial</dt><dd>{len(budget.missing)}</dd></div>'
            f'<div><dt>Área</dt><dd>{escape(area)}</dd></div>'
            f'<div><dt>Ubicación</dt><dd>{place}</dd></div>'
            '</dl>'
            '</section>')


def cost_table(items):
    """Tabla de partidas con fuente y fecha visibles."""
    if not items:
        return ""
    rows = []
    for item in items:
        unit = UNIT_LABELS.get(item.unidad, item.unidad)
        quantity = f"{item.cantidad:g} {unit}"
        unit_price = (_soles(item.precio_min) if item.precio_min == item.precio_max
                       else f"{_soles(item.precio_min)} – {_soles(item.precio_max)}")
        unit_price += f" por {unit}"
        subtotal = (_soles(item.subtotal_min) if item.subtotal_min == item.subtotal_max
                    else f"{_soles(item.subtotal_min)} – {_soles(item.subtotal_max)}")
        note = f"<small>{escape(item.nota)}</small>" if item.nota else ""
        note += f"<small>{escape(item.validacion)}</small>"
        if item.danos_incluidos:
            note += f"<small>Zona: {escape(item.zona_afectada)}. Daños: {escape(', '.join(item.danos_incluidos))}.</small>"
            note += "<small>" + escape(", ".join(WORK_STAGES.get(stage, stage) for stage in item.trabajos)) + "</small>"
        source_details = escape(item.fuente) + f"<small>{escape(item.fecha)}</small>"
        for detail in (item.documento, item.vigencia, item.url_fuente):
            if detail:
                source_details += f"<small>{escape(detail)}</small>"
        rows.append(
            f'<tr><td>{escape(item.intervencion)}{note}</td><td>{escape(quantity)}</td>'
            f'<td>{escape(unit_price)}</td><td>{escape(subtotal)}</td>'
            f'<td>{source_details}</td></tr>')
    return ('<div class="vav-table-wrap"><table class="vav-table">'
            '<thead><tr><th>Intervención</th><th>Cantidad</th><th>Precio unitario</th>'
            '<th>Subtotal</th><th>Fuente y fecha</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')
