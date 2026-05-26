#!/usr/bin/env python3
"""
build.py — render a poster from content.yaml.

Usage:
    python build.py content.example.yaml [-o poster.html]

The build is deterministic: given the same content.yaml, it always
produces byte-identical poster.html.

Dependencies:
    - PyYAML (`pip install pyyaml`). No Jinja2 needed.

Design:
    - Inlines styles.css and scale.js into template.html so the output
      is a single self-contained HTML file.
    - Each section "kind" (see content.schema.yaml) is rendered by a
      pure function. Add new kinds by extending RENDER_SECTION.
"""
from __future__ import annotations

import argparse
import html as _html
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML required. pip install pyyaml")

ROOT = Path(__file__).parent
TEMPLATE = (ROOT / "template.html").read_text()
STYLES = (ROOT / "styles.css").read_text()
SCALE_JS = (ROOT / "scale.js").read_text()


def E(s):
    return _html.escape(str(s), quote=False) if s is not None else ""


# ---------------------------------------------------------------------
# Component renderers
# ---------------------------------------------------------------------

def render_header(meta: dict) -> str:
    host = ""
    if meta.get("venue_host_html"):
        host = f"<br/>{meta['venue_host_html']}"
    return f"""<header class="head">
    <div class="lockup">
        <span class="kicker"><span class="dot"></span>{meta['kicker']}</span>
        <h1 class="title">{meta['title_html']}</h1>
        <p class="deck">{meta['deck']}</p>
        <div class="authors">{meta['authors_html']}
            <span class="aff">{meta['affiliation_html']}</span>
        </div>
    </div>
    <div class="meta">
        <span class="v-name">{meta['venue_name_html']}</span>
        {meta['venue_date']}<br/>
        {meta['venue_location_html']}{host}
    </div>
</header>
<div class="rule"></div>"""


def render_hero_stats(stats: list) -> str:
    items = "\n".join(
        f"""    <div class="stat">
        <div class="n">{s['number']}<span class="u">{s['unit']}</span></div>
        <div class="l">{s['label_html']}</div>
    </div>"""
        for s in stats
    )
    n = len(stats)
    style = "" if n == 4 else f' style="grid-template-columns: repeat({n}, 1fr);"'
    return f'<div class="hero-stats"{style}>\n{items}\n</div>'


def render_problem(p: dict | None) -> str:
    if not p:
        return ""
    body = "\n        ".join(f"<p>{para}</p>" for para in p["body_html"])
    pull_quote = p["pull"]["quote_html"]
    pull_cite = p["pull"].get("cite_html", "")
    return f"""<section class="problem">
    <div class="copy">
        <h2 class="section-h"><span>{p['heading']}</span><span class="ix">{p['index']}</span></h2>
        {body}
    </div>
    <div class="pull">
        <div class="quote">{pull_quote}</div>
        <cite>{pull_cite}</cite>
    </div>
</section>"""


# ---- section kinds ----

def _section_head(s: dict) -> str:
    return (
        f'<h2 class="section-h"><span>{s["heading"]}</span>'
        f'<span class="ix">{s["index"]}</span></h2>'
    )


def render_card(s):
    parts = [_section_head(s)]
    if s.get("lede_html"):
        parts.append(f'<p class="lede">{s["lede_html"]}</p>')
    for para in s.get("body_html", []):
        parts.append(f"<p>{para}</p>")
    return '<section class="card">\n  ' + "\n  ".join(parts) + "\n</section>"


def render_framework(s):
    parts = [_section_head(s)]
    if s.get("lede_html"):
        parts.append(f'<p class="lede">{s["lede_html"]}</p>')
    layer_html = []
    layers = s["layers"]
    for i, L in enumerate(layers):
        cls = "layer foundation" if L.get("foundation") else "layer"
        layer_html.append(
            f'<div class="{cls}"><div class="ix">{L["id"]}</div>'
            f'<div><h4>{L["title"]}</h4><p>{L["body_html"]}</p></div></div>'
        )
        if i < len(layers) - 1:
            layer_html.append('<div class="arrow">&darr;</div>')
    parts.append('<div class="framework">\n' + "\n".join(layer_html) + "\n</div>")
    return '<section class="card">\n  ' + "\n  ".join(parts) + "\n</section>"


def render_svg_bars(chart: dict) -> str:
    """Deterministic SVG horizontal bar chart."""
    x_max = chart.get("x_max", 1.0)
    rows = chart["rows"]
    weight_color = {
        "perfect": "var(--accent-deep)",
        "strong": "var(--accent)",
        "med": "var(--accent)",
        "weak": "var(--accent)",
    }
    weight_opacity = {"perfect": 1, "strong": 1, "med": 0.70, "weak": 0.50}
    label_color = {
        "perfect": "var(--ink)",
        "strong": "var(--ink)",
        "med": "var(--body)",
        "weak": "var(--muted)",
    }
    label_x_end = 200
    bar_x = 210
    bar_x_end = 580
    bar_w = bar_x_end - bar_x
    y0 = 44
    dy = 21
    bar_h = 14

    parts = []
    # gridlines
    for frac in (0.25, 0.5, 0.75):
        x = bar_x + bar_w * frac
        parts.append(
            f'<line x1="{x:.1f}" y1="{y0-6}" x2="{x:.1f}" '
            f'y2="{y0 + dy*len(rows)}" stroke="var(--rule)" stroke-width="1" stroke-dasharray="2 4"/>'
        )
    # mean line
    if chart.get("mean") is not None:
        mx = bar_x + bar_w * (chart["mean"] / x_max)
        parts.append(
            f'<line x1="{mx:.1f}" y1="{y0-10}" x2="{mx:.1f}" '
            f'y2="{y0 + dy*len(rows)}" stroke="var(--accent-ink)" stroke-width="1.4" stroke-dasharray="4 4"/>'
        )
        parts.append(
            f'<text x="{mx:.1f}" y="{y0-14}" fill="var(--accent-ink)" '
            f'font-family="JetBrains Mono" font-size="11" text-anchor="middle">'
            f'mean {chart["mean"]:.3f}</text>'
        )
    # rows
    for i, r in enumerate(rows):
        y = y0 + dy * i
        w = bar_w * (r["value"] / x_max)
        color = weight_color.get(r.get("weight", "strong"), "var(--accent)")
        op = weight_opacity.get(r.get("weight", "strong"), 1)
        lcol = label_color.get(r.get("weight", "strong"), "var(--body)")
        parts.append(
            f'<text x="{label_x_end-4}" y="{y+bar_h-3}" text-anchor="end" '
            f'fill="{lcol}" font-family="Inter" font-size="13">{E(r["label"])}</text>'
        )
        parts.append(
            f'<rect x="{bar_x}" y="{y}" width="{w:.1f}" height="{bar_h}" '
            f'rx="1.5" fill="{color}" fill-opacity="{op}"/>'
        )
        parts.append(
            f'<text x="{bar_x+w+6:.1f}" y="{y+bar_h-3}" '
            f'fill="var(--accent-ink)" font-family="JetBrains Mono" font-size="12">'
            f'{r["value"]:.2f}</text>'
        )
    # axis label
    parts.append(
        f'<text x="{(bar_x+bar_x_end)/2}" y="{y0 + dy*len(rows) + 18}" '
        f'fill="var(--muted)" font-family="Inter" font-size="11" '
        f'text-anchor="middle" font-style="italic">{E(chart.get("x_axis_label",""))}</text>'
    )

    svg = (
        '<svg viewBox="0 0 600 360" preserveAspectRatio="xMidYMid meet" role="img">\n'
        f'  <title>{E(chart.get("title",""))}</title>\n  '
        + "\n  ".join(parts)
        + "\n</svg>"
    )
    cap = chart.get("caption_html", "")
    return (
        '<div class="f1-plot">\n' + svg +
        (f'\n<p class="f1-plot-cap">{cap}</p>' if cap else "") +
        "\n</div>"
    )


def render_results(s):
    parts = [_section_head(s)]
    metrics = "\n".join(
        f'<div class="m"><div class="mn">{m["value"]}</div>'
        f'<div class="ml">{m["label_html"]}</div></div>'
        for m in s["metrics"]
    )
    chart = ""
    if s.get("chart"):
        if s["chart"].get("kind") == "svg_bars":
            chart = render_svg_bars(s["chart"])
    parts.append(
        f"""<div class="results-card">
  <div class="results-head"><h3>{s['title']}</h3><span class="tag">{s['tag']}</span></div>
  <div class="results-metrics">{metrics}</div>
  <div class="results-list">{s.get('narrative_html','')}{chart}</div>
</div>"""
    )
    return '<section class="card">\n  ' + "\n  ".join(parts) + "\n</section>"


def render_vignette(s):
    style = ""
    if s.get("accent_variant") == "deep":
        style = ' style="border-left-color: var(--accent-deep);"'
    parts = [f'<div class="vignette"{style}>']
    if s.get("label_html"):
        parts.append(f'  <div class="vh">{s["label_html"]}</div>')
    if s.get("ask_html"):
        parts.append(f'  <div class="ask">{s["ask_html"]}</div>')
    if s.get("ans_html"):
        no_top = "" if s.get("ask_html") else ' style="border-top:none;padding-top:0;"'
        parts.append(f'  <div class="ans"{no_top}>{s["ans_html"]}</div>')
    parts.append("</div>")
    return "\n".join(parts)


def render_subway(s):
    parts = []
    sts = s["stations"]
    for i, st in enumerate(sts):
        parts.append(
            f'<div class="station"><div class="pip"></div><div class="lbl">{E(st)}</div></div>'
        )
        if i < len(sts) - 1:
            parts.append('<div class="seg"></div>')
    return '<div class="subway" aria-hidden="true">\n  ' + "\n  ".join(parts) + "\n</div>"


def render_micro_grid(s):
    cols = s.get("cols", 3)
    style = "" if cols == 3 else ' style="grid-template-columns: 1fr 1fr;"'
    items = "\n  ".join(
        f'<div class="micro"><div class="lab">{c["label"]}</div>'
        f'<div class="v">{c["value"]}<span class="u">{c["unit"]}</span></div>'
        f'<div class="ctx">{c["ctx_html"]}</div></div>'
        for c in s["cells"]
    )
    return f'<div class="micro-grid"{style}>\n  {items}\n</div>'


def render_corpus_bars(s):
    kind_cls = {"neutral": "hca", "now": "now", "projected": "proj"}
    parts = []
    for r in s["rows"]:
        cls = kind_cls.get(r.get("kind", "neutral"), "hca")
        accent = " accent" if r.get("kind") in ("now", "projected") else ""
        parts.append(f'<div class="clbl">{r["label_html"]}</div>')
        parts.append(
            f'<div class="cbar {cls}" style="width:{r["width_pct"]}%"></div>'
        )
        parts.append(f'<div class="cnum{accent}">{r["number_html"]}</div>')
    return '<div class="corpus">\n  ' + "\n  ".join(parts) + "\n</div>"


def render_feat_list(s):
    items = []
    for it in s["items"]:
        badge = ""
        if it.get("badge"):
            badge = f' <span class="badge">{it["badge"]}</span>'
        items.append(
            f'<li><span class="num">{it["num"]}</span>'
            f'<div><h4>{it["title_html"]}{badge}</h4>'
            f'<p>{it["body_html"]}</p></div></li>'
        )
    return '<ul class="feat-list">\n  ' + "\n  ".join(items) + "\n</ul>"


def render_mock_report(s):
    sections = []
    for sec in s["sections"]:
        lines = "\n      ".join(
            f'<div class="mock-line {ln}"></div>' for ln in sec["lines"]
        )
        sections.append(
            f'<div class="mock-section"><span class="h">{sec["label"]}</span>\n      {lines}\n    </div>'
        )
    note = ""
    if s.get("placeholder_note"):
        note = f'<p class="placeholder-note">{s["placeholder_note"]}</p>'
    cap = ""
    if s.get("caption_html"):
        cap = f'<p class="report-cap">{s["caption_html"]}</p>'
    return f"""<div class="report"><div class="frame">
  <p class="mock-sub">{s.get('subhead_html','')}</p>
  <p class="mock-h">{s['title_html']}</p>
  {"".join(sections)}
  {note}
</div></div>{cap}"""


RENDER_SECTION = {
    "card": render_card,
    "framework": render_framework,
    "results": render_results,
    "vignette": render_vignette,
    "subway": render_subway,
    "micro_grid": render_micro_grid,
    "corpus_bars": render_corpus_bars,
    "feat_list": render_feat_list,
    "mock_report": render_mock_report,
}


def render_columns(columns: list) -> str:
    out = []
    for col in columns:
        sections = col.get("sections", [])
        rendered = []
        for s in sections:
            kind = s["kind"]
            if kind not in RENDER_SECTION:
                raise ValueError(f"Unknown section kind: {kind}")
            rendered.append(RENDER_SECTION[kind](s))
        out.append('<div class="col">\n' + "\n".join(rendered) + "\n</div>")
    return "\n".join(out)


def render_cta(cta: dict) -> str:
    return f"""<div class="cta">
    <div class="msg">{cta['message_html']}</div>
    <div class="contact">{cta['contact_html']}</div>
</div>"""


def render_footer(f: dict) -> str:
    logos = "\n        ".join(
        f'<img src="{E(L["src"])}" alt="{E(L["alt"])}"'
        + (f' style="height:{L["height_in"]}in;"' if L.get("height_in") else "")
        + "/>"
        for L in f["logos"]
    )
    return f"""<footer class="foot">
    <div class="ack">{f['acknowledgments_html']}</div>
    <div class="logos">
        {logos}
    </div>
</footer>"""


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def build(content: dict) -> str:
    out = TEMPLATE
    out = out.replace("{{ STYLES_CSS }}", STYLES)
    out = out.replace("{{ SCALE_JS }}", SCALE_JS)
    out = out.replace("{{ meta.title_plain }}", E(content["meta"]["title_plain"]))
    out = out.replace("{{ COMPONENT_HEADER }}", render_header(content["meta"]))
    out = out.replace("{{ COMPONENT_HERO_STATS }}", render_hero_stats(content["hero_stats"]))
    out = out.replace("{{ COMPONENT_PROBLEM }}", render_problem(content.get("problem")))
    out = out.replace("{{ COMPONENT_BODY_COLUMNS }}", render_columns(content["columns"]))
    out = out.replace("{{ COMPONENT_CTA }}", render_cta(content["cta"]))
    out = out.replace("{{ COMPONENT_FOOTER }}", render_footer(content["footer"]))
    return out


def main():
    ap = argparse.ArgumentParser(description="Render a poster from content.yaml")
    ap.add_argument("content", help="Path to content YAML file")
    ap.add_argument("-o", "--output", default="poster.html", help="Output HTML path")
    args = ap.parse_args()

    with open(args.content) as f:
        content = yaml.safe_load(f)

    html = build(content)
    Path(args.output).write_text(html)
    print(f"Wrote {args.output} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
