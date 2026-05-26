# poster-design — the skill

A reproducible system for generating Zach DeBruine's signature
**48 × 36 in landscape research poster**. Drop in a `content.yaml`,
run `build.py`, get a print-ready single-file HTML poster.

This is the skill that the **GrantWright** agent
(`../../AGENT.md`) calls when a user asks for a poster.

---

## Quick start

```bash
cd agents/grantwright/skills/poster-design
pip install pyyaml
python build.py content.example.yaml -o poster.html
open poster.html   # Chrome → File → Print → Save as PDF (48 × 36 in landscape)
```

The output is **one self-contained `.html` file** (CSS and JS inlined).
The only external dependencies at view-time are the three Google Fonts
loaded via `<link>` — and the optional `assets/` directory referenced
by `footer.logos` and any mock images.

---

## What's in this skill

```
poster-design/
├── SKILL.md               ← read this first; the design system & process
├── README.md              ← (you are here) quick start
├── content.schema.yaml    ← every field, annotated
├── content.example.yaml   ← Clinical Intelligence reproduction
├── template.html          ← shell with {{ }} placeholders
├── styles.css             ← the entire 33 KB stylesheet
├── scale.js               ← auto-scale-to-viewport script
├── build.py               ← deterministic YAML → HTML renderer
├── components/            ← annotated HTML snippets for each section kind
│   ├── 01-header.html
│   ├── 02-hero-stats.html
│   ├── 03-section-card.html
│   ├── 04-framework-stack.html
│   ├── 05-results-card.html
│   ├── 06-vignette.html
│   ├── 07-subway.html
│   ├── 08-micro-grid.html
│   ├── 09-corpus-bars.html
│   ├── 10-svg-bar-chart.html
│   ├── 11-feat-list.html
│   ├── 12-mock-report.html
│   ├── 13-cta.html
│   └── 14-footer.html
└── example/
    ├── poster.html        ← the actual Clinical Intelligence poster
    ├── preview.png        ← rendered preview
    └── assets/            ← logos, photos
```

---

## The intended workflow

1. **Invoke GrantWright** ("Hey GrantWright, make me a poster on X")
2. The agent reads `SKILL.md` and runs the **10-question interview**
   (see `SKILL.md §7 Decision tree`) to collect content.
3. The agent writes `content.yaml` from your answers.
4. The agent runs `python build.py content.yaml -o poster.html`.
5. You print to PDF at **48 × 36 in landscape**, no margins.

If you skip the agent and write `content.yaml` by hand — the schema
in `content.schema.yaml` documents every field. The build is
deterministic: same YAML → byte-identical HTML.

---

## Adding a new section kind

1. Add an entry to `RENDER_SECTION` in `build.py`.
2. Add a `components/NN-yourthing.html` doc snippet.
3. Add the schema entry to `content.schema.yaml`.
4. Add any required CSS to `styles.css`.

---

## Design philosophy

See `SKILL.md` — but in one line:

> *One accent color. Three fonts. A scale and a grid. Trust the
> reader.*

— Zach
