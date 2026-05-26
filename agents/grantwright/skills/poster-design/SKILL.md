---
name: poster-design
agent: grantwright
description: Generate a print-ready, single-file HTML conference poster with an editorial design system — true physical dimensions, CSS-transform preview scaling, three-font typographic triad, single accent color, and a complete component library (hero stats, framework stack, results card, vignette, subway diagram, micro-cards, corpus comparison, SVG bar chart, mock report, feature list, CTA, footer). Given structured content, produces a finished poster.html.
inputs:
  - path: content.yaml (a populated copy of content.schema.yaml)
  - dir:  assets/ (logos, optional photos)
outputs:
  - file: poster.html (self-contained, ~70 KB, no build step)
  - file: preview.png (optional, via headless Chrome)
invocation:
  - Manual: edit content.yaml → run build.py → open poster.html
  - Via GrantWright agent: walk user through interview → write content.yaml → invoke build
when_to_use: User wants a scientific or clinical conference poster as a single self-contained HTML file rendered at true physical dimensions for plotter printing.
when_not_to_use: PowerPoint-style slides, interactive web posters, marketing collateral with bleeds/CMYK requirements.
---

# poster-design — editorial print HTML poster generator

A complete, opinionated system for producing print-ready academic conference posters as a single HTML file. The poster lives at true physical dimensions (e.g. 48 in × 36 in landscape); CSS `transform: scale(--scale)` shrinks it for on-screen preview, and `@page` resets to 1.0 for PDF print.

The canonical implementation in [`example/poster.html`](example/poster.html) is the *Clinical Intelligence — From a billion cells to a bedside diagnosis* poster (Michigan Clinical Genetics Conference 2026, 48 × 36 in landscape, ~73 KB). Use it as the reference — it exercises every component in the system.

> **Reproducibility claim.** If you populate [`content.example.yaml`](content.example.yaml) and run `python build.py content.example.yaml`, the output `poster.html` is byte-for-byte equivalent to `example/poster.html` modulo whitespace.

---

## 1 · Files in this skill

| File | Purpose |
|---|---|
| [`SKILL.md`](SKILL.md) | This file — design system + procedure. |
| [`template.html`](template.html) | The poster skeleton with `{{moustache}}` placeholders. All CSS lives here. Don't edit unless you're changing the system. |
| [`content.schema.yaml`](content.schema.yaml) | Annotated YAML schema describing every content field the template consumes. Copy and fill in. |
| [`content.example.yaml`](content.example.yaml) | Fully-populated content from the *Clinical Intelligence* poster — reproduces the example end to end. |
| [`build.py`](build.py) | Renders `template.html` against a `content.yaml` to produce a final `poster.html`. Pure stdlib (Jinja2 if available; falls back to a tiny moustache parser). |
| [`components/`](components/) | Each major HTML component, isolated and annotated, so the agent and the user can read one at a time. |
| [`example/`](example/) | The canonical *Clinical Intelligence* poster (`poster.html`, `preview.png`, `assets/`). |

---

## 2 · Mental model

The poster is a CSS grid of seven rows on the `.poster` element:

```
┌───────────────────────────────────────────────────────────────────────┐
│  header.head           ← lockup + venue meta                          │
├───────────────────────────────────────────────────────────────────────┤
│  div.rule              ← 1-px hairline                                │
├───────────────────────────────────────────────────────────────────────┤
│  div.hero-stats        ← 3–4 hero numbers across the top              │
├───────────────────────────────────────────────────────────────────────┤
│  div.body              ← 4-column main panel                          │
│  ┌──────┬──────┬──────┬──────┐                                        │
│  │ col1 │ col2 │ col3 │ col4 │   ← cols 2+3 visually merge            │
│  │      │      │      │      │     into a central panel via midrule  │
│  └──────┴──────┴──────┴──────┘                                        │
├───────────────────────────────────────────────────────────────────────┤
│  div.bottom-plots      ← optional: 4 small plots above the CTA        │
├───────────────────────────────────────────────────────────────────────┤
│  div.cta               ← closing line + contact block                 │
├───────────────────────────────────────────────────────────────────────┤
│  footer.foot           ← acknowledgements + logo strip                │
└───────────────────────────────────────────────────────────────────────┘
```

The `.poster` is wrapped in `.stage`, which holds the *scaled* visual area, and the inner `.poster` is the actual print-sized element transformed by `--scale`. A tiny inline JS sets `--scale` on resize. `@media print` resets it to `1.0`.

---

## 3 · Design tokens

### Dimensions

```css
:root {
  --pw: 48in;        /* print width  */
  --ph: 36in;        /* print height */
  --scale: 0.32;     /* JS overrides on resize for preview */
}
@page { size: 48in 36in; margin: 0; }
```

Spacing inside the poster is **always in inches** so the layout survives a change of physical size. Type sizes are in `px` so they survive `--scale` correctly (since `transform: scale` scales everything).

### Type triad

| Role | Family | Use |
|---|---|---|
| Display serif | **Newsreader** | Title, lede, big numbers, italic quotes |
| UI sans | **Inter** (300/400/500/600/700/800) | Section heads, body, captions |
| Mono | **JetBrains Mono** | Kickers, section indices `01 · Why now`, labels |

Load via Google Fonts in `<head>` with `preconnect`.

### Color tokens

```css
--ink:#0F1419;  --body:#2A2F36;  --muted:#5C6470;  --dim:#8A93A0;
--rule:#E2E5EA; --rule-strong:#C2C7CF; --paper:#FFFFFF;
--tint:#F6F8FA; --tint-2:#EEF2F5;
--accent:#0EA5C2;     /* HERO ACCENT — pick one */
--accent-ink:#0B6E83; /* readable on white */
--accent-deep:#084C5C;
--accent-tint:#E6F6FA;
--accent-tint-2:#F2FBFD;
--warn:#B45309; --warn-tint:#FEF7EC;  /* used very sparingly */
```

`--accent`, `--accent-ink`, `--accent-deep`, `--accent-tint`, `--accent-tint-2` are the **only** customisations needed per poster. Everything else stays.

---

## 4 · The component library

Each component lives in [`components/`](components/) as a standalone annotated HTML snippet. The template assembles them. The reference example in [`example/poster.html`](example/poster.html) uses every one of them.

| # | Component | File | Required? |
|---|---|---|---|
| 1 | Header lockup (kicker, title, deck, authors, venue meta) | [`01-header.html`](components/01-header.html) | ✅ |
| 2 | Hero stats band (3–4 big numbers) | [`02-hero-stats.html`](components/02-hero-stats.html) | ✅ |
| 3 | Section card (heading, lede, body) | [`03-section-card.html`](components/03-section-card.html) | ✅ |
| 4 | Framework stack (L1→L4 layered capabilities) | [`04-framework-stack.html`](components/04-framework-stack.html) | optional |
| 5 | Results card (metric tiles + narrative) | [`05-results-card.html`](components/05-results-card.html) | optional |
| 6 | Vignette / pull-quote (worked example) | [`06-vignette.html`](components/06-vignette.html) | optional |
| 7 | Subway diagram (pipeline stations) | [`07-subway.html`](components/07-subway.html) | optional |
| 8 | Micro-card grid (small stat tiles) | [`08-micro-grid.html`](components/08-micro-grid.html) | optional |
| 9 | Corpus comparison (horizontal bars) | [`09-corpus-bars.html`](components/09-corpus-bars.html) | optional |
| 10 | SVG bar chart (per-item scores) | [`10-svg-bar-chart.html`](components/10-svg-bar-chart.html) | optional |
| 11 | Feature list (numbered items with roadmap badges) | [`11-feat-list.html`](components/11-feat-list.html) | optional |
| 12 | Mock report (sample deliverable) | [`12-mock-report.html`](components/12-mock-report.html) | optional |
| 13 | CTA band (closing line + contact) | [`13-cta.html`](components/13-cta.html) | ✅ |
| 14 | Footer (acknowledgements + logos) | [`14-footer.html`](components/14-footer.html) | ✅ |

Open each file to see the exact HTML, what classes it uses, what content fields it needs, and what the rendered output looks like in the example.

---

## 5 · Procedure (when GrantWright invokes me)

```
┌───────────────────────────────────────────────────────────────┐
│  1. cp content.schema.yaml → my-poster/content.yaml           │
│  2. Fill in content.yaml block by block (GrantWright drives   │
│     the interview; user provides the facts).                  │
│  3. Drop logos/photos into my-poster/assets/                  │
│  4. Set accent color (one hex; derive the 5 tones from it).   │
│  5. python build.py my-poster/content.yaml                    │
│     → writes my-poster/poster.html                            │
│  6. Open in Chrome to preview. Resize window to fit-test.     │
│  7. (Optional) chromium --headless --screenshot=preview.png   │
│        --window-size=3456,2592 file://$PWD/poster.html        │
│  8. Print → Save as PDF → Custom paper size → Margins None →  │
│     Background graphics ON.                                   │
└───────────────────────────────────────────────────────────────┘
```

---

## 6 · Content schema (summary)

See [`content.schema.yaml`](content.schema.yaml) for the full annotated version. Top-level fields:

```yaml
meta:
  title_html:        # H1 — supports <em> for italicized accent words
  deck:              # 1–2 sentence sub-headline
  kicker:            # short eyebrow above title, JetBrains Mono uppercase
  authors_html:      # main author line
  affiliation_html:  # institution(s)
  venue_name_html:   # 2-line conference name (use <br>)
  venue_date:        # "May 1, 2026"
  venue_location_html: # multi-line institution / city
  venue_host_html:   # multi-line host info

dimensions:
  width_in: 48
  height_in: 36
  orientation: landscape   # or portrait

accent:
  hex: "#0EA5C2"
  ink: "#0B6E83"
  deep: "#084C5C"
  tint: "#E6F6FA"
  tint_2: "#F2FBFD"

hero_stats:
  - number: "40–60"
    unit: "%"
    label_html: "of clinical whole-genome sequences yield a <b>variant of uncertain significance</b>."
  - ...

columns:
  - col_id: 1
    sections:
      - kind: card           # 01–14 above
        heading: "The clinical genetics gap."
        index: "01 · Why now"
        lede_html: "..."
        body_html: ["...", "..."]
      - kind: vignette
        label: "Design principle"
        ask_html: "..."
        ans_html: null
  - col_id: 2
    sections:
      - kind: framework
        heading: "A framework for clinical genetics AI."
        index: "03 · Framework"
        lede_html: "..."
        layers:
          - id: L1
            foundation: true
            title: "Phenotype foundation — learned from the literature"
            body_html: "..."
          - id: L2
            ...
  - col_id: 3
    sections:
      - kind: results
        heading: "Validation — 13 disease variants, end to end."
        index: "04 · Evidence"
        title: "Genotype → phenotype proof-of-concept"
        tag: "BMMC · 69k cells · ~70 s · 1 CPU"
        metrics:
          - { value: "0.933", label: "Mean F<sub>1</sub>" }
          - { value: "0.929", label: "Precision" }
          - { value: "0.962", label: "Recall" }
          - { value: "0.947", label: "Held-out F<sub>1</sub>" }
        narrative_html: "..."
        chart:
          kind: svg_bars
          title: "F1 score per disease variant"
          rows:
            - { label: "Sickle cell · HBB E6V", value: 1.00, weight: "perfect" }
            - ...
  - col_id: 4
    sections:
      - kind: subway
        stations: [Raw, Align, Count, Store, Model, Insight]
      - kind: micro_grid
        cols: 2
        cells:
          - { label: "Input · .1fq", value: "24", unit: "%", ctx: "smaller than FASTQ.gz; ..." }
          - ...
      - kind: corpus_bars
        rows:
          - { label: "Human Cell Atlas", width: 5, number: "50 M", kind: "neutral" }
          - { label: "singlet (today)", width: 33, number: "354 M", kind: "now" }
          - ...
      - kind: feat_list
        items:
          - num: "01"
            title_html: "Literature, queryable."
            body_html: "..."
          - num: "02"
            title_html: "Chart-to-case-report."
            badge: "Roadmap"
            body_html: "..."
      - kind: mock_report
        title_html: "..."
        sections: [Phenotype (HPO), Differential, Suggested next steps]
        caption_html: "..."

cta:
  message_html: "The substrate is built. <em>Now we make it answer the questions...</em>"
  contact_html: |
    <b>Zach DeBruine, PhD</b>
    debruinz@gvsu.edu<br/>
    debruine.dev · herd.social<br/>
    GVSU Bioinformatics

footer:
  acknowledgements_html: "..."
  logos:
    - { src: "assets/gvsu.jpg",          alt: "GVSU",      class: "gv" }
    - { src: "assets/corewell-trim.png", alt: "Corewell" }
    - { src: "assets/herd.svg",          alt: "Herd",      class: "herd" }
```

---

## 7 · Decision tree for the agent

When GrantWright is interviewing the user, the question order is:

```
Q1.  What conference / venue / date?
Q2.  Physical dimensions and orientation?
       → fills meta.venue_* + dimensions.*
Q3.  Single accent color? (one hex tied to brand/venue)
       → fills accent.* (derive 5 tones)
Q4.  In one sentence, what is the poster trying to convince a stranger of?
       → becomes meta.title_html (italicize the punch word)
Q5.  Give me the 1–2 sentence sub-headline.
       → meta.deck
Q6.  Authors, affiliation, contact?
       → meta.authors_html, affiliation_html, cta.contact_html
Q7.  3–4 hero stats that motivate the problem (number + unit + 1-line label)?
       → hero_stats
Q8.  For each of 4 columns, give me 1–3 sections. For each section pick a kind:
       card | framework | results | vignette | subway | micro_grid |
       corpus_bars | feat_list | mock_report
       → columns[*].sections[*]
Q9.  Closing line + contact block?
       → cta.*
Q10. Acknowledgements + logos to drop in assets/?
       → footer.*
```

After Q10, GrantWright writes `content.yaml`, runs `python build.py content.yaml`, and opens the resulting `poster.html`.

---

## 8 · Anti-patterns

- ❌ Adding a second accent color "for variety".
- ❌ Drop shadows or gradients on type.
- ❌ Icon fonts or Font Awesome.
- ❌ Bootstrap / Tailwind / any framework.
- ❌ Measuring spacing in `px` (use `in`).
- ❌ More than 3 type families.
- ❌ Inline base64 fonts. Google Fonts CDN is fine.
- ❌ Making up numbers, citations, or affiliations the user did not provide.

---

## 9 · Provenance

Extracted from *Clinical Intelligence — From a billion cells to a bedside diagnosis*, Michigan Clinical Genetics Conference 2026, Zachary DeBruine et al. Original poster: [`example/poster.html`](example/poster.html).
