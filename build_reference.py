#!/usr/bin/env python3
"""Generate docs/reference.html — a collapsible single-page reference card."""

import html
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "python"))
import bridgepandas as bp  # noqa: E402

H_NUMERIC = [
    ("h.SPADES, h.HEARTS, h.DIAMONDS, h.CLUBS", "Suit length (0–13)."),
    ("h.HCP",        "High card points: A=4, K=3, Q=2, J=1."),
    ("h.AKQ_POINTS", "AKQ points: A=3, K=2, Q=1."),
    ("h.CONTROLS",   "Controls: A=2, K=1."),
    ("h.ACES, h.KINGS, h.QUEENS, h.JACKS, h.TENS", "Count of that honor held."),
    ("h.TOP2 … h.TOP5", "Count of top-2 through top-5 honors held across all suits."),
    ("h.QUICK_TRICKS_X2", "Quick tricks × 2 (integer): AK=4, AQ=3, A=2, KQ=2, Kx=1. Compare as <code>h.QUICK_TRICKS_X2 &gt;= 3</code> for ≥1.5 quick tricks."),
    ("h.LONGEST_SUIT, h.SECOND_SUIT, h.SHORTEST_SUIT", "Length of the longest / 2nd-longest / shortest suit."),
    ("h.VOIDS, h.SINGLETONS, h.DOUBLETONS", "Number of suits with exactly 0, 1, or 2 cards."),
    ("h.NUM(spec)", 'Count of cards matching a spec — same syntax as <a href="#Hand.num"><code>Hand.num()</code></a>, e.g. <code>h.NUM("A")</code> counts aces.'),
]

H_BOOLEAN = [
    ("h.MATCH_SHAPE(spec)", 'Shape constraint, e.g. <code>"any 5332"</code>, <code>"44xx"</code>, <code>"4432 + 4333"</code>.'),
    ("h.GOOD_SUIT(spec, suit)", 'Suit satisfies a holding pattern — same syntax as <a href="#Hand.good_suit"><code>Hand.good_suit()</code></a>, e.g. <code>h.GOOD_SUIT("AJx,KQx", "H")</code>.'),
    ("h.HAS(card)", 'Hand contains a specific card, e.g. <code>h.HAS("SA")</code>.'),
    ("h.ANY(suit)", 'At least one card held in <em>suit</em>, e.g. <code>h.ANY("S")</code>.'),
    ("h.ALL_HANDS", 'Unconstrained — matches every possible hand.'),
]

H_DEALSET = [
    ("h.NORTH(hs), h.SOUTH(hs), h.EAST(hs), h.WEST(hs)", "Apply a hand constraint to a specific seat to get a deal constraint."),
]

SECTIONS = [
    ("Pandas DataFrames", [bp.random_deals, bp.add_dds]),
    ("Hand Sets", "handsets"),
    ("Deals & Hands", [bp.Deal, bp.Hand]),
    ("Other Bridge Concept Classes", [bp.Auction, bp.Bid, bp.Call, bp.Contract, bp.DeclaredContract, bp.Direction, bp.TableVuln]),
    ("Utility Functions", [bp.score, bp.is_declarer_vulnerable, bp.scorediff_imps, bp.scorediff_matchpoints, bp.board_number_to_dealer_vuln, bp.dealer_vuln_to_board_number]),
]


def esc(s: str) -> str:
    return html.escape(str(s))


def fmt_doc(text: str) -> str:
    """Convert basic rst/markdown to HTML."""
    text = esc(text)
    text = re.sub(r'``(.+?)``', r'<code>\1</code>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*([^*\n]+)\*', r'<em>\1</em>', text)
    text = text.replace('\n', '<br>\n')
    return text


def get_sig(obj) -> str:
    try:
        return esc(str(inspect.signature(obj)))
    except (ValueError, TypeError):
        return "(...)"


def render_fn(fn, name: str | None = None, anchor: str | None = None) -> str:
    name = name or fn.__name__
    doc = fmt_doc(inspect.getdoc(fn) or "")
    inner = f'<div class="doc">{doc}</div>' if doc else ""
    id_attr = f' id="{esc(anchor)}"' if anchor else ""
    return (
        f'<details class="item fn"{id_attr}>'
        f'<summary><code><span class="fn-name">{esc(name)}</span>{get_sig(fn)}</code></summary>'
        f'{inner}'
        f'</details>\n'
    )


def render_prop(name: str, prop: property, anchor: str | None = None) -> str:
    doc = fmt_doc(inspect.getdoc(prop) or "")
    inner = f'<div class="doc">{doc}</div>' if doc else ""
    id_attr = f' id="{esc(anchor)}"' if anchor else ""
    return (
        f'<details class="item prop"{id_attr}>'
        f'<summary><code><span class="prop-badge">prop</span> '
        f'<span class="prop-name">{esc(name)}</span></code></summary>'
        f'{inner}'
        f'</details>\n'
    )


def render_class(cls) -> str:
    name = cls.__name__
    class_doc = fmt_doc(inspect.getdoc(cls) or "")

    init_doc = ""
    if cls.__init__ is not object.__init__:
        raw = inspect.getdoc(cls.__init__) or ""
        if raw and raw != (inspect.getdoc(cls) or ""):
            init_doc = fmt_doc(raw)

    members_html = ""
    for mname, mobj in sorted(cls.__dict__.items()):
        if mname.startswith("_"):
            continue
        anchor = f"{name}.{mname}"
        if isinstance(mobj, property):
            members_html += render_prop(mname, mobj, anchor=anchor)
        else:
            fn = mobj.__func__ if isinstance(mobj, (staticmethod, classmethod)) else mobj
            if callable(fn):
                members_html += render_fn(fn, mname, anchor=anchor)

    inner = ""
    if class_doc:
        inner += f'<div class="doc">{class_doc}</div>'
    if init_doc:
        inner += f'<div class="doc init-doc">{init_doc}</div>'
    if members_html:
        inner += f'<div class="methods">{members_html}</div>'

    return (
        f'<details class="item cls" id="{esc(name)}">'
        f'<summary><code><span class="kw">class</span> '
        f'<span class="cls-name">{esc(name)}</span>{get_sig(cls)}</code></summary>'
        f'{inner}'
        f'</details>\n'
    )


def _metrics_table(rows: list) -> str:
    return '<table class="metrics">' + "".join(
        f'<tr><td><code>{name}</code></td><td>{desc}</td></tr>'
        for name, desc in rows
    ) + '</table>'


def render_handsets() -> str:
    return (
        f'<div class="narrative">'
        f'<p>All hand constraints are built from <code>h</code>. '
        f'Hand sets combine with <code>&amp;</code> (and), <code>|</code> (or), '
        f'<code>~</code> (not). Pass them as constraints to <code>random_deals()</code>.</p>'
        f'<p>Numeric metrics can be added together and multiplied by constants before comparing, '
        f'so composite rules are expressed naturally.</p>'
        f'<p><strong>Example (1NT Opener):</strong> '
        f'<code>(h.HCP &gt;= 15) &amp; (h.HCP &lt;= 17) &amp; h.MATCH_SHAPE("any 4333 + any 4432 + any 5332")</code></p>'
        f'<p><strong>Example (rule of 20):</strong> '
        f'<code>h.HCP + h.LONGEST_SUIT + h.SECOND_SUIT &gt;= 20</code></p>'
        f'<h3 class="metrics-heading">Numeric metrics</h3>'
        f'<p class="metrics-sub">Compare with <code>==</code>, <code>!=</code>, <code>&lt;</code>, '
        f'<code>&lt;=</code>, <code>&gt;</code>, <code>&gt;=</code> to produce a hand set.</p>'
        f'{_metrics_table(H_NUMERIC)}'
        f'<h3 class="metrics-heading">Boolean constraints</h3>'
        f'<p class="metrics-sub">These return a hand set directly — no comparison needed.</p>'
        f'{_metrics_table(H_BOOLEAN)}'
        f'<h3 class="metrics-heading">DealSet converters</h3>'
        f'<p class="metrics-sub">Lift a hand constraint to a full-deal constraint.</p>'
        f'{_metrics_table(H_DEALSET)}'
        f'</div>'
    )


def render_section(title: str, items) -> str:
    if items == "handsets":
        body = render_handsets()
    else:
        body = "".join(
            render_class(obj) if inspect.isclass(obj) else render_fn(obj)
            for obj in items
        )
    return (
        f'<details class="section" open>'
        f'<summary>{esc(title)}</summary>'
        f'<div class="section-body">{body}</div>'
        f'</details>\n'
    )


CSS = """\
*, *::before, *::after { box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 15px;
    line-height: 1.6;
    color: #1a1a2e;
    background: #f4f6fb;
    max-width: 860px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
}

h1 { font-size: 1.8rem; margin-bottom: 0.2rem; }
.subtitle { color: #666; margin-top: 0; margin-bottom: 2rem; }

/* ── Sections ── */
details.section {
    background: #fff;
    border: 1px solid #d0d7e3;
    border-radius: 8px;
    margin-bottom: 1rem;
    overflow: hidden;
}

details.section > summary {
    font-weight: 700;
    font-size: 1.05rem;
    padding: 0.7rem 1rem;
    background: #e8f0fe;
    cursor: pointer;
    user-select: none;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #1a3a7c;
}

details.section > summary::before {
    content: "▶";
    font-size: 0.65rem;
    transition: transform 0.15s;
    color: #4a6cf7;
}
details.section[open] > summary::before { transform: rotate(90deg); }

.section-body { padding: 0.5rem 0.75rem 0.75rem; }

/* ── Items ── */
details.item {
    border-left: 3px solid transparent;
    margin: 0.25rem 0;
    border-radius: 4px;
}
details.item.cls  { border-left-color: #4a6cf7; }
details.item.fn   { border-left-color: #1a9e5c; }
details.item.prop { border-left-color: #e07b00; }

details.item > summary {
    padding: 0.35rem 0.6rem;
    cursor: pointer;
    user-select: none;
    list-style: none;
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    border-radius: 4px;
}
details.item > summary:hover { background: #f0f4ff; }
details.item > summary::before {
    content: "▶";
    font-size: 0.55rem;
    color: #bbb;
    transition: transform 0.15s;
    flex-shrink: 0;
    position: relative;
    top: -1px;
}
details.item[open] > summary::before { transform: rotate(90deg); }
details.item > summary code { font-size: 0.9rem; background: none; }

/* ── Nested methods ── */
.methods { padding-left: 1.25rem; margin-top: 0.2rem; }
.methods details.item { border-left-color: #b0bec5; }
.methods details.item > summary:hover { background: #f5f5f5; }

/* ── Hand Sets narrative ── */
.narrative {
    padding: 0.5rem 0.75rem;
    font-size: 0.92rem;
    color: #333;
}
.narrative p { margin: 0.4rem 0; }
h3.metrics-heading {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #1a3a7c;
    margin: 1rem 0 0.1rem;
    border-bottom: 1px solid #d0d7e3;
    padding-bottom: 0.2rem;
}
p.metrics-sub { margin: 0.2rem 0 0.3rem; color: #555; font-size: 0.85rem; }
table.metrics {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.6rem;
    font-size: 0.88rem;
}
table.metrics td {
    padding: 0.3rem 0.6rem;
    vertical-align: top;
    border-top: 1px solid #e8ecf4;
}
table.metrics tr:first-child td { border-top: none; }
table.metrics td:first-child {
    white-space: nowrap;
    color: #1a56db;
    width: 40%;
}

/* ── Doc text ── */
.doc {
    padding: 0.35rem 0.6rem 0.35rem 1.6rem;
    color: #444;
    font-size: 0.9rem;
}
.init-doc {
    background: #f8f9fa;
    border-radius: 4px;
    margin: 0.2rem 0.4rem;
    padding: 0.35rem 0.6rem 0.35rem 1.6rem;
}

/* ── Syntax colouring ── */
.kw        { color: #7c3aed; font-weight: 600; }
.cls-name  { color: #1a56db; font-weight: 600; }
.fn-name   { color: #166534; font-weight: 600; }
.prop-name { color: #b45309; font-weight: 600; }
.prop-badge {
    font-size: 0.7rem;
    font-weight: 700;
    color: #fff;
    background: #e07b00;
    border-radius: 3px;
    padding: 0 4px;
    vertical-align: middle;
    margin-right: 2px;
}

code {
    font-family: "SF Mono", "Fira Code", Consolas, monospace;
}

summary::-webkit-details-marker { display: none; }
"""

HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>bridgepandas reference</title>
  <style>{css}</style>
</head>
<body>
  <h1>bridgepandas</h1>
  <p class="subtitle">Click any item to expand. Classes in blue, functions in green, properties in orange.</p>
  {body}
  <script>
    function openTarget() {{
      var el = document.getElementById(location.hash.slice(1));
      while (el) {{ if (el.tagName === 'DETAILS') el.open = true; el = el.parentElement; }}
      if (el) el.scrollIntoView();
    }}
    window.addEventListener('hashchange', openTarget);
    if (location.hash) openTarget();
  </script>
</body>
</html>
"""


def main():
    body = "\n".join(render_section(title, items) for title, items in SECTIONS)
    out = ROOT / "docs" / "reference.html"
    out.write_text(HTML.format(css=CSS, body=body))
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
