"""Render `paper.qmd` to a self-contained HTML page.

    uv run python paper/render_paper.py [out.html]

Quarto is the real renderer and is not installed here. This does the part
that matters for reading the thing: run the chunks, substitute every
`` `{python} expr` ``, and emit the document with its numbers in place.

It shares `check_paper.py`'s premise -- the `exec`/`eval` run code from a
tracked source file in this repository, which is exactly what Quarto does
with the same file.
"""

from __future__ import annotations

import ast
import io
import os
import re
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

from markdown_it import MarkdownIt

HERE = Path(__file__).parent
PAPER = HERE / "paper.qmd"

_CHUNK = re.compile(r"^```\{python\}\n(.*?)^```", re.M | re.S)
_INLINE = re.compile(r"`\{python\}\s*(.+?)`")
_LABEL = re.compile(r"^#\|\s*label:\s*(\S+)", re.M)
_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)

STYLE = """
:root {
  --ink:        #14171c;
  --ink-soft:   #4a5260;
  --ink-faint:  #767f8d;
  --ground:     #fbfbfa;
  --panel:      #f2f3f1;
  --rule:       #dcdeda;
  --accent:     #1f4e7a;
  --accent-dim: #eaeff4;
  --flag:       #8a3324;
}
:root:not([data-theme="light"]) {
  @media (prefers-color-scheme: dark) {
    --ink:        #e6e8ea;
    --ink-soft:   #a8b0bb;
    --ink-faint:  #79828f;
    --ground:     #14161a;
    --panel:      #1c1f25;
    --rule:       #2b2f37;
    --accent:     #7fb0dc;
    --accent-dim: #1b2833;
    --flag:       #d9836f;
  }
}
:root[data-theme="dark"] {
  --ink:        #e6e8ea;
  --ink-soft:   #a8b0bb;
  --ink-faint:  #79828f;
  --ground:     #14161a;
  --panel:      #1c1f25;
  --rule:       #2b2f37;
  --accent:     #7fb0dc;
  --accent-dim: #1b2833;
  --flag:       #d9836f;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: Charter, "Bitstream Charter", "Iowan Old Style", Georgia, serif;
  font-size: 18px;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}

.sheet {
  max-width: 46rem;
  margin: 0 auto;
  padding: 5rem 1.5rem 8rem;
  display: flex;
  flex-direction: column;
  gap: 1.15rem;
}

header.masthead {
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
  padding-bottom: 2.2rem;
  margin-bottom: 1.4rem;
  border-bottom: 2px solid var(--ink);
}
.eyebrow {
  font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
  font-size: 0.68rem;
  font-weight: 620;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--accent);
}
h1.title {
  font-size: clamp(1.9rem, 4.4vw, 2.65rem);
  line-height: 1.14;
  font-weight: 600;
  letter-spacing: -0.014em;
  margin: 0;
  text-wrap: balance;
}
.byline {
  font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
  font-size: 0.92rem;
  color: var(--ink-soft);
}

h2 {
  font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
  font-size: 1.22rem;
  font-weight: 640;
  letter-spacing: -0.008em;
  margin: 2.6rem 0 0.2rem;
  padding-top: 1.4rem;
  border-top: 1px solid var(--rule);
  text-wrap: balance;
}
h3 {
  font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
  font-size: 1.02rem;
  font-weight: 640;
  margin: 1.9rem 0 0.1rem;
  text-wrap: balance;
}
p { margin: 0; }
strong { font-weight: 640; }

ul { margin: 0.2rem 0; padding-left: 1.1rem; display: flex; flex-direction: column; gap: 0.6rem; }
li { padding-left: 0.2rem; }
li::marker { color: var(--ink-faint); }

blockquote {
  margin: 0.4rem 0;
  padding: 0.9rem 1.2rem;
  background: var(--accent-dim);
  border-left: 3px solid var(--accent);
  color: var(--ink);
  font-style: italic;
}
blockquote p + p { margin-top: 0.6rem; }

code {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.85em;
  background: var(--panel);
  border: 1px solid var(--rule);
  border-radius: 3px;
  padding: 0.08em 0.34em;
}

.tablewrap { overflow-x: auto; margin: 0.5rem 0; }
table {
  border-collapse: collapse;
  width: 100%;
  font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
  font-size: 0.86rem;
  font-variant-numeric: tabular-nums;
}
th {
  text-align: left;
  font-weight: 640;
  color: var(--ink-soft);
  border-bottom: 1.5px solid var(--ink);
  padding: 0.5rem 0.7rem;
  white-space: nowrap;
}
td {
  padding: 0.42rem 0.7rem;
  border-bottom: 1px solid var(--rule);
}
tbody tr:last-child td { border-bottom: none; }
td:not(:first-child), th:not(:first-child) { text-align: right; }
table code { background: none; border: none; padding: 0; }

a { color: var(--accent); text-underline-offset: 2px; }
a:focus-visible, :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

footer.colophon {
  margin-top: 3.4rem;
  padding-top: 1.4rem;
  border-top: 1px solid var(--rule);
  font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
  font-size: 0.8rem;
  color: var(--ink-faint);
  line-height: 1.55;
}

@media (max-width: 640px) {
  body { font-size: 17px; }
  .sheet { padding: 3rem 1.1rem 5rem; }
}
"""


def chunk_output(body: str, namespace: dict) -> str:
    """Run a chunk and return what Quarto would show for it.

    Quarto displays the value of a trailing expression, which is how the
    tables in this paper are produced -- `pres.to_markdown(index=False)` as
    the last line of a chunk. Anything printed counts too.
    """
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return ""
    tail = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tail = ast.Expression(tree.body[-1].value)
        tree.body = tree.body[:-1]

    printed = io.StringIO()
    with redirect_stdout(printed):
        exec(compile(tree, "<chunk>", "exec"), namespace)  # noqa: S102
        value = eval(compile(tail, "<tail>", "eval"), namespace) if tail else None

    out = printed.getvalue()
    if isinstance(value, str):
        out += value
    elif value is not None:
        out += str(value)
    return out


def render_markdown() -> tuple[str, dict]:
    text = PAPER.read_text()
    front = _FRONT.search(text)
    meta: dict[str, str] = {}
    if front:
        for key in ("title",):
            m = re.search(rf'^{key}:\s*"?(.+?)"?\s*$', front.group(1), re.M)
            if m:
                meta[key] = m.group(1)
        text = text[front.end() :]

    namespace: dict = {}
    out: list[str] = []
    cursor = 0
    for chunk in _CHUNK.finditer(text):
        out.append(text[cursor : chunk.start()])
        body = chunk.group(1)
        label = (_LABEL.search(body) or [None, "(unlabelled)"])[1]
        try:
            out.append("\n" + chunk_output(body, namespace) + "\n")
        except Exception as exc:
            traceback.print_exc()
            raise SystemExit(f"chunk failed: {label}") from exc
        cursor = chunk.end()
    out.append(text[cursor:])

    body = "".join(out)

    def substitute(match: re.Match) -> str:
        try:
            return str(eval(match.group(1), namespace))  # noqa: S307
        except Exception as exc:  # noqa: BLE001 - surfaced in the page
            return f"[UNRESOLVED: {exc}]"

    return _INLINE.sub(substitute, body), namespace


def main() -> int:
    # Resolve the destination against the caller's directory first, then move
    # to ours: the chunks read `../build/tally`, which is right relative to
    # this file and wrong from anywhere else, while the output path is the
    # caller's and means what they meant.
    out_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "paper.html"
    ).resolve()
    os.chdir(HERE)
    markdown, namespace = render_markdown()

    if out_path.suffix == ".md":
        # Markdown is what pandoc wants for the PDF; the HTML path below is
        # for reading in a browser. Same substituted source either way, so
        # the two outputs cannot disagree about a number.
        out_path.write_text(markdown, encoding="utf-8")
        print(f"wrote {out_path} ({len(markdown):,} bytes)")
        return 0

    md = (
        MarkdownIt("commonmark", {"html": True}).enable("table").enable("strikethrough")
    )
    html_body = md.render(markdown)
    # Tables scroll in their own container so the page never scrolls sideways.
    html_body = html_body.replace("<table>", '<div class="tablewrap"><table>').replace(
        "</table>", "</table></div>"
    )

    title = "What Social Science Runs On"
    n_deposits = namespace.get("n_deposits", 0)
    n_files = namespace.get("n_files", 0)

    page = f"""<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{STYLE}</style>
<article class="sheet">
<header class="masthead">
  <div class="eyebrow">Working paper &middot; measurement</div>
  <h1 class="title">What software does social science run on?</h1>
  <div class="byline">Gaurav Sood &middot; measuring validated use in replication code</div>
</header>
{html_body}
<footer class="colophon">
Every number on this page is computed from the released tables at render
time; none is typed into the prose, and a test fails the build if one is.
Rendered from <code>paper/paper.qmd</code> over {n_deposits:,} deposits and
{n_files:,} files.
</footer>
</article>
"""
    out_path.write_text(page, encoding="utf-8")
    print(f"wrote {out_path} ({len(page):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
