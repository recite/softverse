"""The one stylesheet the lookup and package pages share.

Plain on purpose, after Kieran Healy's site rather than a dashboard: one serif
face, black text, one link colour, hairline rules, and nothing drawn around
the numbers but space. Valkyrie, the face that site uses, is licensed, so the
stack starts with Charter, which ships with macOS and Windows' Sitka is close
to; nothing is fetched from a font server.

Old-style figures in running text, lining tabular figures in tables, where
numbers have to line up down a column.
"""

STYLE = """<style>
:root {
  --ink: #111; --soft: #6b6b6b; --rule: #dddddd; --ground: #ffffff;
  --link: #0047ab; --plot: #0047ab;
}
:root:not([data-theme="light"]) {
  @media (prefers-color-scheme: dark) {
    --ink: #e6e6e6; --soft: #9a9a9a; --rule: #333333; --ground: #141414;
    --link: #7fa7e8; --plot: #7fa7e8;
  }
}
:root[data-theme="dark"] {
  --ink: #e6e6e6; --soft: #9a9a9a; --rule: #333333; --ground: #141414;
  --link: #7fa7e8; --plot: #7fa7e8;
}
* { box-sizing: border-box; }
html { font-size: 19px; }
body {
  margin: 0; background: var(--ground); color: var(--ink);
  font-family: Charter, "Bitstream Charter", "Sitka Text", Cambria, serif;
  line-height: 1.5; font-variant-numeric: oldstyle-nums;
  text-rendering: optimizeLegibility;
}
.wrap { max-width: 40rem; margin: 0 auto; padding: 3rem 1rem 5rem; }
header { margin-bottom: 2.2rem; }
.crumb { color: var(--soft); font-size: 0.85rem; margin: 0 0 1.4rem; }
h1 { font-size: 1.9rem; font-weight: 600; line-height: 1.2; margin: 0 0 0.3rem; }
h2 { font-size: 1.1rem; font-weight: 600; margin: 2.4rem 0 0.5rem; }
p { margin: 0 0 0.9rem; }
.lede { color: var(--soft); }
.stat { font-size: 1.35rem; margin: 0.2rem 0 0.4rem; }
.note { color: var(--soft); font-size: 0.85rem; }
a { color: var(--link); text-decoration-thickness: 1px; text-underline-offset: 2px; }
.pkg, code, pre { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
.pkg { font-size: 0.9em; }
table {
  border-collapse: collapse; width: 100%; font-size: 0.9rem;
  font-variant-numeric: lining-nums tabular-nums;
}
th { text-align: left; font-weight: 600; color: var(--soft); border-bottom: 1px solid var(--ink); }
th, td { padding: 0.3rem 0.6rem 0.3rem 0; vertical-align: top; }
td { border-bottom: 1px solid var(--rule); }
th.num, td.num { text-align: right; padding-left: 1.6rem; }
/* A left-aligned column after a number would sit flush against its digits. */
td.num + td:not(.num), th.num + th:not(.num) { padding-left: 1.8rem; }
th button { all: unset; cursor: pointer; }
th button:hover { color: var(--link); }
.scroll { overflow-x: auto; }
pre { font-size: 0.75rem; white-space: pre-wrap; word-break: break-all;
      border-left: 2px solid var(--rule); padding: 0.2rem 0 0.2rem 0.8rem; margin: 0.5rem 0; }
input[type="search"], select {
  font: inherit; font-size: 0.9rem; color: var(--ink); background: var(--ground);
  border: 0; border-bottom: 1px solid var(--ink); padding: 0.2rem 0; margin-right: 1rem;
}
input:focus-visible, select:focus-visible, a:focus-visible, th button:focus-visible {
  outline: 2px solid var(--link); outline-offset: 2px;
}
ol.board { padding-left: 1.4rem; margin: 0; font-variant-numeric: lining-nums tabular-nums; }
ol.board li { display: flex; gap: 0.6rem; font-size: 0.9rem; }
ol.board .n { margin-left: auto; color: var(--soft); }
.boards { display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); gap: 0 1.6rem; }
svg.trend { width: 100%; height: auto; overflow: visible; }
svg.trend text { font-family: inherit; font-size: 11px; fill: var(--soft);
                 font-variant-numeric: lining-nums tabular-nums; }
svg.trend .axis { stroke: var(--rule); }
svg.trend .line { fill: none; stroke: var(--plot); stroke-width: 1.5; }
svg.trend .pt { fill: var(--plot); }
footer { margin-top: 3rem; color: var(--soft); font-size: 0.8rem; }
</style>"""
