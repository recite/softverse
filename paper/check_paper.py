"""Execute every computed value in `paper.qmd` and print it.

    uv run python paper/check_paper.py

The paper's claim is that no number in it is typed -- each is a
`` `{python} expr` `` evaluated against the tally. That claim is only worth
anything if the expressions actually run, and until now nothing ran them:
rendering needs Quarto, Quarto is not installed here, and the numbers were
being checked by hand against a block copied out of the file. Hand-checking a
document whose whole point is that it is not hand-checked is the wrong shape.

This runs the chunks in order in one namespace, then evaluates every inline
expression, and prints each with its value. It exits non-zero on the first
failure, so it works as a test; and its output is a list of every number the
paper will print, which is the thing worth reading before publishing one.

It is not a renderer. Quarto still produces the document; this establishes
that the document *can* be produced and says what it will say.

The `exec`/`eval` below run code from `paper.qmd`, which is exactly what
Quarto does with the same file. The input is a tracked source file in this
repository, not user input, so there is no boundary here to cross.
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

PAPER = Path(__file__).parent / "paper.qmd"

#: ```{python} ... ``` chunks, with their `#| label:` if they carry one.
_CHUNK = re.compile(r"^```\{python\}\n(.*?)^```", re.M | re.S)
#: `{python} expr` inline expressions.
_INLINE = re.compile(r"`\{python\}\s*(.+?)`")
_LABEL = re.compile(r"^#\|\s*label:\s*(\S+)", re.M)


def main() -> int:
    text = PAPER.read_text()
    namespace: dict = {}
    failures = 0

    # Chunks first, in document order: later ones depend on earlier ones, and
    # inline expressions depend on all of them.
    for chunk in _CHUNK.finditer(text):
        body = chunk.group(1)
        label_match = _LABEL.search(body)
        label = label_match.group(1) if label_match else "(unlabelled)"
        try:
            exec(compile(body, f"<chunk {label}>", "exec"), namespace)  # noqa: S102
        except Exception:
            print(f"\nCHUNK FAILED: {label}\n")
            traceback.print_exc()
            return 1
        print(f"  chunk ok  {label}")

    print()
    # Strip the chunks before scanning for inline expressions, or a `{python}`
    # inside a chunk's comment would be picked up as prose.
    prose = _CHUNK.sub("", text)
    seen: set[str] = set()
    for match in _INLINE.finditer(prose):
        expression = match.group(1).strip()
        if expression in seen:
            continue
        seen.add(expression)
        try:
            value = eval(expression, namespace)  # noqa: S307
        except Exception as exc:
            print(f"  FAILED  {expression}\n            {type(exc).__name__}: {exc}")
            failures += 1
            continue
        print(f"  {str(value):<28}  {expression}")

    print(f"\n{len(seen)} inline expressions, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
