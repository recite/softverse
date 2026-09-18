.PHONY: paper site release-tables check test lint clean-paper clean-site

PAPER := paper/paper.qmd
MD    := paper/paper.md
PDF   := paper/softverse.pdf
BIB   := paper/references.bib

# The paper does not build unless both claims it makes about itself hold: that
# every reference resolves, and that no number in the prose was typed by hand.
# Gating on them is the only version of "we checked" that survives a revision,
# since both failures are silent and neither shows up in the rendered output.
paper: $(PDF)

FIGS  := paper/figures/languages.pdf paper/figures/credit.pdf

$(FIGS): build/tally/files.parquet paper/figures.py
	uv run python paper/figures.py

$(PDF): $(MD) $(BIB) $(FIGS) paper/preamble.tex
	uv run python paper/validate_bib.py
	uv run pytest tests/test_paper.py -q --no-cov
	cd paper && pandoc paper.md -o softverse.pdf \
		--citeproc --bibliography=references.bib \
		--pdf-engine=xelatex \
		--include-in-header=preamble.tex \
		--number-sections \
		--shift-heading-level-by=-1 \
		--toc --toc-depth=2 \
		-V geometry:margin=1.15in \
		-V fontsize=11pt \
		-V linkcolor=RoyalBlue \
		-V title="$$(uv run python render_paper.py --meta title)" \
		-V subtitle="$$(uv run python render_paper.py --meta subtitle)" \
		-V author="$$(uv run python render_paper.py --meta author)"
	@echo "built $(PDF)"

# Regenerated from the tally every time: the chunks run, the inline values are
# substituted, and a stale number cannot survive.
$(MD): $(PAPER) build/tally/mentions.parquet
	uv run python paper/render_paper.py $(MD)

# The site. Reads only `data/tally/`, which is tracked, so this is
# the same command locally and in CI, where nothing under `build/tally/`
# exists. `-W` is the point: the site that shipped before this had a toctree
# pointing at nine pages nobody ever wrote, and Sphinx warned about it on
# every build for months while deploying anyway.
# Built from empty every time. Sphinx leaves the HTML for a source you have
# deleted sitting in the output directory, so an incremental build happily
# redeploys a page you removed.
# Generation happens inside docs/conf.py, so this is the same command the
# fleet's docs workflow runs. Building it a second way here is how a site that
# works locally starts failing in CI.
site:
	rm -rf docs/_build
	uv run sphinx-build -b html -W --keep-going docs docs/_build/html
	@echo "built docs/_build/html"

# Refreshes the tracked tables from the pipeline output. Needs the corpus, so
# it runs here and never in CI.
release-tables:
	uv run python scripts/release_tally.py

check:
	uv run python paper/check_paper.py

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

clean-paper:
	rm -f $(MD) $(PDF) paper/paper.html

clean-site:
	rm -rf docs/_build docs/_extra docs/index.md
