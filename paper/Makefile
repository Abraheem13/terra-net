# Build the manuscript. Run from the repository root with `make paper`,
# or here with `make`. Assets are regenerated from outputs/tables first.
PY ?= python

.PHONY: all assets pdf check clean

all: assets pdf check

assets:
	cd .. && $(PY) paper/scripts/make_assets.py

pdf:
	pdflatex -interaction=nonstopmode -halt-on-error main > /dev/null
	bibtex main > /dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main > /dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main > /dev/null

check:
	$(PY) scripts/check_build.py

clean:
	rm -f *.aux *.log *.out *.spl *.bbl *.blg *.toc
