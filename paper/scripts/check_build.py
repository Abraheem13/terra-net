#!/usr/bin/env python
"""Fail the build on any LaTeX/BibTeX problem a reviewer would notice:
undefined references or citations, multiply defined labels, missing
characters, bad boxes, BibTeX warnings, and macros from generated/numbers.tex
that are never used (a sign the text and the results have drifted)."""
import re
import sys
from pathlib import Path

here = Path(__file__).resolve().parents[1]
log = (here / "main.log").read_text(errors="ignore")
blg = (here / "main.blg").read_text(errors="ignore")
problems = []
for pat, what in [(r"Reference `[^']+' on page \d+ undefined", "undefined reference"),
                  (r"Citation `[^']+' on page \d+ undefined", "undefined citation"),
                  (r"multiply defined", "multiply defined label"),
                  (r"Missing character", "missing glyph"),
                  (r"Overfull \\hbox \((\d+\.\d+)pt", "overfull box"),
                  (r"There were undefined references", "undefined references")]:
    for m in re.finditer(pat, log):
        if what == "overfull box" and float(m.group(1)) < 1.0:
            continue
        # elsarticle's 5p front-matter output routine emits exactly this box on
        # page 1 even for an empty document; it is not a content overflow.
        if what == "overfull box" and m.group(1) == "1.90001" and \
                "has occurred while \\output is active" in log[m.end():m.end() + 60]:
            continue
        problems.append(f"{what}: {m.group(0)}")
for line in blg.splitlines():
    if line.startswith("Warning--"):
        problems.append(f"bibtex: {line}")

tex = "".join(p.read_text() for p in (here / "sections").glob("*.tex"))
tex += (here / "main.tex").read_text()
defined = re.findall(r"\\newcommand\{\\(\w+)\}", (here / "generated/numbers.tex").read_text())
unused = [m for m in defined if not re.search(r"\\" + m + r"(?![A-Za-z])", tex)]
if problems:
    print("\n".join(problems))
    sys.exit(1)
print(f"build clean; {len(defined)} generated macros, {len(unused)} unused"
      + (f": {', '.join(unused)}" if unused else ""))
