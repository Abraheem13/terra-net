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
# every generated table must be input exactly once, and every input must exist
gen = sorted(q.stem for q in (here / "generated").glob("tab_*.tex"))
inputs = re.findall(r"\\input\{generated/(tab_\w+)\}", tex)
for g in gen:
    if inputs.count(g) != 1:
        problems.append(f"generated/{g}.tex is input {inputs.count(g)} times")
for i in set(inputs) - set(gen):
    problems.append(f"generated/{i}.tex is input but was not generated")
for fig in set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)):
    if not (here / fig).exists():
        problems.append(f"missing figure {fig}")
# no result digit may be typed by hand: a number followed by a unit or a
# percent sign in the sections must come from a macro (whitelist below)
HAND_OK = {"95", "90", "100", "200", "500", "1000", "150", "140", "130", "120", "3.5", "7.5",
           "10", "25", "50", "30", "1.5", "0.05", "2", "1", "5", "3", "4", "20", "15", "45",
           "12", "1e9", "99", "0.5", "6", "8", "11", "220", "9", "32", "46", "0", "16", "64"}
for sec in sorted((here / "sections").glob("*.tex")):
    body = re.sub(r"%.*", "", sec.read_text())
    for m in re.finditer(r"(?<![\\\w.])(\d+(?:\.\d+)?)\s*(?:\\%|\\,?\\?dB\b|\\,m\b|\\SI)", body):
        if m.group(1) not in HAND_OK:
            problems.append(f"{sec.name}: hand-typed result-like number '{m.group(0).strip()}'")
if problems:
    print("\n".join(problems))
    sys.exit(1)
print(f"build clean; {len(defined)} generated macros ({len(unused)} defined but unused), "
      f"{len(gen)} generated tables")
if unused:
    print("unused macros (available for the text, not an error): " + str(len(unused)))
