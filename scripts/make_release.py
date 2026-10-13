#!/usr/bin/env python
"""Package the submission.

    python scripts/make_release.py [--out release]

Writes, with a SHA-256 manifest inside each archive:
  paper/latex_source.zip                     the LaTeX source the publisher needs
  <out>/terranet-submission-<date>.zip       manuscript, cover letter, highlights,
                                             code, configs, tests, result tables,
                                             processed tiles and splits
  <out>/terranet-corpora-<date>.zip          the ray-traced corpora (data/raw)
Nothing is copied from outside the repository, and build products, virtual
environments and caches are never included.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".git", ".venv", ".venv-rt", "__pycache__", ".pytest_cache", ".ruff_cache",
                ".mypy_cache", "predictions", "cache", "models", "hurdle", "logs", "release"}
EXCLUDE_SUFFIX = {".pyc", ".aux", ".log", ".out", ".spl", ".blg", ".bbl", ".toc", ".fls",
                  ".fdb_latexmk", ".ply"}
SUBMISSION = ["README.md", "DATA.md", "LICENSE", "CITATION.cff", "Makefile", "pyproject.toml",
              "paper/main.pdf", "paper/latex_source.zip", "paper/cover_letter.md",
              "paper/highlights.txt", "src", "scripts", "tests", "configs", "docs",
              "outputs/tables", "outputs/sionna_7p5/tables", "data/processed", "data/splits"]
LATEX = ["main.tex", "main.bbl", "references.bib", "highlights.txt", "sections", "generated",
         "figures"]


def files_under(root: Path):
    if root.is_file():
        yield root
        return
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in EXCLUDE_DIRS)
        for fn in sorted(fns):
            p = Path(dp) / fn
            if p.suffix in EXCLUDE_SUFFIX or fn.startswith("."):
                continue
            yield p


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_zip(out: Path, base: Path, members, prefix: str):
    manifest = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for m in members:
            for p in files_under(base / m):
                arc = f"{prefix}/{p.relative_to(base).as_posix()}"
                z.write(p, arc)
                manifest.append(f"{sha256(p)}  {arc}")
        z.writestr(f"{prefix}/MANIFEST.sha256", "\n".join(manifest) + "\n")
    return len(manifest), out.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release")
    ap.add_argument("--no-corpora", action="store_true")
    args = ap.parse_args()
    date = dt.date.today().isoformat().replace("-", "")
    out = REPO / args.out
    out.mkdir(exist_ok=True)

    paper = REPO / "paper"
    for req in ("main.pdf", "main.bbl"):
        if not (paper / req).exists():
            raise SystemExit(f"paper/{req} missing: build the manuscript first (make -C paper)")
    n, size = build_zip(paper / "latex_source.zip", paper, LATEX, "latex_source")
    print(f"paper/latex_source.zip: {n} files, {size / 1e6:.1f} MB")

    n, size = build_zip(out / f"terranet-submission-{date}.zip", REPO, SUBMISSION,
                        f"terranet-submission-{date}")
    print(f"{out.name}/terranet-submission-{date}.zip: {n} files, {size / 1e6:.1f} MB")
    if not args.no_corpora:
        n, size = build_zip(out / f"terranet-corpora-{date}.zip", REPO, ["data/raw", "DATA.md"],
                            f"terranet-corpora-{date}")
        print(f"{out.name}/terranet-corpora-{date}.zip: {n} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
