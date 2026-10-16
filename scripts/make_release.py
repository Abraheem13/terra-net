#!/usr/bin/env python
"""Package the submission.

    python scripts/make_release.py [--out release]

Writes:
  paper/latex_source.zip                     the LaTeX source for the publisher's
                                             submission system: one flat folder with
                                             a single self-contained manuscript.tex,
                                             its .bbl and .bib, the figure PDFs and
                                             the Elsevier class and style files
  <out>/terranet-submission-<date>.zip       manuscript, cover letter, highlights,
                                             code, configs, tests, result tables,
                                             processed tiles and splits
  <out>/terranet-corpora-<date>.zip          the ray-traced corpora (data/raw)
The two repository archives carry a SHA-256 manifest.
Nothing is copied from outside the repository, and build products, virtual
environments and caches are never included.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import shutil
import subprocess
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
MANUSCRIPT = "manuscript"           # name of the single .tex file in the LaTeX package
ZIP_TIME = (2026, 1, 1, 0, 0, 0)     # fixed timestamp: the package is byte-reproducible


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


def flatten(paper: Path) -> tuple[str, list[Path]]:
    """main.tex with every \\input expanded in place and every figure path reduced to
    its file name. Full-line comments are dropped (TeX ignores them together with
    their line end, so the typeset result is unchanged). Returns the text and the
    figure files it uses."""
    figures: list[Path] = []

    def expand(path: Path) -> str:
        out = []
        for line in path.read_text().splitlines(keepends=True):
            if line.lstrip().startswith("%"):
                continue
            m = re.search(r"\\input\{([^}]+)\}", line)
            if m and not line[:m.start()].rstrip().endswith("%"):
                # every \\input stands alone on its line and ends with "%", so TeX sees the
                # file's lines followed by nothing; reproduce exactly that token stream
                if line[:m.start()].strip() or line[m.end():].rstrip("\n") != "%":
                    raise SystemExit(f"{path.name}: write '\\input{{...}}%' on its own line")
                target = paper / m.group(1)
                target = target if target.suffix == ".tex" else target.with_suffix(".tex")
                out.append(expand(target).rstrip("\n") + "\n%\n")
                continue
            for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", line):
                figures.append(paper / g)
                line = line.replace("{" + g + "}", "{" + Path(g).name + "}")
            out.append(line)
        return "".join(out)

    text = expand(paper / "main.tex")
    names = [f.name for f in figures]
    assert len(names) == len(set(names)), "two figures share a file name"
    assert "\\input{" not in text, "an \\input was left unexpanded"
    return text, figures


def build_latex_zip(paper: Path, out: Path) -> tuple[int, int]:
    """Flat LaTeX package: every file at the root of the archive, no subfolders."""
    text, figures = flatten(paper)
    header = ("% Manuscript submitted to Computer Communications.\n"
              "% Self-contained source: run pdflatex three times from a clean folder; the\n"
              f"% bibliography is supplied both as {MANUSCRIPT}.bbl and as references.bib.\n")
    kpse = {n: Path(subprocess.check_output(["kpsewhich", n], text=True).strip())
            for n in ("elsarticle.cls", "elsarticle-num.bst")}
    members = [(f"{MANUSCRIPT}.tex", (header + text).encode()),
               (f"{MANUSCRIPT}.bbl", (paper / "main.bbl").read_bytes()),
               ("references.bib", (paper / "references.bib").read_bytes()),
               *[(n, p.read_bytes()) for n, p in kpse.items()],
               *[(f.name, f.read_bytes()) for f in figures]]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in members:
            info = zipfile.ZipInfo(name, ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    return len(members), out.stat().st_size


def check_latex_zip(zip_path: Path, work: Path) -> int:
    """Extract into an empty folder, compile with the supplied .bbl only, and fail on
    any error, undefined reference or citation. Returns the page count."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        assert all("/" not in n for n in z.namelist()), "the package must be flat"
        z.extractall(work)
    for _ in range(4):                     # until the cross-references settle
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            f"{MANUSCRIPT}.tex"], cwd=work, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"LaTeX package does not compile:\n{r.stdout[-2000:]}")
        log = (work / f"{MANUSCRIPT}.log").read_text(errors="ignore")
        if "Rerun to get" not in log:
            break
    for bad in ("undefined", "Rerun to get", "Missing character"):
        if bad in log:
            raise SystemExit(f"LaTeX package log reports '{bad}'")
    # the package must typeset exactly the reviewed manuscript (the footer date aside)
    def text(pdf: Path) -> list[str]:
        out = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
        return [ln for ln in out.splitlines() if "Preprint submitted to" not in ln]
    if text(work / f"{MANUSCRIPT}.pdf") != text(REPO / "paper" / "main.pdf"):
        raise SystemExit("the LaTeX package does not reproduce paper/main.pdf")
    info = subprocess.check_output(["pdfinfo", f"{MANUSCRIPT}.pdf"], cwd=work, text=True)
    return int(re.search(r"Pages:\s+(\d+)", info).group(1))


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
    n, size = build_latex_zip(paper, paper / "latex_source.zip")
    pages = check_latex_zip(paper / "latex_source.zip", out / "latex_check")
    shutil.rmtree(out / "latex_check")
    print(f"paper/latex_source.zip: {n} files, {size / 1e6:.1f} MB, "
          f"compiles stand-alone to {pages} pages")

    n, size = build_zip(out / f"terranet-submission-{date}.zip", REPO, SUBMISSION,
                        f"terranet-submission-{date}")
    print(f"{out.name}/terranet-submission-{date}.zip: {n} files, {size / 1e6:.1f} MB")
    if not args.no_corpora:
        n, size = build_zip(out / f"terranet-corpora-{date}.zip", REPO, ["data/raw", "DATA.md"],
                            f"terranet-corpora-{date}")
        print(f"{out.name}/terranet-corpora-{date}.zip: {n} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
