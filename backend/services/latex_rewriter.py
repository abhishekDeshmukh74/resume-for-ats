"""
latex_rewriter.py — Apply AI text replacements to a .tex source and compile
                    it to PDF using xelatex (falls back to pdflatex/lualatex).

Flow:
  1. Decode the original .tex source from bytes.
  2. For each TextReplacement(old, new): replace the first occurrence of
     'old' in the source (the AI generates 'old' values that are verbatim
     substrings of the source text).
  3. Write the patched source to a temp directory.
  4. Invoke xelatex twice (first pass lays out, second resolves references).
  5. Return the compiled PDF bytes.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata

from backend.models import ResumeData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text sanitisation
# ---------------------------------------------------------------------------

_OUTPUT_CHAR_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",             # smart single quotes
    "\u201c": '"', "\u201d": '"',             # smart double quotes
    "\u2013": "-", "\u2014": "-",             # en/em dash
    "\u2026": "...",                           # ellipsis
    "\u00a0": " ",                             # non-breaking space
    "\u200b": "", "\u200c": "", "\u200d": "",  # zero-width chars
    "\ufeff": "",                              # BOM
})


def _sanitize(text: str) -> str:
    """Normalise replacement text to safe Unicode."""
    return unicodedata.normalize("NFKC", text).translate(_OUTPUT_CHAR_MAP)


def _latex_escape(text: str) -> str:
    r"""Escape bare %% and & in plain text to their LaTeX equivalents (\\%% and \\&)."""
    text = re.sub(r"(?<!\\)%", r"\\%", text)
    text = re.sub(r"(?<!\\)&", r"\\&", text)
    return text


# ---------------------------------------------------------------------------
# Compiler detection
# ---------------------------------------------------------------------------

# Well-known MiKTeX / TeX Live installation directories (Windows + Linux/Mac)
_MIKTEX_HINTS = [
    r"C:\Users\{user}\AppData\Local\Programs\MiKTeX\miktex\bin\x64",
    r"C:\Program Files\MiKTeX\miktex\bin\x64",
    r"C:\Program Files (x86)\MiKTeX\miktex\bin",
    r"/usr/bin",
    r"/usr/local/bin",
    r"/Library/TeX/texbin",
]


def _find_compiler() -> str:
    """Return the full path to the first available LaTeX compiler."""
    import getpass

    # 1. Explicit override via environment variable
    override = os.environ.get("LATEX_COMPILER_PATH")
    if override and os.path.isfile(override):
        return override

    # 2. Standard PATH lookup
    for compiler in ("xelatex", "pdflatex", "lualatex"):
        found = shutil.which(compiler)
        if found:
            return found

    # 3. Probe known install locations (handles MiKTeX user install
    #    whose PATH update only takes effect in a new login shell,
    #    not in a long-running uvicorn process)
    try:
        username = getpass.getuser()
    except Exception:
        username = ""

    for hint in _MIKTEX_HINTS:
        hint = hint.replace("{user}", username)
        for compiler in ("xelatex", "pdflatex", "lualatex"):
            for ext in (".exe", ""):
                candidate = os.path.join(hint, compiler + ext)
                if os.path.isfile(candidate):
                    return candidate

    raise RuntimeError(
        "No LaTeX compiler found. "
        "Install MiKTeX (https://miktex.org) or TeX Live and ensure "
        "`xelatex` is on your PATH, or set the LATEX_COMPILER_PATH env var."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_tex(tex_bytes: bytes, resume: ResumeData) -> bytes:
    """Apply AI replacements to .tex source and return compiled PDF bytes."""
    real = [r for r in resume.replacements if r.old and r.new and r.old != r.new]

    tex_source = tex_bytes.decode("utf-8", errors="replace")
    total_matched = 0

    for repl in real:
        new_text = _latex_escape(_sanitize(repl.new))
        if repl.old in tex_source:
            tex_source = tex_source.replace(repl.old, new_text, 1)
            total_matched += 1
        else:
            # Fallback: AI was given plain text where \% appeared as %;
            # re-escape % → \% to find the match in the .tex source.
            escaped_old = _latex_escape(repl.old)
            if escaped_old != repl.old and escaped_old in tex_source:
                tex_source = tex_source.replace(escaped_old, new_text, 1)
                total_matched += 1
                logger.debug("LaTeX rewriter: matched via escape for '%s…'", repl.old[:60])
            else:
                logger.debug("LaTeX rewriter: no match for '%s…'", repl.old[:60])

    logger.info(
        "LaTeX rewriter: matched %d / %d replacements.",
        total_matched, len(real),
    )

    return _compile(tex_source)


def _compile(tex_source: str) -> bytes:
    """Write *tex_source* to a temp dir, run xelatex twice, return PDF bytes."""
    compiler = _find_compiler()
    logger.info("Compiling LaTeX with %s", compiler)

    # XeTeX uses fontspec for font handling; fontenc with T1 encoding
    # causes "Corrupted NFSS tables" errors.  Replace fontenc with fontspec
    # when compiling with xelatex.
    compiler_basename = os.path.basename(compiler).lower()
    if "xelatex" in compiler_basename:
        tex_source = re.sub(
            r"\\usepackage\s*(\[[^\]]*\])?\s*\{fontenc\}",
            r"\\usepackage{fontspec}",
            tex_source,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "resume.tex")
        with open(tex_path, "w", encoding="utf-8") as fh:
            fh.write(tex_source)

        # MiKTeX supports --enable-installer to auto-fetch missing packages.
        # TeX Live ignores unknown options less gracefully, so only add it
        # when we can positively identify MiKTeX by the compiler path.
        is_miktex = "miktex" in compiler.lower()
        compile_args = [
            compiler,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={tmpdir}",
        ]
        if is_miktex:
            compile_args.append("--enable-installer")
        compile_args.append(tex_path)

        last_result = None
        for pass_num in range(1, 3):  # two passes for cross-refs / TOC
            last_result = subprocess.run(
                compile_args,
                capture_output=True,
                timeout=120,
                cwd=tmpdir,
            )
            logger.debug(
                "xelatex pass %d exit code: %d", pass_num, last_result.returncode,
            )

        pdf_path = os.path.join(tmpdir, "resume.pdf")
        if not os.path.exists(pdf_path):
            assert last_result is not None
            stderr = last_result.stderr.decode("utf-8", errors="replace")
            stdout = last_result.stdout.decode("utf-8", errors="replace")
            # Surface the last ~40 lines of output for diagnosis
            log_tail = "\n".join((stdout + "\n" + stderr).splitlines()[-40:])
            raise RuntimeError(
                f"LaTeX compilation failed (compiler: {compiler}).\n{log_tail}"
            )

        with open(pdf_path, "rb") as fh:
            return fh.read()
