import base64
import re

import fitz  # PyMuPDF


def _normalise_text(text: str) -> str:
    """Fix common PDF extraction spacing artefacts."""
    # Collapse runs of spaces / tabs to a single space (keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Insert a space between a lowercase letter and an uppercase letter
    # when they are directly adjacent (e.g. "focusedFull" → "focused Full").
    # Guard against intentional camelCase by only doing this when the uppercase
    # is followed by a lowercase (i.e. a new word starts).
    text = re.sub(r"([a-z])([A-Z][a-z])", r"\1 \2", text)

    # Insert a space after punctuation directly followed by a non-space word
    # character, but NOT inside URLs / emails / decimals.
    text = re.sub(r"([,;:])([^\s,;:/\\@\d])", r"\1 \2", text)

    # Trim trailing whitespace from each line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text


def parse_pdf(file_bytes: bytes) -> tuple[str, str, str, str]:
    """Extract plain text and styled HTML from PDF bytes using PyMuPDF.

    The HTML output positions text spans to approximate the original layout
    with font size, family, bold/italic styling, and colour.
    The base64-encoded PDF is also returned for iframe preview.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts: list[str] = []
    html_pages: list[str] = []

    for i, page in enumerate(doc):
        # Plain text — layout mode preserves column alignment
        text_parts.append(page.get_text("text") or "")

        # Styled HTML via span dict
        page_dict = page.get_text("dict")
        pw = page.rect.width
        ph = page.rect.height
        parts: list[str] = []

        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "")
                    if not txt or not txt.strip():
                        continue

                    x   = span["origin"][0]
                    y   = span["origin"][1]
                    top = ph - y
                    sz  = float(span.get("size", 11.0))

                    # Colour: fitz packs RGB into a single int
                    c     = int(span.get("color", 0))
                    r_hex = (c >> 16) & 0xFF
                    g_hex = (c >>  8) & 0xFF
                    b_hex =  c        & 0xFF
                    color_css = f"#{r_hex:02x}{g_hex:02x}{b_hex:02x}"

                    # Font family heuristic from font name
                    fname  = span.get("font", "").lower()
                    flags  = int(span.get("flags", 0))
                    if any(s in fname for s in ("arial", "helvetica", "sans", "calibri", "verdana")):
                        family = "sans-serif"
                    elif any(s in fname for s in ("courier", "mono", "consolas")):
                        family = "monospace"
                    else:
                        family = "serif"

                    bold   = bool(flags & (1 << 4)) or "bold" in fname or "heavy" in fname
                    italic = bool(flags & (1 << 1)) or "italic" in fname or "oblique" in fname

                    esc    = (txt.replace("&", "&amp;")
                                 .replace("<", "&lt;")
                                 .replace(">", "&gt;"))
                    weight = "bold" if bold else "normal"
                    fstyle = "italic" if italic else "normal"

                    parts.append(
                        f'<span style="position:absolute;left:{x:.1f}px;top:{top:.1f}px;'
                        f"font-size:{sz:.1f}px;font-family:{family};"
                        f"font-weight:{weight};font-style:{fstyle};"
                        f'color:{color_css};">{esc}</span>'
                    )

        if parts:
            page_html = (
                f'<div style="position:relative;width:{pw:.0f}px;height:{ph:.0f}px;">'
                + "".join(parts)
                + "</div>"
            )
        else:
            esc = (text_parts[-1].replace("&", "&amp;")
                   .replace("<", "&lt;").replace(">", "&gt;"))
            page_html = esc.replace("\n", "<br>\n")

        html_pages.append(
            f'<div class="pdf-page" id="page-{i + 1}" '
            f'style="position:relative;margin-bottom:24px;">{page_html}</div>'
        )

    text = _normalise_text("\n".join(text_parts))
    html = (
        '<div class="pdf-document" '
        'style="font-family:sans-serif;background:#f4f4f4;padding:16px;">'
        + "\n".join(html_pages)
        + "</div>"
    )
    return text, html, base64.b64encode(file_bytes).decode(), "pdf"
