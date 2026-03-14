# Rewriter Service — `rewriter.py`

## Purpose

Rewrites the original PDF resume **in-place** using AI-generated `{old, new}` replacement pairs.
The output PDF preserves the original layout, fonts, colours, and formatting — only the text content changes.

## How It Works

### Pipeline

1. **Receive** `file_bytes` (original PDF) + `ResumeData` (AI output with `.replacements` list).
2. **For each page**, build a font map by extracting embedded fonts from that page's font table.
3. **For each replacement**:
   - Use `page.search_for(old_text)` to locate all bounding-box rects for the old text (handles multi-line wrapping automatically).
   - If not found, retry with a sanitised variant (NFKC normalisation + character mapping).
   - Capture the font, font size, and colour from the text span at the first matched rect.
4. **Redact** all matched rects via `page.add_redact_annot()`.
5. **Apply** all redactions in a single `page.apply_redactions()` call.
6. **Re-insert** the new text at the position of the first (topmost) matched rect via `TextWriter`, using the captured font, size, and colour.
7. If zero replacements matched, return the original PDF unchanged (no corruption risk).

### Font Resolution

For each replacement, the rewriter attempts to use the exact font from the matched text span:

1. **Embedded font** — Extract the font data from the PDF by xref and create a `fitz.Font` from the buffer. This handles subset-prefixed names (e.g. `WRAHST+LMRoman10-Regular` → `LMRoman10-Regular`).
2. **Base-14 font** — If the span's font name matches a known Base-14 name (Helvetica, Times-Roman, Courier, etc.), use the corresponding PyMuPDF built-in.
3. **Flag-based fallback** — Based on the span's bold/italic flags, fall back to the appropriate Times variant (`tiro`, `tibo`, `tiit`, `tibi`).

### Text Sanitisation

Before insertion, replacement text is normalised via `_sanitize_text()`:
- Unicode NFKC normalisation
- Smart quotes → straight quotes
- Em/en dashes → hyphens
- Ellipsis → `...`
- Non-breaking spaces → regular spaces
- Zero-width characters and BOM removed

This prevents rendering failures when the target font lacks glyphs for fancy Unicode characters.

## Key Invariants

- **Never return a corrupt PDF** — if zero replacements match, return original bytes unchanged.
- **Redactions are batched** — all per-page redactions are queued, then applied in one `apply_redactions()` call before any text insertion.
- **Embedded fonts preferred** — Base-14 fallback only when font extraction fails.
- **One match per replacement** — `search_for()` returns all rects; all are redacted but new text is inserted only at the first rect.

## Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| PDF unchanged | AI `old` text doesn't match any page text | Check AI prompt; verify text normalisation; check logs |
| Text overlaps | Replacement text significantly longer than original | Prompt tells AI to keep ±20% length |
| Wrong font | Embedded font extraction failed | Falls back to Base-14 (Times/Helvetica) |
| Garbled chars | AI produced smart quotes / em-dashes the font can't render | `_sanitize_text` maps to ASCII equivalents before insertion |

## Dependencies

- **PyMuPDF (`fitz`)** — PDF manipulation (search_for, redactions, TextWriter, Font extraction)
- **`backend.models.ResumeData`** — Pydantic model with `.replacements: list[TextReplacement]`
