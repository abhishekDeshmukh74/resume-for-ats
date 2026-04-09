# Parser Service — `parser.py`

## Purpose

Extracts plain text and styled HTML from an uploaded PDF resume using PyMuPDF.

## How It Works

1. Opens PDF from raw bytes using `fitz.open(stream=..., filetype="pdf")`.
2. For each page:
   - `page.get_text("text")` → raw plain text from the PDF.
   - `page.get_text("dict")` → span-level data used to build styled HTML with accurate font family, size, weight, style, and colour.
3. **Text normalisation** (`_normalise_text`):
   - Collapses runs of spaces/tabs to a single space (preserves newlines).
   - Inserts spaces between camelCase word boundaries (e.g. `focusedFull` → `focused Full`).
   - Inserts spaces after punctuation directly followed by a word character (but not inside URLs/emails/decimals).
   - Trims trailing whitespace from each line.
4. Returns `(text, html, file_b64, file_type)`.

## HTML Generation

Instead of using `page.get_text("html")` (which can produce inconsistent styling), the parser builds HTML from span dicts:

- Each text span is positioned absolutely within a page-sized container.
- Font family is inferred heuristically from the font name (sans-serif / monospace / serif).
- Bold/italic flags are read from span flags and font name patterns.
- Colour is extracted from the span's packed RGB integer.
- If no spans are found on a page, the plain text is used as a `<br>`-delimited fallback.

## Critical: Text Representation Consistency

The plain text returned here is sent to the multi-agent pipeline, where Agents 4a/4b/4c (Skills, Summary, Experience Rewriters) must return **verbatim substrings** of it as `old` values in the replacements array. Agent 5 (QA) validates these substrings exist. The rewriter service (`rewriter.py`) then uses `page.search_for()` to locate these `old` strings in the PDF.

## Returns

| Field | Type | Usage |
|-------|------|-------|
| `text` | `str` | Normalised plain text → sent to AI for analysis |
| `html` | `str` | Styled HTML (span-dict based) → frontend original preview |
| `file_b64` | `str` | Base64 original PDF → sent back to generate endpoint |
| `file_type` | `str` | Always `"pdf"` |
