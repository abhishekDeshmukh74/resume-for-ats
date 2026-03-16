"""Shared LLM instance and helpers for all agents."""

from __future__ import annotations

import json
import logging
import os
import re

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------------
# Prompt-injection sanitisation
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|above|prior)\s+instructions", re.I),
    re.compile(r"override\s+(all\s+)?(previous|above|prior)\s+instructions", re.I),
    re.compile(r"forget\s+(all\s+)?(previous|above|prior)\s+instructions", re.I),
    re.compile(r"you\s+are\s+now\b", re.I),
    re.compile(r"act\s+as\s+if\b", re.I),
    re.compile(r"pretend\s+you\s+are\b", re.I),
    re.compile(r"\bsystem\s*:", re.I),
    re.compile(r"\[/?INST\]", re.I),
    re.compile(r"<\|im_start\|>", re.I),
    re.compile(r"<\|im_end\|>", re.I),
    re.compile(r"<<\s*SYS\s*>>", re.I),
    re.compile(r"BEGIN\s+INSTRUCTION", re.I),
    re.compile(r"END\s+INSTRUCTION", re.I),
    re.compile(r"\bdo\s+not\s+follow\b.*\binstructions\b", re.I),
    re.compile(r"reveal\s+(your|the)\s+(system|initial)\s+prompt", re.I),
    re.compile(r"output\s+your\s+(system|initial)\s+prompt", re.I),
]


def _sanitize_user_input(text: str) -> str:
    """Strip prompt-injection patterns from user-supplied text.

    Removes substrings that attempt to override system instructions, inject
    role tokens, or extract the system prompt.  The cleaned text is returned.
    """
    for pat in _INJECTION_PATTERNS:
        text = pat.sub("", text)
    return text


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
# Matches a complete {"old": "...", "new": "..."} replacement object
_REPL_OBJ_RE = re.compile(
    r'\{"old"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*"new"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}',
    re.DOTALL,
)


def _repair_json(text: str) -> dict:
    """Best-effort repair of malformed/truncated LLM JSON output."""
    # Pass 1: strip trailing commas before ] or }
    repaired = re.sub(r",\s*([\]}])", r"\1", text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Pass 2: truncated JSON — close any unclosed brackets/braces.
    # Scan backwards to find the last position that, when closed, is valid.
    for end in range(len(repaired), max(len(repaired) - 200, 0), -1):
        chunk = repaired[:end]
        opens = chunk.count("{") - chunk.count("}")
        opens_sq = chunk.count("[") - chunk.count("]")
        if opens < 0 or opens_sq < 0:
            continue
        candidate = chunk.rstrip(",\n\r\t ") + ("]" * opens_sq) + ("}" * opens)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Pass 3: regex extraction — pull every complete replacement object out of
    # whatever the LLM produced, even if the outer structure is broken.
    objects = _REPL_OBJ_RE.findall(text)
    if objects:
        replacements = [json.loads(o) for o in objects]
        return {"replacements": replacements, "fixes_applied": ["(recovered via regex)"]}

    raise json.JSONDecodeError(
        f"Could not parse or repair LLM JSON output", text, 0
    )


def parse_llm_json(text) -> dict:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    if text is None:
        raise ValueError("LLM returned no content (None).")
    if isinstance(text, list):
        # Some providers return content as a list of content blocks
        text = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in text
        )
    text = text.strip()
    if not text:
        raise ValueError("LLM returned an empty response.")
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _repair_json(text)


def invoke_llm_json(messages: list, *, retries: int = 2) -> dict:
    """Invoke the shared LLM and return the response parsed as JSON.

    Retries up to *retries* times (with brief exponential back-off) when the
    model returns empty or unparseable content.
    """
    import time

    llm = get_llm()
    last_exc: Exception | None = None
    for attempt in range(1 + retries):
        try:
            resp = llm.invoke(messages)
            return parse_llm_json(resp.content)
        except (ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
            _logger.warning(
                "LLM JSON parse failed (attempt %d/%d): %s",
                attempt + 1, 1 + retries, exc,
            )
            if attempt < retries:
                time.sleep(2 ** attempt)  # 1 s → 2 s
    raise RuntimeError(
        f"LLM failed to return valid JSON after {1 + retries} attempts: {last_exc}"
    ) from last_exc


_logger = logging.getLogger(__name__)

_llm_instance = None


def _get_groq_api_keys() -> list[str]:
    """Read GROQ API keys from env. Supports comma-separated GROQ_API_KEYS
    with fallback to the single GROQ_API_KEY variable."""
    keys_csv = os.environ.get("GROQ_API_KEYS", "")
    if keys_csv:
        keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("GROQ_API_KEY", "")
    if single.strip():
        return [single.strip()]
    raise RuntimeError(
        "Neither GROQ_API_KEYS nor GROQ_API_KEY environment variable is set."
    )


def get_llm():
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    provider = os.environ.get("LLM_PROVIDER", "groq").lower()

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        _llm_instance = ChatGoogleGenerativeAI(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
            google_api_key=api_key,
            temperature=0.2,
            max_output_tokens=8192,
        )
        return _llm_instance

    # Default: groq — build one ChatGroq per API key and chain as fallbacks
    keys = _get_groq_api_keys()
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    _logger.info("Initialising Groq LLM with %d API key(s)", len(keys))

    llms = [
        ChatGroq(model=model, api_key=key, temperature=0.2, max_tokens=8192)
        for key in keys
    ]

    # with_fallbacks automatically tries next LLM on any error (incl. rate-limit)
    _llm_instance = llms[0].with_fallbacks(llms[1:]) if len(llms) > 1 else llms[0]
    return _llm_instance
