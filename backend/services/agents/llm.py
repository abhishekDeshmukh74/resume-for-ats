"""Shared LLM instance, JSON parsing, and prompt-injection sanitisation.

This module is imported by **every** agent node that calls an LLM.  It provides:

* ``get_llm()`` — singleton LLM with failover across multiple Groq API keys
  or a single Gemini instance.
* ``invoke_llm_json()`` — convenience wrapper that invokes the LLM, parses the
  response as JSON with automatic fence stripping and repair, and retries on
  failure (up to 2 extra attempts with exponential back-off).
* ``sanitize_input()`` — strips common prompt-injection patterns from user-
  supplied resume and JD text before it is embedded in LLM prompts.

Centralising the LLM here ensures consistent temperature (0.2), max tokens
(8 192), and retry/fallback behaviour across the pipeline.

Callers:
    intake_parser, jd_analyzer, gap_analyzer, bullet_rewriter,
    summary_optimizer, skills_optimizer, truth_guard, scorer, critic,
    formatter.
"""

from __future__ import annotations

import json
import logging
import os
import re

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------------
# Prompt-injection sanitisation (security — these patterns are necessary)
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
    re.compile(r"output\s+(your|the)\s+(system|initial)\s+prompt", re.I),
]


def sanitize_input(text: str) -> str:
    """Strip prompt-injection patterns from user-supplied text.

    Applied to both ``resume_text`` and ``jd_text`` at the intake stage
    (``parse_resume_node`` and ``analyze_jd_node``) before they are embedded
    in LLM prompts.

    The patterns cover common injection vectors: instruction override commands,
    role-play directives, special token leaks (``[INST]``, ``<|im_start|>``),
    and prompt exfiltration requests.

    Args:
        text: Raw user-supplied string (resume or JD).

    Returns:
        Cleaned string with matched injection patterns removed.
    """
    for pat in _INJECTION_PATTERNS:
        text = pat.sub("", text)
    return text


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _repair_json(text: str) -> dict:
    """Best-effort repair of malformed LLM JSON output.

    Handles the two most common LLM JSON issues:
        1. **Trailing commas** before ``]`` or ``}`` — removed with regex.
        2. **Truncated output** (unclosed brackets/braces) — appends the
           missing closing characters.

    Called only when ``json.loads`` fails in ``parse_llm_json()``.

    Args:
        text: Raw JSON-ish string after fence stripping.

    Returns:
        Parsed ``dict``.

    Raises:
        json.JSONDecodeError: If repair heuristics cannot salvage the text.
    """
    # Remove trailing commas before ] or }
    repaired = re.sub(r",\s*([\]}])", r"\1", text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Close unclosed brackets/braces (handles truncated output)
    stripped = repaired.rstrip(",\n\r\t ")
    opens = stripped.count("{") - stripped.count("}")
    opens_sq = stripped.count("[") - stripped.count("]")
    if opens >= 0 and opens_sq >= 0:
        candidate = stripped + ("]" * opens_sq) + ("}" * opens)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not parse LLM JSON output", text, 0)


def parse_llm_json(text) -> dict:
    """Parse JSON from LLM output, stripping markdown fences if present.

    Handles the full spectrum of LLM response formats:
        * Clean JSON.
        * JSON wrapped in \`\`\`json ... \`\`\` fences.
        * Multi-part content lists (Gemini sometimes returns these).
        * Empty / None responses (raises ``ValueError``).

    Falls back to ``_repair_json()`` if standard ``json.loads`` fails.

    Args:
        text: Raw ``response.content`` from the LLM — may be ``str``,
              ``list``, or ``None``.

    Returns:
        Parsed ``dict``.

    Raises:
        ValueError: If ``text`` is ``None`` or empty.
        json.JSONDecodeError: If all parsing and repair attempts fail.
    """
    if text is None:
        raise ValueError("LLM returned no content (None).")
    if isinstance(text, list):
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
    """Invoke the LLM, parse response as JSON, and retry on failure.

    This is the primary function used by every agent node that needs
    structured data from the LLM.  It encapsulates the full call cycle:

        1. Call ``get_llm().invoke(messages)``.
        2. Parse ``response.content`` via ``parse_llm_json()``.
        3. On ``ValueError`` or ``JSONDecodeError``, log a warning and
           retry after exponential back-off (1 s, 2 s).
        4. After all attempts exhausted, raise ``RuntimeError``.

    Args:
        messages: List of LangChain message dicts/objects for the LLM.
        retries:  Number of **extra** attempts beyond the first (default 2,
                  so 3 total).

    Returns:
        Parsed ``dict`` from the LLM’s JSON response.

    Raises:
        RuntimeError: If all ``1 + retries`` attempts fail.
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
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"LLM failed to return valid JSON after {1 + retries} attempts: {last_exc}"
    ) from last_exc


_logger = logging.getLogger(__name__)

_llm_instance = None


def _get_groq_api_keys() -> list[str]:
    """Read Groq API keys from environment variables.

    Supports two formats:
        * ``GROQ_API_KEYS`` — comma-separated list for multi-key failover.
        * ``GROQ_API_KEY``  — single key fallback.

    The list order determines failover priority: ``get_llm()`` builds a
    LangChain ``.with_fallbacks()`` chain from these keys.

    Returns:
        Non-empty list of API key strings.

    Raises:
        RuntimeError: If neither env var is set.
    """
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
    """Return the singleton LLM instance (lazy-initialised on first call).

    Supports two providers via the ``LLM_PROVIDER`` env var:

    * ``"groq"`` (default) — ``ChatGroq`` with ``llama-3.3-70b-versatile``.
      If multiple API keys are provided, builds a ``.with_fallbacks()``
      chain so requests automatically fail over to the next key on rate
      limits or transient errors.
    * ``"gemini"`` — ``ChatGoogleGenerativeAI`` with ``gemini-2.0-flash``.

    Shared settings across providers:
        * ``temperature = 0.2`` (low creativity for structured outputs).
        * ``max_tokens / max_output_tokens = 8192``.

    Returns:
        A LangChain chat model instance.

    Raises:
        RuntimeError: If the required API key env var is not set.
    """
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

    keys = _get_groq_api_keys()
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    _logger.info("Initialising Groq LLM with %d API key(s)", len(keys))

    llms = [
        ChatGroq(model=model, api_key=key, temperature=0.2, max_tokens=8192)
        for key in keys
    ]

    _llm_instance = llms[0].with_fallbacks(llms[1:]) if len(llms) > 1 else llms[0]
    return _llm_instance
