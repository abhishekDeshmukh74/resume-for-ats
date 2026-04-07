"""Shared LLM instance and helpers for all agents."""

from __future__ import annotations

import json
import logging
import os
import re
import time

import litellm

# Silence litellm's verbose logging unless explicitly enabled
litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# Thin LangChain-compatible wrapper around litellm.completion()
# ---------------------------------------------------------------------------


class _LLMResponse:
    """Minimal response object matching LangChain's AIMessage interface."""

    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


class _LiteLLMChat:
    """Drop-in replacement for LangChain ChatGroq/ChatGemini/ChatOllama.

    Only the ``.invoke(messages)`` method is implemented — that's all the
    pipeline ever calls.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        num_retries: int = 2,
        **extra,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.extra = extra

    def invoke(self, messages):
        msgs = []
        for m in messages:
            if isinstance(m, dict):
                msgs.append({"role": m["role"], "content": m["content"]})
            else:
                # LangChain message objects
                msgs.append({"role": getattr(m, "type", "user"), "content": m.content})
        response = litellm.completion(
            model=self.model,
            messages=msgs,
            api_key=self.api_key,
            api_base=self.api_base,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            num_retries=self.num_retries,
            **self.extra,
        )
        return _LLMResponse(response.choices[0].message.content)

    def with_fallbacks(self, others: list[_LiteLLMChat]) -> _LiteLLMChatWithFallbacks:
        return _LiteLLMChatWithFallbacks(self, others)


class _LiteLLMChatWithFallbacks:
    """Try primary LLM, then fall through to alternatives on any error."""

    def __init__(self, primary: _LiteLLMChat, fallbacks: list[_LiteLLMChat]):
        self._chain = [primary, *fallbacks]

    def invoke(self, messages):
        last_exc: Exception | None = None
        for llm in self._chain:
            try:
                return llm.invoke(messages)
            except Exception as exc:
                last_exc = exc
                _logger.warning("LiteLLM fallback: %s failed (%s), trying next.", llm.model, exc)
        raise RuntimeError(f"All LLM fallbacks exhausted: {last_exc}") from last_exc

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


# ── Provider-to-LiteLLM model mapping ────────────────────────────────────
# Each provider function returns a _LiteLLMChat (or fallbacks wrapper).

def _build_groq_llm() -> _LiteLLMChat | _LiteLLMChatWithFallbacks:
    keys = _get_groq_api_keys()
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    litellm_model = f"groq/{model}"
    _logger.info("Initialising LiteLLM (Groq) with %d API key(s), model=%s", len(keys), litellm_model)
    llms = [
        _LiteLLMChat(litellm_model, api_key=key)
        for key in keys
    ]
    return llms[0].with_fallbacks(llms[1:]) if len(llms) > 1 else llms[0]


def _build_gemini_llm() -> _LiteLLMChat:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    litellm_model = f"gemini/{model}"
    _logger.info("Initialising LiteLLM (Gemini): model=%s", litellm_model)
    return _LiteLLMChat(litellm_model, api_key=api_key)


def _build_ollama_llm() -> _LiteLLMChat:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    litellm_model = f"ollama/{model}"
    _logger.info("Initialising LiteLLM (Ollama): model=%s base_url=%s", litellm_model, base_url)
    return _LiteLLMChat(litellm_model, api_base=base_url)


def _build_openai_llm() -> _LiteLLMChat:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    _logger.info("Initialising LiteLLM (OpenAI): model=%s", model)
    return _LiteLLMChat(model, api_key=api_key)


def _build_anthropic_llm() -> _LiteLLMChat:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    litellm_model = f"anthropic/{model}"
    _logger.info("Initialising LiteLLM (Anthropic): model=%s", litellm_model)
    return _LiteLLMChat(litellm_model, api_key=api_key)


def _build_deepseek_llm() -> _LiteLLMChat:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set.")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    litellm_model = f"deepseek/{model}"
    _logger.info("Initialising LiteLLM (DeepSeek): model=%s", litellm_model)
    return _LiteLLMChat(litellm_model, api_key=api_key)


def _build_openrouter_llm() -> _LiteLLMChat:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")
    model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3-70b")
    litellm_model = f"openrouter/{model}"
    _logger.info("Initialising LiteLLM (OpenRouter): model=%s", litellm_model)
    return _LiteLLMChat(litellm_model, api_key=api_key)


_PROVIDER_BUILDERS = {
    "groq": _build_groq_llm,
    "gemini": _build_gemini_llm,
    "ollama": _build_ollama_llm,
    "openai": _build_openai_llm,
    "anthropic": _build_anthropic_llm,
    "deepseek": _build_deepseek_llm,
    "openrouter": _build_openrouter_llm,
}


def get_llm():
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    provider = os.environ.get("LLM_PROVIDER", "groq").lower()

    builder = _PROVIDER_BUILDERS.get(provider)
    if builder is None:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER={provider!r}. "
            f"Supported: {', '.join(sorted(_PROVIDER_BUILDERS))}"
        )

    _llm_instance = builder()
    return _llm_instance
