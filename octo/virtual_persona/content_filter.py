"""Content filter for VP — sanitizes incoming messages before LLM processing.

Regex-based, no external deps. Strips PII, credentials, scripts, and
control characters to reduce content-policy violations from Bedrock/Anthropic.
"""
from __future__ import annotations

import logging
import re
import string

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns — compiled once at import time
# ---------------------------------------------------------------------------

# PII patterns
_RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_RE_CREDIT_CARD = re.compile(
    r"\b(?:\d[ -]?){13,19}\b"  # 13–19 digits with optional separators
)
_RE_PHONE_INTL = re.compile(
    r"\+\d{1,3}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{2,4}[\s.-]?\d{2,4}[\s.-]?\d{0,4}"
)

# Credential patterns
_RE_BEARER = re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.I)
_RE_API_KEY = re.compile(
    r"(?:api[_-]?key|secret|password|token|authorization)\s*[:=]\s*\S+",
    re.I,
)
_RE_SK_TOKEN = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")
_RE_BASE64_BLOCK = re.compile(
    r"[A-Za-z0-9+/]{500,}={0,3}"  # Very long base64 blocks (>500 chars)
    # Only strips truly massive blobs — preserves shorter b64 that may be
    # meaningful (e.g. small inline images, encoded config snippets).
    # data: URIs are handled separately by _RE_DATA_URI.
)

# HTML/script patterns (post-strip-html residuals)
_RE_SCRIPT = re.compile(r"<script[\s\S]*?</script>", re.I)
_RE_JS_URI = re.compile(r"javascript\s*:", re.I)
_RE_DATA_URI = re.compile(r"data:[a-z]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+", re.I)

# Control characters (keep \n, \r, \t)
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Max length for content sent to LLM
_MAX_CONTENT_LENGTH = 4000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sanitize(text: str) -> tuple[str, list[str]]:
    """Sanitize text for safe LLM consumption.

    Returns (cleaned_text, list_of_applied_filters).
    Filters clean content but never reject — use ``is_safe_for_llm`` to reject.
    """
    if not text:
        return text, []

    actions: list[str] = []
    result = text

    # 1. Script/HTML cleanup
    if _RE_SCRIPT.search(result):
        result = _RE_SCRIPT.sub("[script removed]", result)
        actions.append("script_removed")
    if _RE_JS_URI.search(result):
        result = _RE_JS_URI.sub("[js-uri removed]", result)
        actions.append("js_uri_removed")

    # 2. Data URIs (large inline binary)
    if _RE_DATA_URI.search(result):
        result = _RE_DATA_URI.sub("[data-uri removed]", result)
        actions.append("data_uri_removed")

    # 3. PII masking
    if _RE_SSN.search(result):
        result = _RE_SSN.sub("[SSN]", result)
        actions.append("ssn_masked")
    if _RE_CREDIT_CARD.search(result):
        # Validate Luhn-like pattern (at least 13 consecutive digits after removing separators)
        def _mask_cc(m: re.Match) -> str:
            digits = re.sub(r"[ -]", "", m.group())
            if len(digits) >= 13 and digits.isdigit():
                return f"[card ending {digits[-4:]}]"
            return m.group()

        new_result = _RE_CREDIT_CARD.sub(_mask_cc, result)
        if new_result != result:
            result = new_result
            actions.append("credit_card_masked")
    if _RE_PHONE_INTL.search(result):
        result = _RE_PHONE_INTL.sub("[phone]", result)
        actions.append("phone_masked")

    # 4. Credential stripping
    if _RE_BEARER.search(result):
        result = _RE_BEARER.sub("[bearer-token]", result)
        actions.append("bearer_stripped")
    if _RE_SK_TOKEN.search(result):
        result = _RE_SK_TOKEN.sub("[api-key]", result)
        actions.append("sk_token_stripped")
    if _RE_API_KEY.search(result):
        result = _RE_API_KEY.sub("[credential]", result)
        actions.append("api_key_stripped")
    if _RE_BASE64_BLOCK.search(result):
        result = _RE_BASE64_BLOCK.sub("[base64-data removed]", result)
        actions.append("base64_block_removed")

    # 5. Control character removal
    if _RE_CONTROL.search(result):
        result = _RE_CONTROL.sub("", result)
        actions.append("control_chars_removed")

    # 6. Length cap
    if len(result) > _MAX_CONTENT_LENGTH:
        result = result[:_MAX_CONTENT_LENGTH] + "... [truncated]"
        actions.append(f"truncated_to_{_MAX_CONTENT_LENGTH}")

    if actions:
        logger.info("Content filter applied: %s", ", ".join(actions))

    return result, actions


def is_safe_for_llm(text: str) -> tuple[bool, str]:
    """Quick pre-check: is this text worth sending to an LLM?

    Returns (safe, reason). Returns False only for content that is
    fundamentally not processable — not for content that just needs cleaning.

    This check runs on **aggregated** queries (multiple messages combined),
    so thresholds are intentionally lenient. Short acknowledgments like
    "ок", "да", emoji reactions etc. are fine — the LLM confidence scorer
    will handle irrelevant content.
    """
    if not text or not text.strip():
        return False, "empty"

    stripped = text.strip()

    # Check for readable content — only reject truly binary/garbage content.
    # Must have actual letters (any script), digits, or spaces.
    # Only applies to longer texts (>200 chars) to avoid rejecting short
    # messages in any language. Threshold is very low (15%) — even content
    # with lots of special chars / markup usually has some readable parts.
    if len(stripped) > 200:
        word_chars = sum(1 for c in stripped if c.isalpha() or c.isdigit() or c.isspace())
        if word_chars / len(stripped) < 0.15:
            return False, "low_readable_ratio"

    return True, ""
