"""Tests for octo.virtual_persona.content_filter — regex patterns individually."""
from __future__ import annotations

import re

import pytest

from octo.virtual_persona.content_filter import (
    _MAX_CONTENT_LENGTH,
    _RE_API_KEY,
    _RE_BASE64_BLOCK,
    _RE_BEARER,
    _RE_CONTROL,
    _RE_CREDIT_CARD,
    _RE_DATA_URI,
    _RE_JS_URI,
    _RE_PHONE_INTL,
    _RE_SCRIPT,
    _RE_SK_TOKEN,
    _RE_SSN,
    is_safe_for_llm,
    sanitize,
)


# ======================================================================
# Individual regex tests
# ======================================================================


class TestSSNRegex:
    """_RE_SSN — Social Security Number pattern."""

    @pytest.mark.parametrize("ssn", [
        "123-45-6789",
        "000-00-0000",
        "999-99-9999",
    ])
    def test_matches_valid_ssns(self, ssn: str):
        assert _RE_SSN.search(ssn), f"Should match: {ssn}"

    @pytest.mark.parametrize("text", [
        "12-345-6789",       # wrong grouping
        "1234-56-789",       # wrong grouping
        "123456789",         # no dashes
        "123-45-678",        # too short last group
        "123-45-67890",      # too long last group
        "abc-de-fghi",       # letters
    ])
    def test_rejects_non_ssns(self, text: str):
        assert not _RE_SSN.search(text), f"Should NOT match: {text}"

    def test_ssn_in_sentence(self):
        text = "my SSN is 123-45-6789 please help"
        assert _RE_SSN.search(text)

    def test_ssn_at_boundaries(self):
        # Should match with word boundary
        assert _RE_SSN.search("SSN:123-45-6789.")
        # Should NOT match inside longer number
        assert not _RE_SSN.search("1123-45-67891")


class TestCreditCardRegex:
    """_RE_CREDIT_CARD — 13–19 digit patterns with separators."""

    @pytest.mark.parametrize("cc", [
        "4111111111111111",           # Visa 16-digit
        "4111 1111 1111 1111",        # Visa with spaces
        "4111-1111-1111-1111",        # Visa with dashes
        "5500000000000004",           # Mastercard
        "340000000000009",            # Amex (15 digits)
        "6011000000000004",           # Discover
        "3530111333300000",           # JCB
    ])
    def test_matches_card_patterns(self, cc: str):
        assert _RE_CREDIT_CARD.search(cc), f"Should match: {cc}"

    @pytest.mark.parametrize("text", [
        "12345",                # too short
        "123456789012",         # 12 digits — below minimum
        "abcdefghijklmnop",     # letters
    ])
    def test_rejects_non_cards(self, text: str):
        assert not _RE_CREDIT_CARD.search(text), f"Should NOT match: {text}"

    def test_sanitize_masks_card_with_last_four(self):
        text = "card 4111111111111111 on file"
        cleaned, actions = sanitize(text)
        assert "credit_card_masked" in actions
        assert "[card ending 1111]" in cleaned
        assert "4111111111111111" not in cleaned

    def test_sanitize_preserves_short_numbers(self):
        """Regular numbers like years or IDs should not be masked."""
        text = "order 12345 from 2024"
        cleaned, actions = sanitize(text)
        assert "credit_card_masked" not in actions
        assert cleaned == text


class TestPhoneRegex:
    """_RE_PHONE_INTL — international phone numbers."""

    @pytest.mark.parametrize("phone", [
        "+1 555 123 4567",
        "+44 20 7946 0958",
        "+380 50 123 4567",
        "+1-555-123-4567",
        "+1.555.123.4567",
        "+49 (30) 12345678",
    ])
    def test_matches_international_phones(self, phone: str):
        assert _RE_PHONE_INTL.search(phone), f"Should match: {phone}"

    @pytest.mark.parametrize("text", [
        "555-1234",             # no country code
        "call me at noon",      # no phone
        "+1",                   # too short
    ])
    def test_rejects_non_phones(self, text: str):
        assert not _RE_PHONE_INTL.search(text), f"Should NOT match: {text}"

    def test_phone_in_sentence(self):
        text = "reach me at +1 555 123 4567 anytime"
        cleaned, actions = sanitize(text)
        assert "phone_masked" in actions
        assert "[phone]" in cleaned


class TestBearerRegex:
    """_RE_BEARER — Bearer token patterns."""

    @pytest.mark.parametrize("token", [
        "Bearer eyJhbGciOiJIUzI1NiJ9",
        "bearer abc123def456",
        "BEARER some-token.with.dots",
        "Bearer AAAABBBBcccc+/dddd==",
    ])
    def test_matches_bearer_tokens(self, token: str):
        assert _RE_BEARER.search(token), f"Should match: {token}"

    def test_rejects_bare_bearer_word(self):
        # "Bearer" alone (no token after) — should not match
        assert not _RE_BEARER.search("Bearer ")

    def test_sanitize_strips_bearer(self):
        text = "header: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0"
        cleaned, actions = sanitize(text)
        assert "bearer_stripped" in actions
        assert "eyJ" not in cleaned
        assert "[bearer-token]" in cleaned


class TestSKTokenRegex:
    """_RE_SK_TOKEN — sk-* API key patterns."""

    @pytest.mark.parametrize("token", [
        "sk-abc123def456ghijklmnop",           # 24 chars after sk-
        "sk-ant12345678901234567890",           # anthropic-style
        "sk-proj1234567890abcdefghij",          # openai-style
        "sk-AABBCCDD11223344556677889900",      # mixed case
    ])
    def test_matches_sk_tokens(self, token: str):
        assert _RE_SK_TOKEN.search(token), f"Should match: {token}"

    @pytest.mark.parametrize("text", [
        "sk-short",                # too short (<20 chars after sk-)
        "sk-abc",                  # way too short
        "skeleton key",            # not a token
        "disk-operating",          # 'sk' in middle of word
    ])
    def test_rejects_non_tokens(self, text: str):
        assert not _RE_SK_TOKEN.search(text), f"Should NOT match: {text}"

    def test_sanitize_masks_sk_token(self):
        text = "use key sk-ant12345678901234567890 for auth"
        cleaned, actions = sanitize(text)
        assert "sk_token_stripped" in actions
        assert "[api-key]" in cleaned
        assert "sk-ant" not in cleaned


class TestAPIKeyRegex:
    """_RE_API_KEY — generic credential patterns."""

    @pytest.mark.parametrize("cred", [
        "api_key: abc123",
        "api-key = def456",
        "API_KEY=ghijkl",
        "secret: myS3cret!",
        "password: hunter2",
        "token: tok_12345",
        "authorization: Basic dXNlcjpwYXNz",
        "PASSWORD = superSecret",
    ])
    def test_matches_credential_patterns(self, cred: str):
        assert _RE_API_KEY.search(cred), f"Should match: {cred}"

    @pytest.mark.parametrize("text", [
        "the secret to success",     # "secret" without = or :
        "my password was stolen",    # "password" without = or :
        "token of appreciation",     # "token" without = or :
    ])
    def test_rejects_casual_mentions(self, text: str):
        assert not _RE_API_KEY.search(text), f"Should NOT match: {text}"


class TestBase64BlockRegex:
    """_RE_BASE64_BLOCK — very long base64-encoded blocks (>500 chars)."""

    def test_matches_very_long_base64(self):
        b64 = "A" * 600
        assert _RE_BASE64_BLOCK.search(b64)

    def test_matches_base64_with_padding(self):
        b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/" * 10 + "=="
        assert _RE_BASE64_BLOCK.search(b64)

    def test_preserves_moderate_base64(self):
        """Base64 blocks under 500 chars are kept (could be small images, configs)."""
        b64 = "A" * 300
        assert not _RE_BASE64_BLOCK.search(b64)

    def test_rejects_short_base64(self):
        b64 = "A" * 100
        assert not _RE_BASE64_BLOCK.search(b64)

    def test_normal_text_not_matched(self):
        # Normal English text has spaces, punctuation — breaks the pattern
        text = "This is a normal sentence with words and spaces. " * 10
        assert not _RE_BASE64_BLOCK.search(text)

    def test_sanitize_removes_massive_base64(self):
        text = f"here is data: {'A' * 600} end"
        cleaned, actions = sanitize(text)
        assert "base64_block_removed" in actions
        assert "A" * 600 not in cleaned
        assert "[base64-data removed]" in cleaned

    def test_sanitize_preserves_small_inline_image(self):
        """Small base64 images (e.g. icons) should not be stripped."""
        small_b64 = "iVBORw0KGgo" + "A" * 200  # ~211 chars, under threshold
        text = f"see image: {small_b64} above"
        cleaned, actions = sanitize(text)
        assert "base64_block_removed" not in actions
        assert small_b64 in cleaned


class TestScriptRegex:
    """_RE_SCRIPT — <script> tag patterns."""

    @pytest.mark.parametrize("script", [
        "<script>alert(1)</script>",
        "<SCRIPT>alert('xss')</SCRIPT>",
        '<script type="text/javascript">var x=1;</script>',
        "<script\n>multi\nline\n</script>",
    ])
    def test_matches_script_tags(self, script: str):
        assert _RE_SCRIPT.search(script), f"Should match: {script}"

    def test_rejects_partial_script(self):
        # Unclosed script tag should not match (no </script>)
        assert not _RE_SCRIPT.search("<script>alert(1)")

    def test_sanitize_removes_script(self):
        text = "hello <script>alert('xss')</script> world"
        cleaned, actions = sanitize(text)
        assert "script_removed" in actions
        assert "<script>" not in cleaned
        assert "hello" in cleaned and "world" in cleaned


class TestJSURIRegex:
    """_RE_JS_URI — javascript: URI patterns."""

    @pytest.mark.parametrize("uri", [
        "javascript:alert(1)",
        "JavaScript:void(0)",
        "JAVASCRIPT: doStuff()",
        "javascript :alert(1)",  # space before colon
    ])
    def test_matches_js_uris(self, uri: str):
        assert _RE_JS_URI.search(uri), f"Should match: {uri}"

    def test_rejects_javascript_word(self):
        # The word "javascript" without colon
        assert not _RE_JS_URI.search("I love javascript")


class TestDataURIRegex:
    """_RE_DATA_URI — data: URI patterns."""

    @pytest.mark.parametrize("uri", [
        "data:image/png;base64,iVBORw0KGgo=",
        "data:text/plain;base64,SGVsbG8=",
        "data:application/pdf;base64,JVBER",
        "data:image/jpeg;base64,/9j/4AAQ",
    ])
    def test_matches_data_uris(self, uri: str):
        assert _RE_DATA_URI.search(uri), f"Should match: {uri}"

    def test_rejects_non_data_uris(self):
        assert not _RE_DATA_URI.search("data is important")
        assert not _RE_DATA_URI.search("data:not-a-type")


class TestControlCharRegex:
    """_RE_CONTROL — non-printable control characters."""

    def test_matches_null_byte(self):
        assert _RE_CONTROL.search("\x00")

    def test_matches_bell(self):
        assert _RE_CONTROL.search("\x07")

    def test_matches_escape(self):
        assert _RE_CONTROL.search("\x1b")

    def test_matches_del(self):
        assert _RE_CONTROL.search("\x7f")

    def test_preserves_tab_newline_cr(self):
        # \t = 0x09, \n = 0x0a, \r = 0x0d — should NOT match
        assert not _RE_CONTROL.search("\t")
        assert not _RE_CONTROL.search("\n")
        assert not _RE_CONTROL.search("\r")

    def test_sanitize_strips_control_chars(self):
        text = "hello\x00\x01\x02world\x1b[31m"
        cleaned, actions = sanitize(text)
        assert "control_chars_removed" in actions
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned
        assert "helloworld" in cleaned


# ======================================================================
# sanitize() integration tests
# ======================================================================


class TestSanitizeIntegration:
    """End-to-end sanitize() behavior."""

    def test_clean_text_unchanged(self):
        text = "How do I set up a LangGraph agent with memory?"
        cleaned, actions = sanitize(text)
        assert cleaned == text
        assert actions == []

    def test_empty_string(self):
        cleaned, actions = sanitize("")
        assert cleaned == ""
        assert actions == []

    def test_multiple_filters_applied(self):
        text = (
            "My SSN is 123-45-6789, "
            "use Bearer eyJhbGciOiJIUzI1NiJ9 "
            "and key sk-abcdefghijklmnopqrstuv"
        )
        cleaned, actions = sanitize(text)
        assert "ssn_masked" in actions
        assert "bearer_stripped" in actions
        assert "sk_token_stripped" in actions
        assert "123-45-6789" not in cleaned
        assert "eyJ" not in cleaned
        assert "sk-abc" not in cleaned

    def test_truncation_at_max_length(self):
        text = "word " * 2000  # ~10000 chars
        cleaned, actions = sanitize(text)
        assert f"truncated_to_{_MAX_CONTENT_LENGTH}" in actions
        assert len(cleaned) <= _MAX_CONTENT_LENGTH + 20  # +margin for "... [truncated]"
        assert cleaned.endswith("... [truncated]")

    def test_normal_conversation_untouched(self):
        """Typical Teams messages should pass through unchanged."""
        messages = [
            "Hey, can you review my PR?",
            "The build is failing on main branch",
            "Let's sync at 3pm to discuss the architecture",
            "I pushed the fix for JIRA-1234",
            "Thanks! That looks good to merge.",
            "Can you help me set up the local dev environment?",
        ]
        for msg in messages:
            cleaned, actions = sanitize(msg)
            assert cleaned == msg, f"Should be unchanged: {msg}"
            assert actions == [], f"Should have no actions: {msg}"

    def test_html_residuals_cleaned(self):
        text = 'click <script>document.cookie</script> here or javascript:void(0)'
        cleaned, actions = sanitize(text)
        assert "script_removed" in actions
        assert "js_uri_removed" in actions
        assert "<script>" not in cleaned
        assert "javascript:" not in cleaned.lower()

    def test_mixed_pii_and_credentials(self):
        text = (
            "Please update credentials:\n"
            "SSN: 999-88-7777\n"
            "Phone: +1 555 987 6543\n"
            "API key: sk-ant12345678901234567890\n"
            "Password: SuperSecret123!\n"
        )
        cleaned, actions = sanitize(text)
        assert "ssn_masked" in actions
        assert "phone_masked" in actions
        assert "sk_token_stripped" in actions
        # All sensitive data should be masked
        assert "999-88-7777" not in cleaned
        assert "555 987 6543" not in cleaned
        assert "sk-ant" not in cleaned

    def test_preserves_normal_numbers(self):
        """Regular numbers in context should not be masked."""
        text = "JIRA-1234 is due on 2024-12-31, build #5678"
        cleaned, actions = sanitize(text)
        assert "credit_card_masked" not in actions
        assert "ssn_masked" not in actions
        assert "1234" in cleaned
        assert "5678" in cleaned

    def test_unicode_preserved(self):
        text = "Привет! Как дела? 🚀 Let's discuss the план"
        cleaned, actions = sanitize(text)
        assert cleaned == text
        assert actions == []

    def test_code_snippets_preserved(self):
        """Code in messages should not be mangled."""
        text = 'def hello():\n    print("world")\n    return 42'
        cleaned, actions = sanitize(text)
        assert cleaned == text
        assert actions == []


# ======================================================================
# is_safe_for_llm() tests
# ======================================================================


class TestIsSafeForLLM:
    """Safety pre-check for LLM consumption."""

    def test_normal_text_safe(self):
        safe, reason = is_safe_for_llm("How to build a LangGraph agent?")
        assert safe is True
        assert reason == ""

    def test_empty_unsafe(self):
        safe, reason = is_safe_for_llm("")
        assert safe is False
        assert reason == "empty_or_too_short"

    def test_whitespace_only_unsafe(self):
        safe, reason = is_safe_for_llm("   ")
        assert safe is False
        assert reason == "empty_or_too_short"

    def test_single_char_unsafe(self):
        safe, reason = is_safe_for_llm("x")
        assert safe is False
        assert reason == "empty_or_too_short"

    def test_two_chars_safe(self):
        safe, reason = is_safe_for_llm("ok")
        assert safe is True

    def test_url_only_unsafe(self):
        safe, reason = is_safe_for_llm("https://example.com")
        assert safe is False
        assert reason == "urls_only"

    def test_multiple_urls_only_unsafe(self):
        safe, reason = is_safe_for_llm("https://a.com https://b.com")
        assert safe is False
        assert reason == "urls_only"

    def test_url_with_text_safe(self):
        safe, reason = is_safe_for_llm("Check this out: https://example.com — great article!")
        assert safe is True

    def test_binary_blob_unsafe(self):
        blob = "AAAA" * 50  # 200 chars, all base64-valid
        safe, reason = is_safe_for_llm(blob)
        assert safe is False
        assert reason == "binary_content"

    def test_short_alnum_safe(self):
        # Short enough (<50 chars) that binary check doesn't trigger
        safe, reason = is_safe_for_llm("ABC123")
        assert safe is True

    def test_low_printable_ratio_unsafe(self):
        # >50% non-printable
        text = "hi" + "\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89" * 5
        safe, reason = is_safe_for_llm(text)
        assert safe is False
        assert reason == "low_printable_ratio"

    def test_normal_unicode_safe(self):
        safe, reason = is_safe_for_llm("Привет мир! Hello world! 你好世界 🌍")
        assert safe is True

    def test_none_unsafe(self):
        safe, reason = is_safe_for_llm(None)  # type: ignore[arg-type]
        assert safe is False
