"""
PII protection.

The chatbot does not ask for name or contact details. If a customer types identifiers anyway,
mask_text() redacts them before the text is logged or sent to the LLM:
mobile numbers, emails, Aadhaar, PAN, and card / account-like digit runs.
"""
import re

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.I), "[PAN]"),
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[AADHAAR]"),
    (re.compile(r"(?:\+?91[\s-]?)?\b[6-9]\d{4}[\s-]?\d{5}\b"), "[MOBILE]"),
    (re.compile(r"\b\d{13,19}\b"), "[ACCOUNT_NO]"),
]


def mask_text(text: str) -> str:
    out = text
    for pat, tag in _PATTERNS:
        out = pat.sub(tag, out)
    return out
