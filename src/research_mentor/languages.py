"""Supported languages for AI response translation.

Canonical list — used by API validation, UI language picker, and presenter node.
ISO 639-1 codes. English first (default), then alphabetically by English name.
"""

from __future__ import annotations

SUPPORTED_LANGUAGES: list[dict[str, str]] = [
    {"code": "en", "name": "English", "nativeName": "English"},
    {"code": "ar", "name": "Arabic", "nativeName": "العربية"},
    {"code": "zh", "name": "Chinese", "nativeName": "中文"},
    {"code": "fr", "name": "French", "nativeName": "Français"},
    {"code": "de", "name": "German", "nativeName": "Deutsch"},
    {"code": "hi", "name": "Hindi", "nativeName": "हिन्दी"},
    {"code": "ja", "name": "Japanese", "nativeName": "日本語"},
    {"code": "ko", "name": "Korean", "nativeName": "한국어"},
    {"code": "ru", "name": "Russian", "nativeName": "Русский"},
    {"code": "es", "name": "Spanish", "nativeName": "Español"},
]

LANGUAGE_CODES = {lang["code"] for lang in SUPPORTED_LANGUAGES}

LANGUAGE_NAMES = {lang["code"]: lang["name"] for lang in SUPPORTED_LANGUAGES}


def validate_language_code(code: str) -> bool:
    """Check if a language code is supported."""
    return code in LANGUAGE_CODES


def get_language_name(code: str) -> str:
    """Get English display name for a language code."""
    return LANGUAGE_NAMES.get(code, code.upper())


def get_default_language() -> str:
    """Get the default language code."""
    return "en"
