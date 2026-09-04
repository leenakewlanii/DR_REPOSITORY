"""Reviewed UI translation surface; add languages here before exposing them."""

LANGUAGES = ["English", "Hindi"]

TRANSLATIONS = {
    "English": {},
    "Hindi": {
        "screening": "रेटिना इमेज स्क्रीनिंग",
        "model_result": "AI स्क्रीनिंग परिणाम",
        "what_means": "इसका क्या मतलब है?",
        "what_next": "अब मुझे क्या करना चाहिए?",
        "listen": "परिणाम सुनें",
        "analyze": "इमेज का विश्लेषण करें",
        "report": "स्क्रीनिंग रिपोर्ट डाउनलोड करें",
    },
}


def t(key: str, language: str = "English", fallback: str = "") -> str:
    """Return a reviewed translation, falling back to English copy."""
    return TRANSLATIONS.get(language, {}).get(key) or TRANSLATIONS["English"].get(key) or fallback
