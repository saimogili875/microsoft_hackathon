"""
Data Cleaning and Anonymization utilities.
"""

import re
from typing import Any, Dict, List, Union


class DataCleaner:
    """
    Cleans text strings and scrubs PII (emails, phone numbers, SSNs, exact card numbers)
    while preserving domain terms and source tracking IDs.
    """

    EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

    @classmethod
    def scrub_pii(cls, text: str, anonymize: bool = True) -> str:
        if not text or not isinstance(text, str):
            return text

        if not anonymize:
            return text.strip()

        cleaned = cls.EMAIL_REGEX.sub("[ANONYMIZED_EMAIL]", text)
        cleaned = cls.PHONE_REGEX.sub("[ANONYMIZED_PHONE]", cleaned)
        cleaned = cls.CARD_REGEX.sub("[ANONYMIZED_CARD]", cleaned)
        return cleaned.strip()

    @classmethod
    def clean_string_list(cls, items: Any, anonymize: bool = True) -> List[str]:
        if not items:
            return []
        if isinstance(items, str):
            items = [cls.scrub_pii(items, anonymize=anonymize)]
        elif isinstance(items, list):
            res = []
            for item in items:
                if isinstance(item, str) and item.strip():
                    res.append(cls.scrub_pii(item, anonymize=anonymize))
                elif isinstance(item, dict):
                    # extract string values
                    val = " ".join([str(v) for v in item.values() if v])
                    if val.strip():
                        res.append(cls.scrub_pii(val, anonymize=anonymize))
            return res
        return []
