from __future__ import annotations

import re
from bs4 import BeautifulSoup
from typing import Dict
from ..utils.logging_utils import timeit


@timeit
def clean_text(raw_html_or_text: str) -> str:
    # If contains tags, strip them
    if "<" in raw_html_or_text and ">" in raw_html_or_text:
        soup = BeautifulSoup(raw_html_or_text, "lxml")
        for el in soup(["script", "style", "noscript"]):
            el.decompose()
        text = soup.get_text("\n")
    else:
        text = raw_html_or_text
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n ?", "\n", text)
    return text.strip()
