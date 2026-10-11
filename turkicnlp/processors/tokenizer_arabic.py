"""
Tokenizer for Perso-Arabic script Turkic languages (Uyghur, Ottoman Turkish,
South Azerbaijani).

Handles RTL text, ZWNJ inside words, Arabic punctuation, decimals written
with Arabic-Indic digits, and kashida (tatweel) stripping.
"""

from __future__ import annotations

import re
from typing import Optional

from turkicnlp.models.document import Document
from turkicnlp.processors.base import Processor
from turkicnlp.processors.tokenizer import _Tok, build_sentences, split_sentences
from turkicnlp.scripts import Script


class ArabicScriptTokenizer(Processor):
    """Tokenizer for Perso-Arabic script Turkic languages.

    Key differences from Latin/Cyrillic:
    - RTL text direction
    - ZWNJ (U+200C) joins the parts of one word (``مەكتەپ‌كە``); it is
      *not* a token boundary
    - No uppercase/lowercase distinction
    - Arabic punctuation (``،`` ``؛`` ``؟`` ``۔``)
    - Kashida (tatweel, U+0640) and bidi control marks are stripped from the
      token *form*; ``start_char``/``end_char`` always index the original text
    - Decimals (``3.14``, ``٣٫١٤``, ``3,5``) stay one token
    """

    NAME = "tokenize"
    PROVIDES = ["tokenize"]
    REQUIRES = []
    SUPPORTED_SCRIPTS = [Script.PERSO_ARABIC]

    ZWNJ = "‌"
    KASHIDA = "ـ"
    # Invisible characters removed from token forms (offsets unaffected).
    STRIP_CHARS = "ـ‎‏؜‪‫‬‭‮⁦⁧⁨⁩"
    PUNCT_CHARS = "،؛؟!.,;:\\(\\)\\[\\]\\{\\}«»“”‘’'\"۔…؍٭"
    ARABIC_PUNCT = re.compile(rf"(?:[{PUNCT_CHARS}])+")
    # A word is a run of non-space, non-punctuation characters; a "." or ","
    # (or Arabic decimal/thousands separator) between two digits is kept
    # inside the word so decimals are not split.
    TOKEN = re.compile(
        rf"(?:[^\s{PUNCT_CHARS}]|(?<=\d)[.,](?=\d))+"  # word / number
        r"|\.{2,}"  # ellipsis
        rf"|[{PUNCT_CHARS}]"  # single punctuation mark
    )

    def __init__(
        self,
        lang: str,
        script: Optional[Script] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(lang, script, config)
        self._normalize_map: dict[str, str] = {}

    def load(self, model_path: Optional[str] = None) -> None:
        """Load language-specific normalization rules."""
        self._normalize_map = self._load_normalization(self.lang)
        self._loaded = True

    def tokenize(self, text: str) -> list[_Tok]:
        """Return located tokens; offsets index the original ``text``."""
        toks: list[_Tok] = []
        for m in self.TOKEN.finditer(text):
            surface = m.group()
            form = self._normalize(surface)
            if not form:
                continue  # e.g. a stray kashida or bidi mark
            is_punct = bool(self.ARABIC_PUNCT.fullmatch(surface))
            toks.append(_Tok(surface, m.start(), m.end(), punct=is_punct, form=form))
        return toks

    def process(self, doc: Document) -> Document:
        """Tokenize Perso-Arabic script text."""
        toks = self.tokenize(doc.text or "")
        build_sentences(doc, split_sentences(toks, case_sensitive=False))
        doc._processor_log.append("tokenize:arabic_script")
        return doc

    def _normalize(self, text: str) -> str:
        """Normalize Arabic script text (kashida removal, alef variants, etc.)."""
        for ch in self.STRIP_CHARS:
            text = text.replace(ch, "")
        text = re.sub(r"[أإآٱ]", "ا", text)
        for old, new in self._normalize_map.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def _load_normalization(lang: str) -> dict[str, str]:
        """Load language-specific Arabic character normalization rules."""
        if lang == "uig":
            return {}
        elif lang == "ota":
            return {}
        return {}
