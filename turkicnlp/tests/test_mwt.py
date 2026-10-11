"""Tests for MWT expansion processor."""

from __future__ import annotations

import pytest

from turkicnlp.models.document import Document
from turkicnlp.processors.mwt import MWTProcessor
from turkicnlp.processors.tokenizer import RegexTokenizer
from turkicnlp.processors.tokenizer_arabic import ArabicScriptTokenizer
from turkicnlp.scripts import Script


def _tokenize(lang: str, script: Script, text: str) -> Document:
    doc = Document(text=text, lang=lang, script=script.value)
    if script == Script.PERSO_ARABIC:
        tok = ArabicScriptTokenizer(lang=lang, script=script)
    else:
        tok = RegexTokenizer(lang=lang, script=script)
    tok.load()
    tok.process(doc)
    return doc


@pytest.mark.parametrize(
    "lang,script,text,surface,pieces",
    [
        ("tur", Script.LATIN, "Korkutata geldi.", "Korkutata", ["Korkut", "ata"]),
        ("aze", Script.LATIN, "Korkutata geldi.", "Korkutata", ["Korkut", "ata"]),
        ("uzb", Script.LATIN, "Korkutata keldi.", "Korkutata", ["Korkut", "ata"]),
        ("crh", Script.LATIN, "Korkutata keldi.", "Korkutata", ["Korkut", "ata"]),
        ("kaa", Script.LATIN, "Korkutata keldi.", "Korkutata", ["Korkut", "ata"]),
        ("gag", Script.LATIN, "Korkutata geldi.", "Korkutata", ["Korkut", "ata"]),
        ("otk", Script.LATIN, "Korkutata geldi.", "Korkutata", ["Korkut", "ata"]),
        ("kaz", Script.CYRILLIC, "Қорқытата келді.", "Қорқытата", ["Қорқыт", "ата"]),
        ("kir", Script.CYRILLIC, "Коркытата келди.", "Коркытата", ["Коркыт", "ата"]),
        ("tat", Script.CYRILLIC, "Коркытата килде.", "Коркытата", ["Коркыт", "ата"]),
        ("bak", Script.CYRILLIC, "Коркытата килде.", "Коркытата", ["Коркыт", "ата"]),
        ("chv", Script.CYRILLIC, "Коркытата килче.", "Коркытата", ["Коркыт", "ата"]),
        ("sah", Script.CYRILLIC, "Коркытата келди.", "Коркытата", ["Коркыт", "ата"]),
        ("nog", Script.CYRILLIC, "Коркытата келди.", "Коркытата", ["Коркыт", "ата"]),
        ("kum", Script.CYRILLIC, "Коркытата келди.", "Коркытата", ["Коркыт", "ата"]),
        ("krc", Script.CYRILLIC, "Коркытата келди.", "Коркытата", ["Коркыт", "ата"]),
        ("alt", Script.CYRILLIC, "Коркытата келди.", "Коркытата", ["Коркыт", "ата"]),
        ("tyv", Script.CYRILLIC, "Коркытата келди.", "Коркытата", ["Коркыт", "ата"]),
        ("kjh", Script.CYRILLIC, "Коркытата килде.", "Коркытата", ["Коркыт", "ата"]),
        ("azb", Script.PERSO_ARABIC, "قورقوتاتا آمد.", "قورقوتاتا", ["قورقوت", "اتا"]),
        ("uig", Script.PERSO_ARABIC, "قورقوتاتا كەلدى.", "قورقوتاتا", ["قورقوت", "اتا"]),
        ("ota", Script.PERSO_ARABIC, "قورقوتاتا آمد.", "قورقوتاتا", ["قورقوت", "اتا"]),
    ],
)
def test_mwt_expansion_mechanism(
    lang: str, script: Script, text: str, surface: str, pieces: list[str]
) -> None:
    # The shipped "Korkutata" placeholder rule was removed (B16); exercise the
    # expansion machinery per language with an injected rule instead.
    doc = _tokenize(lang, script, text)
    mwt = MWTProcessor(lang=lang, script=script)
    mwt.load("")
    mwt._rules = {surface: pieces}
    mwt.process(doc)

    assert doc.sentences[0].tokens[0].is_mwt
    assert doc.sentences[0].tokens[0].text == surface
    assert [w.text for w in doc.sentences[0].tokens[0].words] == pieces


@pytest.mark.parametrize(
    "lang,text,surface,pieces",
    [
        ("tur", "SONRAKİ adım.", "SONRAKİ", ["SONRA", "Kİ"]),
        ("tur", "Bu doğrudur.", "doğrudur", ["doğru", "dur"]),
    ],
)
def test_mwt_expansion_with_extracted_rules(
    lang: str, text: str, surface: str, pieces: list[str]
) -> None:
    doc = _tokenize(lang, Script.LATIN, text)
    mwt = MWTProcessor(lang=lang, script=Script.LATIN)
    mwt.load("")
    mwt.process(doc)
    tok = next(t for t in doc.sentences[0].tokens if t.text == surface)
    assert tok.is_mwt
    assert [w.text for w in tok.words] == pieces


def test_placeholder_rule_removed() -> None:
    mwt = MWTProcessor(lang="tur", script=Script.LATIN)
    mwt.load("")
    assert "Korkutata" not in mwt._rules


def test_mwt_no_rule_no_change() -> None:
    doc = _tokenize("tur", Script.LATIN, "Merhaba dunya.")
    mwt = MWTProcessor(lang="tur", script=Script.LATIN)
    mwt.load("")
    mwt.process(doc)

    assert not doc.sentences[0].tokens[0].is_mwt
    assert doc.sentences[0].tokens[0].text == "Merhaba"
