"""Regression tests for tokenizer offsets (A4), tokenization rules (B15)
and MWT rule integrity (B16)."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from turkicnlp.models.document import Document
from turkicnlp.processors.tokenizer import RegexTokenizer
from turkicnlp.processors.tokenizer_arabic import ArabicScriptTokenizer


def _regex(text: str, lang: str = "uzb") -> Document:
    doc = Document(text=text, lang=lang)
    tok = RegexTokenizer(lang=lang)
    tok.load()
    return tok.process(doc)


def _arabic(text: str, lang: str = "uig") -> Document:
    doc = Document(text=text, lang=lang)
    tok = ArabicScriptTokenizer(lang=lang)
    tok.load()
    return tok.process(doc)


def _words(doc: Document) -> list[str]:
    return [w.text for s in doc.sentences for w in s.words]


def _sents(doc: Document) -> list[str]:
    return [s.text for s in doc.sentences]


# ---------------------------------------------------------------------------
# A4: offsets
# ---------------------------------------------------------------------------

REGEX_OFFSET_TEXTS = [
    "Bir.  Ikki.",
    "   Bir.\n\n\tIkki!   Uch?  ",
    "Men bardım, ok? Boladi!",
    "Prof. Aliyev keldi.   1. Birinchi bob.\nHa...  Yo‘q…",
    "«Salom», dedi u. (Ha!) [1] {x} „Bu“ 'bu' “bu” ‘bu’.",
    "Олма, алма т.б.    Біз келдік.\r\nСол 1991 ж. болды.",
    "Kur'an'ı okudu.  Sonra 3.14 ve 3,5 dedi.",
    "",
    "    ",
]


@pytest.mark.parametrize("text", REGEX_OFFSET_TEXTS)
def test_regex_offsets_index_original_text(text: str) -> None:
    doc = _regex(text)
    for sent in doc.sentences:
        assert text[sent.tokens[0].start_char : sent.tokens[-1].end_char] == sent.text
        for tok in sent.tokens:
            assert text[tok.start_char : tok.end_char] == tok.text
        for w in sent.words:
            assert text[w.start_char : w.end_char] == w.text


def test_regex_offsets_double_space_between_sentences() -> None:
    doc = _regex("Bir.  Ikki.")
    ikki = doc.sentences[1].words[0]
    assert ikki.text == "Ikki"
    assert (ikki.start_char, ikki.end_char) == (6, 10)


ARABIC_OFFSET_TEXTS = [
    "مەكتەپ‌كە؟  باردىم.",
    "  بۇ ٣٫١٤ ۋە 3.14.\n\nئۇ «ياخشى» دېدى!",
    "كتابــلر  اوقودم.   آمد، گلدی؛ نه اولدی؟ ",
    "ســـــلام‏ دنیا.",
]


@pytest.mark.parametrize("text", ARABIC_OFFSET_TEXTS)
def test_arabic_offsets_index_original_text(text: str) -> None:
    doc = _arabic(text, lang="ota")
    tok = ArabicScriptTokenizer(lang="ota")
    tok.load()
    assert doc.sentences
    for sent in doc.sentences:
        for w in sent.words:
            surface = text[w.start_char : w.end_char]
            # Token forms may be normalized (kashida stripped, alef unified);
            # offsets must still point at the original surface.
            assert tok._normalize(surface) == w.text
            assert surface.strip() == surface and surface


def test_arabic_kashida_offsets() -> None:
    text = "كتابــلر اوقودم."
    doc = _arabic(text, lang="ota")
    words = doc.sentences[0].words
    assert [w.text for w in words] == ["كتابلر", "اوقودم", "."]
    assert text[words[0].start_char : words[0].end_char] == "كتابــلر"
    assert text[words[1].start_char : words[1].end_char] == "اوقودم"


# ---------------------------------------------------------------------------
# B15: tokenization rules
# ---------------------------------------------------------------------------


def test_brackets_and_quotes_split() -> None:
    doc = _regex("«Salom» (do‘st) [ha] {yo‘q} „bu“ “bu” ‘bu’ \"bu\" 'bu'.")
    assert _words(doc) == [
        "«", "Salom", "»", "(", "do‘st", ")", "[", "ha", "]", "{", "yo‘q", "}",
        "„", "bu", "“", "“", "bu", "”", "‘", "bu", "’", '"', "bu", '"',
        "'", "bu", "'", ".",
    ]


def test_cyrillic_guillemets() -> None:
    doc = _regex("Ол «Абай жолы» романын оқыды.", lang="kaz")
    assert _words(doc) == ["Ол", "«", "Абай", "жолы", "»", "романын", "оқыды", "."]


def test_title_abbreviation_does_not_split() -> None:
    doc = _regex("Prof. Aliyev keldi.")
    assert _sents(doc) == ["Prof. Aliyev keldi."]
    assert _words(doc) == ["Prof.", "Aliyev", "keldi", "."]


@pytest.mark.parametrize(
    "text,lang",
    [
        ("Dr. Ahmet geldi.", "tur"),
        ("Проф. Әлиев келді.", "kaz"),
        ("A. Navoiy buyuk shoir.", "uzb"),
        ("Doç. Dr. Ayşe Yılmaz konuştu.", "tur"),
    ],
)
def test_more_title_abbreviations(text: str, lang: str) -> None:
    assert len(_regex(text, lang).sentences) == 1


def test_terminal_abbreviations() -> None:
    doc = _regex("Олма, алма т.б. жемістер бар.", lang="kaz")
    assert len(doc.sentences) == 1
    assert "т.б." in _words(doc)
    doc = _regex("Olma, nok va b. mevalar bor.")
    assert len(doc.sentences) == 1
    assert _words(doc)[4] == "b."
    doc = _regex("Elma, armut vb. Sonra eve gittik.", lang="tur")
    assert _sents(doc) == ["Elma, armut vb.", "Sonra eve gittik."]


def test_ordinal_does_not_end_sentence() -> None:
    doc = _regex("1. Birinchi bob. 2. Dünya Savaşı bitti.")
    assert _sents(doc) == ["1. Birinchi bob.", "2. Dünya Savaşı bitti."]
    assert _words(doc)[0] == "1."


def test_number_at_sentence_end_is_split() -> None:
    doc = _regex("Bu yil 2020. Keyin keldi.")
    assert _sents(doc) == ["Bu yil 2020.", "Keyin keldi."]
    assert _words(doc)[:4] == ["Bu", "yil", "2020", "."]


@pytest.mark.parametrize("ellipsis", ["...", "…"])
def test_ellipsis_single_token(ellipsis: str) -> None:
    doc = _regex(f"Ha{ellipsis} Keyin keldi.")
    assert _words(doc)[:2] == ["Ha", ellipsis]
    assert len(doc.sentences) == 2


def test_quoted_exclamation_followed_by_lowercase() -> None:
    doc = _regex('«Salom!» dedi u. "Geldim." diye söyledi.')
    assert _sents(doc) == ["«Salom!» dedi u.", '"Geldim." diye söyledi.']


def test_closing_quote_stays_in_sentence() -> None:
    doc = _regex("U «Keldim.» Keyin ketdi.")
    assert _sents(doc) == ["U «Keldim.»", "Keyin ketdi."]


def test_uzbek_apostrophe_letters() -> None:
    assert _words(_regex("O‘qing, g‘alaba bo‘ladi.")) == [
        "O‘qing", ",", "g‘alaba", "bo‘ladi", ".",
    ]


def test_decimals_kept_latin() -> None:
    assert _words(_regex("Narx 3.14 yoki 3,5 som.")) == ["Narx", "3.14", "yoki", "3,5", "som", "."]


def test_arabic_zwnj_not_split() -> None:
    doc = _arabic("مەكتەپ‌كە باردىم.")
    assert _words(doc) == ["مەكتەپ‌كە", "باردىم", "."]


@pytest.mark.parametrize("num", ["3.14", "٣٫١٤", "3,5", "۳٫۵", "١٬٠٠٠"])
def test_arabic_decimals_kept(num: str) -> None:
    doc = _arabic(f"بۇ {num} دانە.")
    assert _words(doc) == ["بۇ", num, "دانە", "."]


def test_arabic_punctuation_and_sentences_ottoman() -> None:
    doc = _arabic("«كتاب» اوقودم، گلدم؛ نه اولدی؟ ایی... تمام۔ بیتدی!", lang="ota")
    assert _words(doc) == [
        "«", "كتاب", "»", "اوقودم", "،", "گلدم", "؛", "نه", "اولدی", "؟",
        "ایی", "...", "تمام", "۔", "بیتدی", "!",
    ]
    assert len(doc.sentences) == 4


# ---------------------------------------------------------------------------
# B16: MWT rule files must be self-consistent
# ---------------------------------------------------------------------------

_RULE_DIRS = [
    Path(__file__).resolve().parents[1] / "resources" / "mwt_rules",
    Path(__file__).resolve().parents[1] / "resources" / "mwt_rules_extracted",
]
_RULE_FILES = sorted(p for d in _RULE_DIRS for p in d.glob("*.json"))


def _norm(text: str) -> str:
    """Case-insensitive key (Turkish I/İ aware, kashida-free)."""
    text = unicodedata.normalize("NFC", text).replace("ـ", "")
    text = text.replace("İ", "i").replace("I", "ı")
    text = unicodedata.normalize("NFD", text.lower()).replace("̇", "")
    return unicodedata.normalize("NFC", text)


def test_rule_files_found() -> None:
    assert len(_RULE_FILES) >= 23


@pytest.mark.parametrize("path", _RULE_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_mwt_rules_reconstruct_surface(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    bad = []
    for rule in data.get("rules", []):
        surface = rule.get("surface")
        pieces = rule.get("split") or rule.get("pieces") or []
        if surface == "Korkutata":
            bad.append((surface, pieces, "placeholder"))
            continue
        if (
            len(pieces) < 2
            or any(not p or p == "_" for p in pieces)
            or _norm("".join(pieces)) != _norm(surface)
        ):
            bad.append((surface, pieces))
    assert not bad, f"{path.name}: corrupt MWT rules: {bad[:10]}"
