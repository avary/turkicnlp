"""Regression tests for transliteration fixes (issues A3, B13, B14)."""

from __future__ import annotations

import unicodedata

import pytest

from turkicnlp.scripts import Script
from turkicnlp.scripts.transliterator import (
    CTS_ALPHABET,
    CTS_EXTENSIONS,
    TRANSLITERATION_TABLES,
    Transliterator,
)

L, C, A, T, R = (Script.LATIN, Script.CYRILLIC, Script.PERSO_ARABIC,
                 Script.COMMON_TURKIC, Script.OLD_TURKIC_RUNIC)


def tr(lang: str, src: Script, tgt: Script, text: str) -> str:
    return Transliterator(lang, src, tgt).transliterate(text)


# ---------------------------------------------------------------------------
# A3: Old Turkic runes
# ---------------------------------------------------------------------------

# Independent expectation, keyed by the last word of the Unicode name.
_EXPECTED_RUNE = {
    "A": "a", "AE": "ä", "I": "ı", "E": "e", "O": "o", "OE": "ö",
    "AB": "b", "AEB": "b", "AG": "g", "AEG": "g", "AD": "d", "AED": "d",
    "EZ": "z", "AY": "y", "AEY": "y", "AEK": "k", "OEK": "k",
    "AL": "l", "AEL": "l", "ELT": "lt", "EM": "m", "AN": "n", "AEN": "n",
    "ENT": "nt", "ENC": "nch", "ENY": "ny", "ANG": "ŋ", "ENG": "ŋ",
    "AENG": "ŋ", "EP": "p", "OP": "op", "IC": "ch", "EC": "ch",
    "AQ": "q", "IQ": "q", "OQ": "q", "AR": "r", "AER": "r",
    "AS": "s", "AES": "s", "ASH": "sh", "ESH": "sh",
    "AT": "t", "AET": "t", "OT": "ot", "BASH": "bash",
}


class TestOldTurkic:
    def test_every_assigned_code_point_is_name_consistent(self) -> None:
        t = Transliterator("otk", R, L)
        seen = 0
        for cp in range(0x10C00, 0x10C50):
            name = unicodedata.name(chr(cp), None)
            if name is None:
                assert chr(cp) not in TRANSLITERATION_TABLES["otk_Orkh_to_Latn"]
                continue
            assert name.startswith("OLD TURKIC LETTER ")
            suffix = name.rsplit(" ", 1)[1]
            assert t.transliterate(chr(cp)) == _EXPECTED_RUNE[suffix], name
            seen += 1
        assert seen == 73  # U+10C00..U+10C48

    def test_turk(self) -> None:
        # 𐱅 ORKHON AET, 𐰇 ORKHON OE, 𐰼 ORKHON AER, 𐰚 ORKHON AEK
        assert tr("otk", R, L, "𐱅𐰇𐰼𐰚") == "törk"
        assert tr("otk", R, L, "\U00010C45\U00010C07\U00010C3C\U00010C1A") == "törk"

    def test_bash_is_a_letter_not_separator(self) -> None:
        assert tr("otk", R, L, "\U00010C48") == "bash"

    def test_word_separator(self) -> None:
        assert tr("otk", R, L, "𐱅𐰇𐰼𐰚:𐰉𐰆𐰑𐰣") == "törk:bodn"
        assert tr("otk", R, L, "𐱅𐰇𐰼𐰚⁚") == "törk:"


# ---------------------------------------------------------------------------
# B13: all-caps handling
# ---------------------------------------------------------------------------


class TestAllCaps:
    @pytest.mark.parametrize("lang,src,tgt,text,expected", [
        ("uzb", C, L, "ШАҲАР", "SHAHAR"),
        ("uzb", C, L, "Шаҳар", "Shahar"),
        ("uzb", C, L, "Ш", "Sh"),
        ("uzb", C, L, "ЧОЙ ЁКИ ЮЗ", "CHOY YOKI YUZ"),
        ("uzb", C, L, "ЕР", "YER"),
        ("kaz", C, L, "ЩУКА", "ŞTŞUKA"),
        ("kaz", C, L, "Щука", "Ştşuka"),
        ("kaz", C, L, "ЖАЯУ", "JAİAU"),
        ("kaz", C, L, "ЧАЙ", "TŞAİ"),
        ("tuk", C, L, "ЕЛ", "ÝEL"),
        ("tuk", C, L, "ЩЁТКА", "ŞÇÝOTKA"),
        ("tat", C, L, "ЕЛ", "YEL"),
        ("kaa", C, L, "ШАҲАР", "SHAHAR"),
        ("crh", C, L, "КЪЫРЫМ", "QIRIM"),
        ("aze", C, L, "ЈАХШЫ", "YAXŞI"),
        ("tur", L, T, "İSTANBUL", "İSTANBUL"),
        ("kir", C, T, "ЖИГИТ", "CİGİT"),
        ("uzb", L, C, "SHAHAR", "ШАҲАР"),
        ("uzb", L, C, "MAS'UL", "МАСЪУЛ"),
        ("kaz", L, C, "ŞTŞUKA", "ЩУКА"),
    ])
    def test_all_caps(self, lang, src, tgt, text, expected) -> None:
        assert tr(lang, src, tgt, text) == expected

    def test_mixed_sentence(self) -> None:
        assert tr("uzb", C, L, "ШАҲАР ва Шаҳар") == "SHAHAR va Shahar"


# ---------------------------------------------------------------------------
# B14: table fixes
# ---------------------------------------------------------------------------


class TestUzbek:
    def test_s_h_cyrl_to_latn(self) -> None:
        assert tr("uzb", C, L, "Исҳоқ") == "Isʼhoq"
        assert tr("uzb", C, L, "ИСҲОҚ") == "ISʼHOQ"
        assert tr("uzb", C, L, "шаҳар") == "shahar"

    @pytest.mark.parametrize("text", ["Isʼhoq", "Is'hoq", "Is’hoq"])
    def test_s_h_latn_to_cyrl(self, text: str) -> None:
        assert tr("uzb", L, C, text) == "Исҳоқ"

    def test_s_h_round_trip(self) -> None:
        for word in ["Исҳоқ", "асҳоб", "шаҳар"]:
            assert tr("uzb", L, C, tr("uzb", C, L, word)) == word

    @pytest.mark.parametrize("apos", ["'", "‘", "’", "ʻ", "ʼ"])
    def test_latn_to_cts_apostrophe_variants(self, apos: str) -> None:
        assert tr("uzb", L, T, f"o{apos}g{apos}il") == "öğil"
        assert tr("uzb", L, T, f"O{apos}zbekiston") == "Özbekiston"
        assert tr("uzb", L, T, f"yo{apos}q") == "yöq"

    def test_latn_to_cts_s_h(self) -> None:
        assert tr("uzb", L, T, "Isʼhoq") == "İshoq"
        assert tr("uzb", C, T, "Исҳоқ") == "İshoq"


class TestAzerbaijani:
    def test_je_cyrl_to_latn(self) -> None:
        assert tr("aze", C, L, "Јол") == "Yol"
        assert tr("aze", C, L, "јахшы") == "yaxşı"
        assert tr("aze", C, L, "йол") == "yol"  # й accepted as input variant

    def test_je_never_leaks(self) -> None:
        for tgt in (L, T):
            out = tr("aze", C, tgt, "Ајдын ЈАЗЫ ај")
            assert "ј" not in out and "Ј" not in out

    def test_latn_to_cyrl_uses_je(self) -> None:
        assert tr("aze", L, C, "yol") == "јол"
        assert tr("aze", L, C, "Yaxşı") == "Јахшы"
        assert "й" not in tr("aze", L, C, "ay, oyun, yuxu")

    def test_cyrl_to_cts(self) -> None:
        assert tr("aze", C, T, "Јол") == "Yol"
        assert tr("aze", C, T, "журнал") == "jurnal"  # ж = /ʒ/ = CTS j


class TestUyghur:
    def test_cyrl_input_case(self) -> None:
        assert tr("uig", C, L, "Уйғур") == "Uyghur"
        assert tr("uig", C, L, "УЙҒУР") == "UYGHUR"
        assert tr("uig", C, L, "уйғур тили") == "uyghur tili"
        assert tr("uig", C, A, "Салам") == tr("uig", C, A, "салам")

    def test_latn_input_keeps_case(self) -> None:
        assert tr("uig", L, C, "Uyghur") == "Уйғур"
        assert tr("uig", L, C, "UYGHUR") == "УЙҒУР"
        assert tr("uig", L, T, "Shinjang") == "Şincañ"
        assert tr("uig", T, L, "Şincañ") == "Shinjang"

    def test_arabic_punctuation_out(self) -> None:
        assert tr("uig", A, L, "سالام، ياخشىمۇ؟") == "salam, yaxshimu?"
        assert tr("uig", A, C, "سالام؛") == "салам;"
        assert tr("uig", A, T, "سالام؟") == "salam?"

    def test_arabic_punctuation_in(self) -> None:
        assert tr("uig", L, A, "salam, yaxshimu?") == "سالام، ياخشىمۇ؟"
        assert tr("uig", C, A, "салам;") == "سالام؛"
        assert tr("uig", T, A, "salam?") == "سالام؟"


class TestTurkmen:
    @pytest.mark.parametrize("cyrl,latn", [
        ("ел", "ýel"),
        ("Ел", "Ýel"),
        ("поезд", "poýezd"),
        ("аеропорт", "aýeroport"),
        ("дүе", "düýe"),
        ("мен", "men"),
    ])
    def test_e_after_vowel(self, cyrl: str, latn: str) -> None:
        assert tr("tuk", C, L, cyrl) == latn

    def test_cts(self) -> None:
        assert tr("tuk", C, T, "ая") == "aya"
        assert tr("tuk", C, T, "юрт") == "yurt"
        assert tr("tuk", C, T, "ел") == "yel"


class TestCTSOutputAlphabet:
    _ALLOWED = CTS_ALPHABET | CTS_EXTENSIONS | {"'"}

    def test_alphabet_size(self) -> None:
        assert len(CTS_ALPHABET) == 68  # 34 letters x 2 cases

    @pytest.mark.parametrize(
        "key", sorted(k for k in TRANSLITERATION_TABLES if k.endswith("_to_CTS"))
    )
    def test_table_outputs_are_cts(self, key: str) -> None:
        for src, out in TRANSLITERATION_TABLES[key].items():
            for ch in out:
                assert ch in self._ALLOWED, (key, src, out)

    @pytest.mark.parametrize("lang,src,text", [
        ("tuk", C, "Юрт ая ёл ел Ялы"),
        ("bak", C, "ҡыҙ ҫу Ҙур Ҫәй"),
        ("azb", A, "سئوگی، گؤز ایل بیر آی ائو ئیش نه؟"),
        ("uzb", C, "Исҳоқ ўғил"),
        ("aze", C, "Јол жург"),
    ])
    def test_converted_text_is_cts(self, lang, src, text) -> None:
        out = tr(lang, src, T, text)
        for ch in out:
            assert ch in self._ALLOWED or not ch.isalpha(), (lang, out)

    def test_bashkir(self) -> None:
        assert tr("bak", C, T, "ҡыҙ") == "qız"
        assert tr("bak", C, T, "ҫу") == "su"

    def test_azb_vowels_kept(self) -> None:
        assert tr("azb", A, T, "سئوگی") == "sevgi"
        assert tr("azb", A, T, "گؤز") == "göz"
        assert tr("azb", A, T, "ایل") == "il"
        assert tr("azb", A, T, "بیر") == "bir"
        assert tr("azb", A, T, "آی") == "ay"
        assert tr("azb", A, T, "ائو") == "ev"
        assert tr("azb", A, T, "نه؟") == "nä?"
