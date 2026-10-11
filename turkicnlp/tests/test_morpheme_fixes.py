"""Tests for the MorphemeTokenizer fixes (B19)."""

from __future__ import annotations

from turkicnlp.processors import morpheme_tokenizer as mt
from turkicnlp.processors.morpheme_tokenizer import (
    MorphemeTokenizer,
    _Phonology,
    _find_stem_boundary,
    _form_variants,
    _segment_by_ud_features,
    _turkic_lower,
)


def _loaded(lang: str) -> MorphemeTokenizer:
    tok = MorphemeTokenizer(lang=lang)
    tok._loaded = True  # no backends: rule-based paths only
    return tok


class TestDotlessI:
    def test_dotless_languages(self):
        for lang in ("tur", "aze", "crh", "gag", "tat", "kaz"):
            assert _turkic_lower("IRAK", lang) == "ırak", lang
            assert _turkic_lower("İSTANBUL", lang) == "istanbul", lang

    def test_languages_without_dotless_i(self):
        for lang in ("uzb", "tuk", "kaa"):
            assert _turkic_lower("ISHLAR", lang) == "ishlar", lang
            assert _turkic_lower("İ", lang) == "i", lang

    def test_default_is_turkish_convention(self):
        assert _turkic_lower("IRAK") == "ırak"

    def test_tokenizer_uses_language(self):
        assert MorphemeTokenizer(lang="uzb")._phon.lang == "uzb"
        assert MorphemeTokenizer(lang="tur")._phon.lang == "tur"

    def test_uzbek_capitalised_stem_boundary(self):
        # "ISHLAR" with lemma "ish": I must lowercase to i, not ı
        assert _find_stem_boundary("ISHLAR", "ish", "uzb", "NOUN") == 3
        # Turkish keeps the dotless reading
        assert _find_stem_boundary("ILIK", "ılık", "tur", "ADJ") == 4


class TestEmptyInput:
    def test_space_and_empty(self):
        tok = _loaded("tur")
        assert tok.segment(" ").segments == []
        assert tok.segment("").segments == []
        assert tok.segment("  \t").segments == []

    def test_surrounding_whitespace(self):
        tok = _loaded("tur")
        assert tok.segment(" ev ").segments == ["ev"]


class TestBaseCacheBounded:
    def test_lru(self):
        class FakeHFST:
            calls = 0

            def _analyze(self, w):
                FakeHFST.calls += 1
                return [{"lemma": w, "pos": "n"}]

        tok = _loaded("tur")
        tok._hfst_available = True
        tok._hfst_analyzer = FakeHFST()
        tok._base_cache_size = 3
        for w in ("ev", "okul", "kitap", "masa", "kalem"):
            assert tok._is_lexical_base(w, verb=False)
        assert len(tok._base_cache) == 3
        assert ("ev", False) not in tok._base_cache
        n = FakeHFST.calls
        tok._is_lexical_base("kalem", verb=False)  # cache hit
        assert FakeHFST.calls == n

    def test_default_bound(self):
        assert MorphemeTokenizer(lang="tur")._base_cache_size == 100_000


class TestFormVariantsCache:
    def test_truncated_list_not_cached(self):
        form = "aeaeaeae"  # 6**8 alternants -> truncated
        mt._VARIANT_CACHE.pop((form, "Latn"), None)
        v = _form_variants(form, "Latn")
        assert len(v) > mt._MAX_VARIANTS
        assert (form, "Latn") not in mt._VARIANT_CACHE

    def test_complete_list_cached(self):
        mt._VARIANT_CACHE.pop(("de", "Latn"), None)
        v = _form_variants("de", "Latn")
        assert "da" in v and "te" in v
        assert mt._VARIANT_CACHE[("de", "Latn")] is v


class TestApostrophe:
    def test_last_apostrophe(self):
        assert _find_stem_boundary("Kur'an'ı", "Kur'an", "tur", "PROPN") == 6
        assert _find_stem_boundary("İstanbul'da", "İstanbul", "tur", "PROPN") == 8

    def test_trailing_apostrophe_ignored(self):
        assert _find_stem_boundary("ev'", "ev", "tur", "NOUN") == 2

    def test_segment_by_features(self):
        phon = _Phonology("Latn", "tur")
        morphs = _segment_by_ud_features(
            "Kur'an'ı", 6, "PROPN", "Case=Acc", mt._TUR_SUFFIXES, phon, "Latn",
        )
        assert [m.surface for m in morphs] == ["Kur'an", "'ı"]
        assert morphs[1].label == "ACC"

    def test_uzbek_apostrophe_is_a_letter(self):
        # o' / g' are letters in Uzbek: no apostrophe boundary
        assert _find_stem_boundary("o'g'il", "o'g'il", "uzb", "NOUN") == 6


class TestDeadCodeRemoved:
    def test_removed(self):
        assert not hasattr(MorphemeTokenizer, "_try_hfst")
        assert not hasattr(mt, "_greedy_match_one")

    def test_docstring_describes_hybrid_design(self):
        doc = MorphemeTokenizer.__doc__
        assert "21 languages" not in doc
        assert "rule" in doc
