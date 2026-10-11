"""Regression tests for the pipeline, morphology, tag-mapping and I/O fixes."""

from __future__ import annotations

import json

import pytest

from turkicnlp.models.document import Document, Sentence, Token, Word
from turkicnlp.pipeline import Pipeline, default_processors
from turkicnlp.processors.base import Processor, ProcessorRequirementsError
from turkicnlp.resources.registry import ModelRegistry, ProcessorRegistry


# ---------------------------------------------------------------------------
# A1 / D33: placeholders are never chosen, requirements come from the backend


SMALL_LANGS = ["klj", "crh", "chv", "gag", "alt", "tyv", "kjh", "sah", "ota", "otk"]


@pytest.mark.parametrize("lang", SMALL_LANGS)
def test_default_plan_has_no_placeholders(lang):
    plan = Pipeline(lang)._plan()
    for name, backend in plan:
        cls = ProcessorRegistry.get(name, backend)
        assert getattr(cls, "IMPLEMENTED", True), (lang, name, backend)


def test_klj_does_not_pull_in_pos():
    names = [n for n, _ in Pipeline("klj")._plan()]
    assert "pos" not in names


def test_implemented_backends_skip_placeholders():
    assert "neural" not in ProcessorRegistry.implemented_backends("pos")
    assert "dictionary" not in ProcessorRegistry.implemented_backends("lemma")
    assert ProcessorRegistry.get_any("lemma").IMPLEMENTED


def test_sentiment_without_catalog_entry_defaults_to_llm():
    pipe = Pipeline("tur", processors=None)
    assert pipe._choose_backend("sentiment", {}) == "llm"


def test_default_processors_drops_unregistered_backend():
    names = default_processors({
        "tokenize": {"default": "rule", "backends": {"rule": {}}},
        "lemma": {"default": "not_installed", "backends": {"not_installed": {}}},
    })
    assert names == ["tokenize"]


def test_default_processors_drop_glot500_parts_next_to_apertium():
    cat = {
        "tokenize": {"default": "rule"},
        "morph": {"default": "apertium"},
        "feats": {"default": "multilingual_glot500_morph"},
        "lemma": {"default": "multilingual_glot500_morph"},
    }
    assert default_processors(cat) == ["tokenize", "morph"]


# ---------------------------------------------------------------------------
# A2: empty input


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_empty_input_does_not_crash(text):
    nlp = Pipeline("kaz", processors=["tokenize", "mwt"], tokenize_backend="rule")
    doc = nlp(text)
    assert doc.sentences == []
    assert doc.script


class _NeedsPos(Processor):
    NAME = "x"
    REQUIRES = ["tokenize", "pos"]

    def load(self, model_path):
        pass

    def process(self, doc):
        return doc


def test_pos_requirement_checks_all_words():
    words = [Word(id=1, text="a"), Word(id=2, text="b", upos="NOUN")]
    doc = Document(text="a b", sentences=[Sentence(text="a b", words=words)])
    _NeedsPos(lang="tur").check_requirements(doc)  # first word untagged is fine
    words[1].upos = None
    with pytest.raises(ProcessorRequirementsError):
        _NeedsPos(lang="tur").check_requirements(doc)


# ---------------------------------------------------------------------------
# A5: one owner per field


class _FakeMorph(Processor):
    NAME = "morph"
    PROVIDES = ["lemma", "pos", "feats"]

    def load(self, model_path):
        pass

    def process(self, doc):
        for w in doc.words:
            w.lemma, w.upos, w.feats = "m_" + w.text, "NOUN", "Case=Nom"
        return doc


class _FakePos(Processor):
    NAME = "pos"
    PROVIDES = ["pos"]

    def load(self, model_path):
        pass

    def process(self, doc):
        for w in doc.words:
            w.upos = "VERB" if w.text == "keldi" else "NOUN"
        return doc


def _run(processors, text="Ali keldi"):
    pipe = Pipeline("tur", processors=None)
    words = [Word(id=i + 1, text=t) for i, t in enumerate(text.split())]
    pipe._processors = list(processors)
    pipe._owners = pipe._field_owners()
    doc = Document(text=text, sentences=[Sentence(text=text, words=words,
                                                  tokens=[Token(id=(w.id,), text=w.text, words=[w]) for w in words])])
    for p in pipe._processors:
        doc = pipe._run_processor(p, doc)
    pipe._reconcile_feats(doc)
    return doc


def test_later_tagger_owns_upos_and_feats_follow_it():
    doc = _run([_FakeMorph(lang="tur"), _FakePos(lang="tur")])
    ali, keldi = doc.words
    assert ali.upos == "NOUN" and ali.feats == "Case=Nom"  # morph agrees with tagger
    assert keldi.upos == "VERB" and keldi.feats is None      # disagreeing FEATS dropped
    assert keldi.lemma == "m_keldi"                           # morph still owns LEMMA
    assert keldi._provenance["morph"]["upos"] == "NOUN"       # morph output is kept


def test_earlier_processor_cannot_overwrite_owned_field():
    doc = _run([_FakePos(lang="tur"), _FakeMorph(lang="tur")])
    # morph runs last, so it owns UPOS here
    assert [w.upos for w in doc.words] == ["NOUN", "NOUN"]


# ---------------------------------------------------------------------------
# A6 / A7: script bridging


def test_uzbek_cyrillic_is_bridged_to_latin_models():
    pipe = Pipeline("uzb", script="Cyrl", processors=["tokenize"], tokenize_backend="rule")
    assert str(pipe._model_script) == "Latn"
    text = "Мен  мактабга бордим.  Исҳоқ келди!"
    doc = pipe(text)
    assert doc.script == "Cyrl" and doc.text == text
    for w in doc.words:
        assert text[w.start_char:w.end_char] == w.text
    assert [s.text for s in doc.sentences] == ["Мен  мактабга бордим.", "Исҳоқ келди!"]


def test_kazakh_latin_auto_detected_is_transliterated():
    pipe = Pipeline("kaz", processors=["tokenize"], tokenize_backend="rule")
    doc = pipe("Men mektepke bardym.")
    assert doc._model_text.startswith("Мен")
    assert [w.text for w in doc.words] == ["Men", "mektepke", "bardym", "."]
    assert doc.script == "Latn"


def test_catalog_scripts_match_script_config():
    from turkicnlp.scripts import get_script_config

    catalog = ModelRegistry.load_catalog()
    for lang in ("tuk", "uig", "ota"):
        cfg = {str(s) for s in get_script_config(lang).available} - {"CTS"}
        assert set(catalog[lang]["scripts"]["available"]) == cfg


def test_ota_uses_arabic_tokenizer():
    assert ModelRegistry.load_catalog()["ota"]["processors"]["Arab"]["tokenize"]["default"] == "rule_arabic"


# ---------------------------------------------------------------------------
# A9 / A10 / B11: Apertium morphology


def _apertium(lang="kaz"):
    from turkicnlp.processors.morphology import ApertiumMorphProcessor

    return ApertiumMorphProcessor(lang=lang)


def test_tagger_output_regex_matches():
    proc = _apertium()
    readings = [[{"lemma": "кітап", "pos": "n", "feats": ["nom"]}]]
    out = proc._parse_tagger_output("^кітап/кітап<n><nom>$", readings)
    assert out == [readings[0][0]]
    out = proc._parse_tagger_output("^кітап<n><nom>$", readings)
    assert out == [readings[0][0]]


def test_generate_sorts_hfst_tuple():
    class _Gen:
        def lookup(self, s):
            return (("b", 2.0), ("a", 1.0))

    proc = _apertium()
    proc._generator = _Gen()
    proc._needs_translit = False
    assert proc.generate("x", ["n"]) == "a"


def test_uzbek_apostrophe_is_not_a_name_suffix():
    proc = _apertium("uzb")
    res = proc._fallback_for_unknown("O'qing", [Word(id=1, text="Men"), Word(id=2, text="O'qing")], 1)
    assert not (res and res.get("lemma") == "O")


def test_turkish_name_suffix_uses_last_apostrophe():
    proc = _apertium("tur")
    res = proc._fallback_for_unknown("Kur'an'ı", [Word(id=1, text="Bu"), Word(id=2, text="Kur'an'ı")], 1)
    assert res["lemma"] == "Kur'an" and res["upos"] == "PROPN"


# ---------------------------------------------------------------------------
# B12: tag maps


def test_ind_depends_on_pos():
    from turkicnlp.resources.tag_mappings import load_tag_map

    m = load_tag_map("kaz")
    assert "Mood=Ind" in m.to_ud_feats(["ind"], pos="v")
    assert "PronType=Ind" in m.to_ud_feats(["ind"], pos="prn")


def test_aorist_and_derivation_n():
    from turkicnlp.resources.tag_mappings import load_tag_map

    m = load_tag_map("tur")
    feats = m.to_ud_feats(["aor", "p3", "sg"], pos="v")
    assert "Tense=Aor" not in feats and "Aspect=Hab" in feats
    assert "Gender" not in m.to_ud_feats(["ger_past", "n", "acc"], pos="v")


@pytest.mark.parametrize("lang", ["ota", "otk", "klj"])
def test_other_turkic_languages_use_common_mapper(lang):
    from turkicnlp.resources.tag_mappings import load_tag_map

    assert load_tag_map(lang).to_ud_feats(["dat"]) == "Case=Dat"


# ---------------------------------------------------------------------------
# D29: downloads and catalog


def test_fetch_is_atomic(tmp_path, monkeypatch):
    from turkicnlp.resources import downloader

    def boom(url, path, reporthook=None):
        with open(path, "w") as f:
            f.write("partial")
        raise OSError("connection reset")

    monkeypatch.setattr(downloader.urllib.request, "urlretrieve", boom)
    dest = tmp_path / "model.pt"
    with pytest.raises(OSError):
        downloader._fetch("http://x", dest)
    assert not dest.exists() and not list(tmp_path.iterdir())


def test_hf_snapshot_complete(tmp_path):
    from turkicnlp.resources.downloader import HF_COMPLETE_MARKER, hf_snapshot_complete

    (tmp_path / "config.json").write_text("{}")
    assert not hf_snapshot_complete(tmp_path)
    (tmp_path / "model.safetensors").write_text("x")
    assert hf_snapshot_complete(tmp_path)
    (tmp_path / "x.incomplete").write_text("")
    assert not hf_snapshot_complete(tmp_path)
    (tmp_path / HF_COMPLETE_MARKER).write_text("ok")
    assert hf_snapshot_complete(tmp_path)


def test_packaged_catalog_wins_over_cached(tmp_path, monkeypatch):
    (tmp_path / "catalog.json").write_text(json.dumps({"tur": {"name": "OLD"}}))
    monkeypatch.setenv("TURKICNLP_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(ModelRegistry, "_catalog", None)
    try:
        assert ModelRegistry.load_catalog()["tur"]["name"] == "Turkish"
    finally:
        ModelRegistry._catalog = None


def test_model_dir_override(tmp_path):
    with ModelRegistry.using_dir(tmp_path):
        assert ModelRegistry.default_dir() == tmp_path
    assert ModelRegistry.default_dir() != tmp_path


# ---------------------------------------------------------------------------
# D31: CoNLL-U


def test_conllu_roundtrip_keeps_single_script_and_valid_lines():
    from turkicnlp.utils.conllu import parse_conllu

    w = Word(id=1, text="Мен", script="Cyrl", misc="SpaceAfter=No")
    sent = Sentence(text="Мен\nбардым", tokens=[Token(id=(1,), text="Мен", words=[w])], words=[w])
    out = Document(text="", sentences=[sent]).to_conllu()
    assert "# text = Мен бардым" in out
    again = parse_conllu(out).to_conllu()
    assert again.count("Script=Cyrl") == 1
    assert parse_conllu(again).to_conllu() == again


def test_conllu_reader_robustness():
    from turkicnlp.utils.conllu import parse_conllu

    data = (
        "﻿# text = a b\n"
        "1-2\tab\t_\t_\t_\t_\t_\t_\t_\t_\n"
        "1\ta\t_\t_\t_\t_\tX\t_\t_\t_\n"
        "\n"
        "1\tc\t_\t_\t_\t_\t0\troot\t_\t_\n"
        "\n"
    )
    doc = parse_conllu(data)
    assert doc.sentences[0].text == "a b"
    assert doc.sentences[0].words[0].head is None
    assert len(doc.sentences[1].tokens) == 1 and not doc.sentences[1].tokens[0].is_mwt


def test_to_dict_full():
    w = Word(id=1, text="Ali", ner="B-PER", misc="SpaceAfter=No")
    sent = Sentence(text="Ali", words=[w], translation="Ali", sentiment="neutral")
    doc = Document(text="Ali", sentences=[sent], translation="Ali")
    assert doc.to_dict()[0][0]["ner"] == "B-PER"
    full = doc.to_dict(full=True)
    assert full["translation"] == "Ali"
    assert full["sentences"][0]["sentiment"] == "neutral"
    json.dumps(full)


# ---------------------------------------------------------------------------
# D32: batch and repr


def test_batch_uses_batch_size_and_matches_single_calls():
    nlp = Pipeline("kaz", processors=["tokenize"], tokenize_backend="rule", script="Cyrl")
    texts = ["Мен бардым.", "", "Сен келдің. Ол кетті."]
    batch = nlp.batch(texts, batch_size=2)
    single = [nlp(t) for t in texts]
    assert [[w.text for w in d.words] for d in batch] == [[w.text for w in d.words] for d in single]
    with pytest.raises(ValueError):
        nlp.batch(texts, batch_size=0)


def test_repr_before_loading_lists_plan():
    r = repr(Pipeline("kjh"))
    assert "tokenize:rule" in r and "not loaded" in r


def test_split_audio_empty():
    from turkicnlp.asr import split_audio

    assert split_audio([], 16000) == []
