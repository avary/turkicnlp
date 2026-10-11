"""LLM-based processors.

Every class here is an ordinary TurkicNLP processor whose backend is a chat LLM reached
through :class:`turkicnlp.llm.LLMClient`. The prompt is a packaged template
(``turkicnlp/resources/prompts/*.json``) that the processor fills with the tokens /
sentence and the language; the model's JSON answer is validated and written into the
normal :class:`~turkicnlp.models.document.Document` fields.

==============================  =========================  =================================
processor / backend             writes                     prompt template
==============================  =========================  =================================
``morph`` / ``llm``             lemma, upos, feats         ``morph.json``
``morph`` / ``apertium_llm``    lemma, upos, feats         ``morph_disambiguate.json`` (LLM
                                                           picks one of the Apertium readings)
``ner`` / ``llm``               word.ner (BIO), entities   ``ner.json``
``translate`` / ``llm``         sentence/doc.translation   ``translate.json``
``sentiment`` / ``llm``         sentence.sentiment         ``sentiment.json``
``spellcheck`` / ``llm``        sentence.correction        ``spellcheck.json``
==============================  =========================  =================================

Example::

    nlp = turkicnlp.Pipeline("kaz", processors=["tokenize", "ner"], ner_backend="llm",
                             llm_model="Qwen/Qwen3-8B", llm_base_url="http://gpu:8000/v1",
                             llm_api_key="...")
"""

from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Optional

from turkicnlp.llm.client import LLMClient, LLMError
from turkicnlp.llm.prompts import PromptTemplate, base_variables, target_language_name, tokens_json
from turkicnlp.models.document import Document, Sentence, Span
from turkicnlp.processors.base import Processor
from turkicnlp.processors.morphology import ApertiumMorphProcessor

logger = logging.getLogger(__name__)

UPOS = {"ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
        "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"}
_FEAT = re.compile(r"^[A-Z][A-Za-z]*(\[[a-z0-9]+\])?=[A-Z0-9][A-Za-z0-9]*(,[A-Z0-9][A-Za-z0-9]*)*$")


def normalize_feats(feats: Any) -> str:
    """Keep well-formed UD ``Name=Value`` pairs, sorted as in CoNLL-U."""
    if isinstance(feats, dict):
        feats = "|".join(f"{k}={v}" for k, v in feats.items())
    if not isinstance(feats, str) or feats.strip() in ("", "_"):
        return "_"
    pairs = {}
    for part in feats.replace(";", "|").split("|"):
        part = part.strip()
        if _FEAT.match(part):
            name, value = part.split("=", 1)
            pairs.setdefault(name, value)
    return "|".join(f"{k}={pairs[k]}" for k in sorted(pairs, key=str.lower)) or "_"


def _list_field(data: Any, key: str) -> list:
    """``data[key]`` as a list; also accepts a bare list or a dict with one list value
    (small models often rename the key)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get(key), list):
            return data[key]
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    raise ValueError(f"expected a JSON object with a '{key}' list")


def _sentences(doc: Document) -> list[Sentence]:
    """Sentences of the document; a document without tokenisation counts as one sentence."""
    if not doc.sentences:
        doc.sentences = [Sentence(text=doc.text)]
    return doc.sentences


class LLMProcessor(Processor):
    """Shared setup: client, prompt template, few-shot examples, error policy.

    Config keys (pipeline options without the ``<processor>_`` prefix):
    ``llm_*`` (client, see :class:`turkicnlp.llm.LLMConfig`), ``llm_client`` (reuse a
    client), ``prompt_file``, ``examples``, ``n_examples`` and ``on_error``
    (``warn`` = leave the fields empty, ``raise``).
    """

    PROMPT = ""
    JSON_SCHEMA: Optional[dict] = None

    def load(self, model_path: str = "") -> None:
        self.client = LLMClient.from_options(self.config)
        self.prompt = PromptTemplate.load(self.PROMPT, self.config.get("prompt_file"))
        self.extra_examples = self.config.get("examples")
        self.n_examples = int(self.config.get("n_examples", 3))
        self.on_error = self.config.get("on_error", "warn")
        self._loaded = True

    def _messages(self, doc: Document, **variables) -> list[dict]:
        vars_ = {**base_variables(self.lang, doc.script or self.script), **variables}
        return self.prompt.render(vars_, lang=self.lang, extra_examples=self.extra_examples,
                                  n_examples=self.n_examples)

    def _run(self, requests: list[list[dict]], validate) -> list[Any]:
        results = self.client.map_json(requests, json_schema=self.JSON_SCHEMA, validate=validate)
        failures = [r for r in results if isinstance(r, LLMError)]
        if failures:
            if self.on_error == "raise":
                raise failures[0]
            logger.warning("%s/llm: %d of %d LLM answers were unusable (%s); fields left empty",
                           self.NAME, len(failures), len(results), failures[0])
        return results

    def _log(self, doc: Document, n_failed: int = 0) -> None:
        tag = f"{self.NAME}:llm({self.client.model}, {self.prompt.tag})"
        if n_failed:
            tag += f"[{n_failed} failed]"
        doc._processor_log.append(tag)


# --------------------------------------------------------------------------------------
# Morphology: lemma + UPOS + features
# --------------------------------------------------------------------------------------

class LLMMorphAnalyzer(LLMProcessor):
    NAME = "morph"
    PROVIDES = ["lemma", "pos", "feats"]
    REQUIRES = ["tokenize"]
    PROMPT = "morph"
    JSON_SCHEMA = {
        "type": "object",
        "properties": {"words": {"type": "array", "items": {
            "type": "object",
            "properties": {"i": {"type": "integer"}, "token": {"type": "string"}, "lemma": {"type": "string"},
                           "upos": {"type": "string"}, "feats": {"type": "string"}},
            "required": ["i", "token", "lemma", "upos", "feats"]}}},
        "required": ["words"],
    }

    def process(self, doc: Document) -> Document:
        self.check_requirements(doc)
        sents = [s for s in doc.sentences if s.words]
        requests = [self._messages(doc, tokens_json=tokens_json([w.text for w in s.words]), text=s.text)
                    for s in sents]
        validators = [(lambda data, toks=[w.text for w in s.words]: self._validate(data, toks)) for s in sents]
        results = self._run(requests, validators)
        failed = 0
        for s, res in zip(sents, results):
            if isinstance(res, LLMError):
                failed += 1
                continue
            for w, (lemma, upos, feats) in zip(s.words, res):
                w.lemma, w.upos, w.feats = lemma or w.text, upos, feats
        self._log(doc, failed)
        return doc

    @staticmethod
    def _validate(data: Any, tokens: list[str]) -> list[tuple[Optional[str], str, str]]:
        """Align the answer with the tokens.

        Entries are matched by their ``token`` text, so a model that splits
        "Қасым-Жомарт" into two entries cannot shift every later analysis by one;
        entries without ``token`` are matched by index. Unanalysed punctuation is filled
        in; any other missing token makes the answer invalid (and triggers a retry).
        """
        items = [x for x in _list_field(data, "words") if isinstance(x, dict)]
        n = len(tokens)
        by_index: dict[int, dict] = {}
        if items and all("token" in x for x in items):
            j = 0
            for item in items:
                t = str(item["token"]).strip().casefold()
                for k in range(j, min(n, j + 3)):
                    if tokens[k].casefold() == t:
                        by_index[k] = item
                        j = k + 1
                        break
        else:
            for pos, item in enumerate(items):
                idx = item.get("i", pos)
                if isinstance(idx, int) and 0 <= idx < n:
                    by_index.setdefault(idx, item)
        missing = [k for k in range(n) if k not in by_index]
        for k in list(missing):
            if all(not ch.isalnum() for ch in tokens[k]):
                by_index[k] = {"lemma": tokens[k], "upos": "PUNCT", "feats": "_"}
                missing.remove(k)
        if missing:
            raise ValueError(f"no analysis for tokens {[tokens[k] for k in missing][:5]}; "
                             f"give exactly one entry per token and copy each token exactly")
        out = []
        for i in range(n):
            item = by_index[i]
            upos = str(item.get("upos", "X")).upper().strip()
            lemma = str(item.get("lemma") or "").strip()
            out.append((lemma or None, upos if upos in UPOS else "X", normalize_feats(item.get("feats"))))
        return out


# --------------------------------------------------------------------------------------
# Morphology: Apertium analyses, LLM chooses
# --------------------------------------------------------------------------------------

class ApertiumLLMMorphProcessor(ApertiumMorphProcessor):
    """Apertium HFST analysis; ambiguous words are disambiguated by the LLM.

    The LLM only *chooses* among the transducer's readings, so every analysis is one the
    transducer licenses. A configured ``apertium_tagger_cmd`` keeps working: words the
    tagger resolves are not sent to the LLM. All sentences of a document are sent as one
    parallel batch. Readings are shown to the LLM in the document's script (lemmas are
    transliterated back when the transducer uses another script). When the LLM fails,
    the heuristic disambiguation is used.
    Extra config: ``max_options`` (readings shown per word, default 8).
    """

    PROMPT = "morph_disambiguate"
    JSON_SCHEMA = {
        "type": "object",
        "properties": {"choices": {"type": "array", "items": {
            "type": "object", "properties": {"i": {"type": "integer"}, "option": {"type": "integer"}},
            "required": ["i", "option"]}}},
        "required": ["choices"],
    }

    def load(self, model_path: str = "") -> None:
        if not model_path:
            model_path = self._apertium_path()
        super().load(model_path)
        self.client = LLMClient.from_options(self.config)
        self.prompt = PromptTemplate.load(self.PROMPT, self.config.get("prompt_file"))
        self.extra_examples = self.config.get("examples")
        self.n_examples = int(self.config.get("n_examples", 3))
        self.max_options = int(self.config.get("max_options", 8))
        self.on_error = self.config.get("on_error", "warn")
        self._llm_calls = self._llm_failures = 0

    def _apertium_path(self) -> str:
        from turkicnlp.resources.downloader import download
        from turkicnlp.resources.registry import ModelRegistry

        script = str(self.script or self._apertium_script or get_primary_script(self.lang))
        try:
            return str(ModelRegistry.get_model_path(self.lang, "morph", "apertium", script=script))
        except FileNotFoundError:
            download(self.lang, processors=["morph"], script=script)
            return str(ModelRegistry.get_model_path(self.lang, "morph", "apertium", script=script))

    def _user_lemma(self, lemma: str) -> str:
        """A transducer lemma in the document's script."""
        back = getattr(self, "_from_fst_translit", None)
        if getattr(self, "_needs_translit", False) and back is not None:
            try:
                return back.transliterate(lemma)
            except Exception:  # noqa: BLE001
                return lemma
        return lemma

    def _describe(self, reading: dict) -> str:
        upos = self._tag_mapper.to_ud_pos(reading["pos"])
        feat_tags = list(reading["feats"])
        if reading["pos"] in ("qst", "encl"):
            feat_tags.insert(0, "qst")
        feats = self._normalize_ud_feats_for_upos(upos, self._tag_mapper.to_ud_feats(feat_tags, pos=reading["pos"]))
        return f"{self._user_lemma(reading['lemma'])} {upos} {feats}"

    # -- per-sentence readings (same rules as ApertiumMorphProcessor.process) ----------

    def _sentence_readings(self, sentence) -> list[Optional[list[dict]]]:
        out: list[Optional[list[dict]]] = []
        for word in sentence.words:
            if word.upos == "PUNCT" or self._is_punctuation_token(word.text):
                out.append(None)
                continue
            surface = word.text
            if self._needs_translit and self._to_fst_translit:
                surface = self._to_fst_translit.transliterate(surface)
            out.append(self._analyze_with_fallback(surface))
        return out

    def _ambiguous(self, readings_by_word, resolved=None) -> dict[int, list[tuple[dict, str]]]:
        ambiguous: dict[int, list[tuple[dict, str]]] = {}
        for i, readings in enumerate(readings_by_word):
            if not readings or len(readings) < 2:
                continue
            if resolved is not None and resolved[i] is not None:
                continue  # the Apertium tagger already chose
            seen: dict[str, dict] = {}
            for r in readings:
                seen.setdefault(self._describe(r), r)
            if len(seen) >= 2:
                ambiguous[i] = [(r, d) for d, r in list(seen.items())[: self.max_options]]
        return ambiguous

    def _request(self, sentence_words, ambiguous) -> list[dict]:
        lines = []
        for i, opts in ambiguous.items():
            lines.append(f'word {i} "{sentence_words[i].text}":')
            lines.extend(f"  {k}: {desc}" for k, (_, desc) in enumerate(opts))
        tokens = [w.text for w in sentence_words]
        variables = {**base_variables(self.lang, self.script), "tokens_json": tokens_json(tokens),
                     "options_text": "\n".join(lines)}
        return self.prompt.render(variables, lang=self.lang, extra_examples=self.extra_examples,
                                  n_examples=self.n_examples)

    @staticmethod
    def _validator(ambiguous):
        def validate(data):
            choices = {}
            for c in _list_field(data, "choices"):
                try:
                    i, k = int(c["i"]), int(c["option"])
                except (KeyError, TypeError, ValueError):
                    continue
                if i in ambiguous and 0 <= k < len(ambiguous[i]):
                    choices[i] = k
            if not choices:
                raise ValueError("no valid choice")
            return choices
        return validate

    def _tagger_disambiguate(self, sentence_words, readings_by_word):
        """Apertium tagger (if configured) first, then the LLM's choices for the rest."""
        pending = getattr(self, "_pending", None) or {}
        if id(sentence_words) in pending:  # computed in one batch by process()
            tagged, llm = pending[id(sentence_words)]
        else:
            tagged = super()._tagger_disambiguate(sentence_words, readings_by_word)
            llm = self._disambiguate_with_llm([(sentence_words, readings_by_word, tagged)])[0]
        if tagged is None:
            return llm
        if llm is None:
            return tagged
        return [t if t is not None else c for t, c in zip(tagged, llm)]

    def _disambiguate_with_llm(self, items) -> list[Optional[list[Optional[dict]]]]:
        """One parallel LLM batch for ``[(words, readings_by_word, tagger_result), ...]``."""
        jobs, ambig = [], []
        for words, readings_by_word, tagged in items:
            amb = self._ambiguous(readings_by_word, tagged)
            ambig.append(amb)
            if amb:
                jobs.append((len(ambig) - 1, self._request(words, amb), self._validator(amb)))
        out: list[Optional[list[Optional[dict]]]] = [None] * len(items)
        if not jobs:
            return out
        self._llm_calls += len(jobs)
        results = self.client.map_json([j[1] for j in jobs], json_schema=self.JSON_SCHEMA,
                                       validate=[j[2] for j in jobs])
        failures = [r for r in results if isinstance(r, LLMError)]
        if failures:
            self._llm_failures += len(failures)
            if self.on_error == "raise":
                raise failures[0]
            logger.warning("morph/apertium_llm: %d of %d LLM answers unusable, heuristic disambiguation "
                           "used for them (%s)", len(failures), len(results), failures[0])
        for (idx, _, _), choices in zip(jobs, results):
            if isinstance(choices, LLMError):
                continue
            amb = ambig[idx]
            n = len(items[idx][1])
            out[idx] = [amb[i][choices[i]][0] if i in choices else None for i in range(n)]
        return out

    def process(self, doc: Document) -> Document:
        self.check_requirements(doc)
        self._pending = {}
        if self._analyzer is not None:
            sents = [s for s in doc.sentences if s.words]
            items = []
            for s in sents:
                readings = self._sentence_readings(s)
                tagged = ApertiumMorphProcessor._tagger_disambiguate(self, s.words, readings)
                items.append((s.words, readings, tagged))
            for (words, _, tagged), res in zip(items, self._disambiguate_with_llm(items)):
                self._pending[id(words)] = (tagged, res)
        try:
            doc = super().process(doc)
        finally:
            self._pending = {}
        if doc._processor_log and doc._processor_log[-1].startswith("morph:apertium"):
            doc._processor_log[-1] += f"+llm({self.client.model}, {self.prompt.tag})"
        return doc


def get_primary_script(lang: str) -> str:
    from turkicnlp.scripts import get_script_config

    return str(get_script_config(lang).primary)


# --------------------------------------------------------------------------------------
# Named entities
# --------------------------------------------------------------------------------------

DEFAULT_NER_LABELS = ["PER", "LOC", "ORG"]
_LABEL_HELP = {
    "PER": "PER = people (also fictional characters)",
    "LOC": "LOC = places: countries, regions, cities, rivers, mountains",
    "ORG": "ORG = organisations: companies, institutions, parties, media, teams",
    "GPE": "GPE = countries, cities and states as political entities",
    "MISC": "MISC = other names: events, works, products, nationalities used as names",
    "DATE": "DATE = dates and periods",
}


class LLMNERProcessor(LLMProcessor):
    NAME = "ner"
    PROVIDES = ["ner"]
    REQUIRES = ["tokenize"]
    PROMPT = "ner"
    JSON_SCHEMA = {
        "type": "object",
        "properties": {"entities": {"type": "array", "items": {
            "type": "object",
            "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}, "label": {"type": "string"}},
            "required": ["start", "end", "label"]}}},
        "required": ["entities"],
    }

    def load(self, model_path: str = "") -> None:
        super().load(model_path)
        self.labels = [str(l).upper() for l in self.config.get("labels", DEFAULT_NER_LABELS)]

    def process(self, doc: Document) -> Document:
        self.check_requirements(doc)
        sents = [s for s in doc.sentences if s.words]
        label_help = "\n".join(_LABEL_HELP.get(l, f"{l} = {l.lower()} names") for l in self.labels)
        requests = [self._messages(doc, tokens_json=tokens_json([w.text for w in s.words]), text=s.text,
                                   labels=", ".join(self.labels), label_guide=label_help) for s in sents]
        validators = [(lambda data, n=len(s.words): self._validate(data, n)) for s in sents]
        results = self._run(requests, validators)
        failed = 0
        for s, res in zip(sents, results):
            if isinstance(res, LLMError):
                failed += 1
                continue
            spans = res
            tags = ["O"] * len(s.words)
            entities = []
            for start, end, label in spans:
                tags[start] = f"B-{label}"
                for j in range(start + 1, end):
                    tags[j] = f"I-{label}"
                words = s.words[start:end]
                entities.append(Span(text=" ".join(w.text for w in words), type=label,
                                     start_char=words[0].start_char or 0,
                                     end_char=words[-1].end_char or 0, words=list(words)))
            for w, t in zip(s.words, tags):
                w.ner = t
            s.entities = entities
        self._log(doc, failed)
        return doc

    def _validate(self, data: Any, n: int) -> list[tuple[int, int, str]]:
        """Valid, non-overlapping spans. Malformed entities (missing / non-integer
        bounds or label) and unusable ones (unknown label, out of range) are dropped one
        by one; the answer is only rejected (and the request retried) when it had
        entities and every one of them was malformed."""
        items = _list_field(data, "entities")
        parsed, malformed, unusable = [], 0, 0
        for e in items:
            try:
                start, end = int(e["start"]), int(e["end"])
                label = str(e["label"]).upper().strip()
            except (KeyError, TypeError, ValueError, IndexError):
                malformed += 1
                continue
            if end == start:
                end = start + 1  # some models give inclusive single-token spans
            if not (0 <= start < end <= n) or label not in self.labels:
                unusable += 1
                continue
            parsed.append((start, end, label))
        if malformed and malformed == len(items):
            raise ValueError("malformed entities: give integer 'start'/'end' token indices and a 'label'")
        if malformed or unusable:
            logger.info("ner/llm: dropped %d malformed / %d unusable entities, kept %d",
                        malformed, unusable, len(parsed))
        taken = [False] * n
        spans = []
        for start, end, label in sorted(parsed, key=lambda x: (x[0], -x[1])):
            if any(taken[start:end]):
                continue
            for j in range(start, end):
                taken[j] = True
            spans.append((start, end, label))
        return spans


# --------------------------------------------------------------------------------------
# Translation, sentiment, spelling
# --------------------------------------------------------------------------------------

def _require_str(key: str):
    def validate(data):
        value = data[key] if isinstance(data, dict) else data
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{key}' must be a non-empty string")
        return value.strip()
    return validate


class LLMTranslateProcessor(LLMProcessor):
    """Config: ``tgt_lang`` (ISO 639-3 code, code with script such as ``uzb_Latn`` /
    ``uzb_Cyrl``, or a language name; default ``eng``) and ``tgt_script`` (e.g. ``Cyrl``).
    The prompt names the target script when it is known, e.g. "Uzbek (Latin script)"."""

    NAME = "translate"
    PROVIDES = ["translation"]
    REQUIRES = []
    PROMPT = "translate"
    JSON_SCHEMA = {"type": "object", "properties": {"translation": {"type": "string"}},
                   "required": ["translation"]}

    def process(self, doc: Document) -> Document:
        tgt = str(self.config.get("tgt_lang", "eng"))
        tgt_name = target_language_name(tgt, self.config.get("tgt_script"))
        sents = _sentences(doc)
        requests = [self._messages(doc, text=s.text, target_language=tgt_name) for s in sents]
        results = self._run(requests, _require_str("translation"))
        failed = 0
        for s, res in zip(sents, results):
            if isinstance(res, LLMError):
                failed += 1
                continue
            s.translation = res
        doc.translation = " ".join(s.translation for s in sents if s.translation) or None
        self._log(doc, failed)
        return doc


class LLMSentimentProcessor(LLMProcessor):
    """Config: ``labels`` (default positive / negative / neutral)."""

    NAME = "sentiment"
    PROVIDES = ["sentiment"]
    REQUIRES = []
    PROMPT = "sentiment"
    JSON_SCHEMA = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}

    def process(self, doc: Document) -> Document:
        labels = [str(l).lower() for l in self.config.get("labels", ["positive", "negative", "neutral"])]

        def validate(data):
            label = str(data["label"] if isinstance(data, dict) else data).lower().strip()
            if label not in labels:
                raise ValueError(f"label must be one of {labels}")
            return label

        if labels != ["positive", "negative", "neutral"] and not self.extra_examples:
            self.n_examples = 0  # the packaged examples use the default labels
        sents = _sentences(doc)
        requests = [self._messages(doc, text=s.text, labels=", ".join(labels)) for s in sents]
        results = self._run(requests, validate)
        failed = 0
        for s, res in zip(sents, results):
            if isinstance(res, LLMError):
                failed += 1
            else:
                s.sentiment = res
        self._log(doc, failed)
        return doc


def _char_distance(a: str, b: str) -> int:
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def _word_cores(text: str) -> list[tuple[str, int, int]]:
    """Whitespace-separated words without leading/trailing punctuation, with the
    character offsets of that core in ``text``; pure punctuation tokens are skipped."""
    out = []
    for m in re.finditer(r"\S+", text):
        s, e = m.start(), m.end()
        while s < e and _is_punct(text[s]):
            s += 1
        while e > s and _is_punct(text[e - 1]):
            e -= 1
        if s < e:
            out.append((text[s:e], s, e))
    return out


class LLMSpellcheckProcessor(LLMProcessor):
    """Spelling correction per sentence.

    Writes ``sentence.correction`` (corrected text) and ``sentence.corrections``
    (``[(original, corrected), ...]`` word pairs); ``doc.corrected_text`` joins them.
    Config: ``max_change`` (default 0.3) — an answer that changes a larger share of the
    characters is treated as a rewrite and discarded; ``max_edits_per_word`` (default 2)
    — word changes needing more letter edits are reverted.

    Words are compared without the punctuation attached to them, so "окыдым" →
    "оқыдым." counts as a one-letter correction; punctuation itself is never changed,
    and the corrected words are written into the original sentence, keeping its
    spacing, line breaks and punctuation.
    """

    NAME = "spellcheck"
    PROVIDES = ["correction"]
    REQUIRES = []
    PROMPT = "spellcheck"
    JSON_SCHEMA = {"type": "object", "properties": {"corrected": {"type": "string"}}, "required": ["corrected"]}

    def process(self, doc: Document) -> Document:
        max_change = float(self.config.get("max_change", 0.3))
        sents = _sentences(doc)
        requests = [self._messages(doc, text=s.text) for s in sents]
        results = self._run(requests, _require_str("corrected"))
        failed = rejected = 0
        for s, res in zip(sents, results):
            if isinstance(res, LLMError):
                failed += 1
                continue
            src_words = " ".join(c for c, _, _ in _word_cores(s.text))
            hyp_words = " ".join(c for c, _, _ in _word_cores(res))
            if _char_distance(src_words, hyp_words) > max_change * max(len(src_words), 1):
                rejected += 1
                res = s.text
            res, changes, dropped = self._safe_changes(s.text, res)
            rejected += bool(dropped)
            s.correction = res
            s.corrections = changes
        if rejected:
            logger.info("spellcheck/llm: rewrites discarded in %d sentence(s)", rejected)
        self._log(doc, failed)
        return doc

    def _safe_changes(self, src: str, hyp: str) -> tuple[str, list[tuple[str, str]], int]:
        """Keep only word changes that look like spelling corrections.

        Words (without attached punctuation) of the answer are aligned with the source
        one-to-one, one-to-two (a split) or two-to-one (a merge). A change is kept if it
        needs at most ``max_edits_per_word`` (default 2) letter edits and at most one per
        four letters; anything else — rewording, changed suffixes, added or dropped words
        — is reverted to the source text. Kept changes replace the words in ``src`` at
        their offsets, so whitespace and punctuation of the source are preserved.
        """
        src_cores = _word_cores(src)
        a = [c for c, _, _ in src_cores]
        b = [c for c, _, _ in _word_cores(hyp)]
        edits: list[tuple[int, int, str]] = []  # (start, end, replacement) in src
        changes: list[tuple[str, str]] = []
        dropped = 0

        max_edits = int(self.config.get("max_edits_per_word", 2))

        def ok(orig: str, new: str) -> bool:
            o, n_ = orig.replace(" ", ""), new.replace(" ", "")
            return _char_distance(o, n_) <= min(max_edits, max(1, len(o) // 4))

        for op, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            orig, new = a[i1:i2], b[j1:j2]
            if op == "equal":
                continue
            i = j = 0
            while i < len(orig):
                options = []
                for di, dj in ((1, 1), (1, 2), (2, 1)):
                    if i + di <= len(orig) and j + dj <= len(new):
                        o, n_ = " ".join(orig[i:i + di]), " ".join(new[j:j + dj])
                        dist = _char_distance(o.replace(" ", ""), n_.replace(" ", ""))
                        options.append((dist + (di != dj), di, dj, o, n_))
                best = min(options) if options else None
                if best and ok(best[3], best[4]):
                    _, di, dj, o, n_ = best
                    if o != n_:
                        start = src_cores[i1 + i][1]
                        end = src_cores[i1 + i + di - 1][2]
                        edits.append((start, end, n_))
                        changes.append((src[start:end] if di == 1 else o, n_))
                    i, j = i + di, j + dj
                else:
                    dropped += 1
                    i, j = i + 1, j + 1
            if j < len(new):
                dropped += 1  # extra words in the answer are ignored
        out = src
        for start, end, rep in sorted(edits, reverse=True):
            out = out[:start] + rep + out[end:]
        return out, changes, dropped
