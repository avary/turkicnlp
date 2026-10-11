"""Tests for the LLM client and the LLM-based processors.

A small OpenAI-compatible HTTP server runs in a thread; it answers according to the
prompt it receives, so the full path (pipeline → processor → prompt → HTTP → JSON →
Document) is tested without network access or a real model.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from turkicnlp.llm import LLMClient, LLMError, PromptTemplate, ask, extract_json
from turkicnlp.llm.client import LLMConfig
from turkicnlp.processors.llm_backend import normalize_feats

REQUESTS: list[dict] = []


def _answer_for(body: dict) -> str:
    system = body["messages"][0]["content"]
    user = body["messages"][-1]["content"]
    if "named entities" in system:
        tokens = json.loads(user.split("Tokens: ", 1)[1])
        ents = [{"start": i, "end": i + 1, "label": "LOC"} for i, t in tokens if t.startswith("Алмат")]
        return json.dumps({"entities": ents})
    if "morphology" in system and "Tokens:" in user:
        tokens = json.loads(user.split("Tokens: ", 1)[1])
        words = [{"i": i, "token": t, "lemma": t.lower().rstrip("."), "upos": "PUNCT" if t == "." else "NOUN",
                  "feats": "Number=Sing|Case=Nom" if t != "." else "_"} for i, t in tokens]
        return "```json\n" + json.dumps({"words": words}, ensure_ascii=False) + "\n```"
    if "Ambiguous words" in user:
        first = re.search(r"word (\d+)", user.split("Ambiguous words:")[1]).group(1)
        return json.dumps({"choices": [{"i": int(first), "option": 1}]})
    if "translator" in system:
        return json.dumps({"translation": f"[{system.split(' into ')[1].split('.')[0]}] {user}"})
    if "sentiment" in system:
        return json.dumps({"label": "negative" if "нашар" in user else "neutral"})
    if "proofreader" in system:
        if "REWRITE" in user:
            return json.dumps({"corrected": "Something completely different and much longer."})
        return json.dumps({"corrected": user.replace("китап", "кітап")}, ensure_ascii=False)
    return "Free-form answer."


class _Handler(BaseHTTPRequestHandler):
    reject_schema = False
    reject_object = False
    fail_first = 0

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if _Handler.fail_first > 0:
            _Handler.fail_first -= 1
            self.send_response(503)
            self.end_headers()
            return
        if body.get("model") == "reject-temperature" and "temperature" in body:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "temperature: only 1 is supported for this model"}')
            return
        rf = body.get("response_format", {})
        user = body["messages"][-1]["content"]
        if body.get("model") == "too-long":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "This model\'s maximum context length is 8192 tokens (response_format ok)"}')
            return
        if (_Handler.reject_schema and rf.get("type") == "json_schema") or \
                (_Handler.reject_object and rf.get("type") == "json_object"):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "response_format of this type is not supported"}')
            return
        if "SEND-HTML" in user:
            out = b"<html><body>502 Bad Gateway</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        if "SEND-TRUNCATED" in user:
            out = b'{"choices": [{"message": {"content": "{\\"label\\": \\"neu'
            self.send_response(200)
            self.send_header("Content-Length", str(len(out) + 50))  # body ends early
            self.end_headers()
            self.wfile.write(out)
            return
        content = _answer_for(body)
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):  # silence
        pass


class _Server(ThreadingHTTPServer):
    request_queue_size = 64  # parallel client requests must not be refused
    daemon_threads = True


@pytest.fixture(scope="module")
def server():
    httpd = _Server(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


@pytest.fixture()
def opts(server, tmp_path):
    REQUESTS.clear()
    _Handler.reject_schema = False
    _Handler.reject_object = False
    _Handler.fail_first = 0
    return {"llm_model": "fake-model", "llm_base_url": server, "llm_api_key": "secret-key",
            "llm_cache_dir": str(tmp_path), "llm_max_retries": 2}


# --------------------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------------------

def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('Sure! Here it is: {"x": "y {z}"} hope it helps') == {"x": "y {z}"}
    assert extract_json("<think>reasoning {not json}</think>\n{\"ok\": true}") == {"ok": True}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_client_sends_auth_and_caches(opts):
    client = LLMClient.from_options(opts)
    msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": "hello"}]
    assert client.chat(msgs) == "Free-form answer."
    assert REQUESTS[-1]["auth"] == "Bearer secret-key"
    assert REQUESTS[-1]["path"] == "/v1/chat/completions"
    assert REQUESTS[-1]["body"]["model"] == "fake-model"
    n = len(REQUESTS)
    client.chat(msgs)
    assert len(REQUESTS) == n  # served from the cache


def test_client_falls_back_when_schema_unsupported(opts):
    _Handler.reject_schema = True
    client = LLMClient.from_options({**opts, "llm_cache": False})
    schema = {"type": "object", "properties": {"label": {"type": "string"}}}
    data = client.chat_json([{"role": "system", "content": "sentiment"}, {"role": "user", "content": "x"}],
                            json_schema=schema)
    assert data == {"label": "neutral"}
    assert REQUESTS[-1]["body"]["response_format"] == {"type": "json_object"}


def test_client_retries_server_errors(opts):
    _Handler.fail_first = 1
    client = LLMClient.from_options({**opts, "llm_cache": False})
    assert client.chat([{"role": "user", "content": "hi"}]) == "Free-form answer."


def test_client_requires_model(monkeypatch):
    monkeypatch.delenv("TURKICNLP_LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="No LLM model"):
        LLMClient.from_options({"llm_base_url": "http://x"})


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("TURKICNLP_LLM_MODEL", "m")
    monkeypatch.setenv("TURKICNLP_LLM_BASE_URL", "http://host:1/v1")
    monkeypatch.setenv("TURKICNLP_LLM_API_KEY", "k")
    cfg = LLMConfig.from_options({})
    assert (cfg.model, cfg.base_url, cfg.api_key) == ("m", "http://host:1/v1", "k")
    assert "api_key" not in cfg.public_dict()


def test_unreachable_server_raises(tmp_path):
    client = LLMClient(model="m", base_url="http://127.0.0.1:9/v1", max_retries=0, timeout=2,
                       cache_dir=str(tmp_path))
    with pytest.raises(LLMError, match="cannot reach"):
        client.chat([{"role": "user", "content": "hi"}])


def test_prompt_templates_render_for_every_language():
    from turkicnlp.llm.prompts import base_variables, tokens_json

    for name in ["ner", "morph", "morph_disambiguate", "translate", "sentiment", "spellcheck", "ask"]:
        tpl = PromptTemplate.load(name)
        for lang in ["tur", "kaz", "kjh", "uig", "otk", "klj"]:
            msgs = tpl.render({**base_variables(lang, "Cyrl"), "tokens_json": tokens_json(["a"]), "text": "a",
                               "labels": "x", "label_guide": "", "target_language": "English",
                               "options_text": "", "context": "", "question": "q"}, lang=lang)
            assert msgs[0]["role"] == "system" and msgs[-1]["role"] == "user"
            assert "{language}" not in msgs[0]["content"]


def test_normalize_feats():
    assert normalize_feats("Number=Sing|Case=Dat") == "Case=Dat|Number=Sing"
    assert normalize_feats("Case=Dat|garbage|Person[psor]=1") == "Case=Dat|Person[psor]=1"
    assert normalize_feats({"Case": "Nom"}) == "Case=Nom"
    assert normalize_feats("") == "_" and normalize_feats(None) == "_"


# --------------------------------------------------------------------------------------
# processors through the pipeline
# --------------------------------------------------------------------------------------

def _pipeline(lang, processors, opts, **kw):
    """Pipeline with the rule-based tokenizer: needs no Stanza model download."""
    import turkicnlp

    kw.setdefault("tokenize_backend", "rule")
    return turkicnlp.Pipeline(lang, processors=processors, **opts, **kw)


def test_llm_ner_in_pipeline(opts):
    nlp = _pipeline("kaz", ["tokenize", "ner"], opts, ner_backend="llm")
    doc = nlp("Мен Алматыда тұрамын.")
    assert [(e.text, e.type) for e in doc.entities] == [("Алматыда", "LOC")]
    assert [w.ner for w in doc.words][:3] == ["O", "B-LOC", "O"]
    assert doc._processor_log[-1].startswith("ner:llm(fake-model, ner/v1)")
    sent = REQUESTS[-1]["body"]["messages"]
    assert "Kazakh" in sent[0]["content"] and "Cyrillic" in sent[0]["content"]
    assert len(sent) == 1 + 2 * 3 + 1  # system + 3 few-shot pairs + input


def test_llm_morph_for_a_small_language(opts):
    nlp = _pipeline("kjh", ["tokenize", "morph"], opts, morph_backend="llm")
    doc = nlp("Пу прай наа чуртас.")
    assert [w.upos for w in doc.words] == ["NOUN", "NOUN", "NOUN", "NOUN", "PUNCT"]
    assert doc.words[0].lemma == "пу"
    assert doc.words[0].feats == "Case=Nom|Number=Sing"  # sorted by the validator
    assert "Khakas" in REQUESTS[-1]["body"]["messages"][0]["content"]


def test_llm_translate_sentiment_spellcheck(opts):
    nlp = _pipeline("kaz", ["tokenize", "translate", "sentiment", "spellcheck"], opts,
                    translate_backend="llm", translate_tgt_lang="rus", sentiment_backend="llm")
    doc = nlp("Бұл китап нашар. Ертең келемін.")
    assert doc.sentences[0].translation.startswith("[Russian]")
    assert [s.sentiment for s in doc.sentences] == ["negative", "neutral"]
    assert doc.sentences[0].correction == "Бұл кітап нашар."
    assert doc.sentences[0].corrections == [("китап", "кітап")]
    assert doc.corrected_text == "Бұл кітап нашар. Ертең келемін."


def test_spellcheck_rejects_rewrites(opts):
    nlp = _pipeline("tur", ["tokenize", "spellcheck"], opts)
    doc = nlp("Bu bir REWRITE testi.")
    assert doc.sentences[0].correction == "Bu bir REWRITE testi."
    assert doc.sentences[0].corrections == []


def test_per_processor_override(opts):
    nlp = _pipeline("tur", ["tokenize", "sentiment"], opts, sentiment_backend="llm",
                    sentiment_llm_model="other-model")
    nlp("Güzel bir gün.")
    assert REQUESTS[-1]["body"]["model"] == "other-model"


def test_invalid_answers_leave_fields_empty(opts, monkeypatch):
    import turkicnlp.llm.client as client_mod

    monkeypatch.setattr(client_mod.LLMClient, "chat", lambda self, *a, **k: "not json at all")
    nlp = _pipeline("tur", ["tokenize", "ner"], opts, ner_backend="llm")
    doc = nlp("Ankara güzel.")
    assert doc.entities == []
    assert "[1 failed]" in doc._processor_log[-1]
    nlp = _pipeline("tur", ["tokenize", "ner"], opts, ner_backend="llm", ner_on_error="raise")
    with pytest.raises(LLMError):
        nlp("Ankara güzel.")


def test_shared_client_object(opts):
    client = LLMClient.from_options(opts)
    nlp = _pipeline("uzb", ["tokenize", "sentiment"], {"llm_client": client}, sentiment_backend="llm")
    nlp("Bugun yaxshi kun.")
    assert REQUESTS[-1]["body"]["model"] == "fake-model"


def test_ask_with_document_context(opts):
    import turkicnlp

    nlp = turkicnlp.Pipeline("kaz", processors=["tokenize"], tokenize_backend="rule")
    doc = nlp("Мен кеше мектепке бардым.")
    answer = ask("Explain the verb.", doc=doc, **opts)
    assert answer == "Free-form answer."
    system = REQUESTS[-1]["body"]["messages"][0]["content"]
    assert "Мен кеше мектепке бардым." in system and "Common Turkic Alphabet" in system
    assert REQUESTS[-1]["body"]["messages"][-1]["content"] == "Explain the verb."


def test_apertium_llm_disambiguation(opts):
    """Readings come from a fake transducer; the LLM picks option 1 for the ambiguous word."""
    from turkicnlp.models.document import Document, Sentence, Token, Word
    from turkicnlp.processors.llm_backend import ApertiumLLMMorphProcessor
    from turkicnlp.resources.tag_mappings import load_tag_map

    class FakeFST:
        def lookup(self, s):
            return {"yarın": [("yar<n><gen>", 0.0), ("yarın<adv>", 0.0)],
                    "geldim": [("gel<v><iv><ifi><p1><sg>", 0.0)]}.get(s, [])

    proc = ApertiumLLMMorphProcessor(lang="tur", config={k: v for k, v in opts.items()})
    proc._analyzer = FakeFST()
    proc._tag_mapper = load_tag_map("tur")
    proc._lexicon = {}
    proc._needs_translit = False
    from turkicnlp.llm.client import LLMClient as C
    from turkicnlp.llm.prompts import PromptTemplate as P
    proc.client = C.from_options(opts)
    proc.prompt = P.load("morph_disambiguate")
    proc.extra_examples, proc.n_examples, proc.max_options, proc.on_error = None, 3, 8, "warn"
    proc._llm_calls = proc._llm_failures = 0
    words = [Word(id=1, text="yarın"), Word(id=2, text="geldim")]
    doc = Document(text="yarın geldim", sentences=[Sentence(text="yarın geldim", tokens=[
        Token(id=(1,), text="yarın", words=[words[0]]), Token(id=(2,), text="geldim", words=[words[1]])],
        words=words)], script="Latn")
    proc.process(doc)
    assert (words[0].lemma, words[0].upos) == ("yarın", "ADV")
    assert words[1].feats and "Tense=Past" in words[1].feats
    assert "Ambiguous words" in REQUESTS[-1]["body"]["messages"][-1]["content"]


def test_anthropic_provider_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("TURKICNLP_LLM_API_KEY", raising=False)
    cfg = LLMConfig.from_options({"llm_provider": "anthropic", "llm_model": "claude-sonnet-4-6"})
    assert cfg.base_url == "https://api.anthropic.com/v1"
    assert cfg.api_key == "sk-ant-test"
    assert cfg.temperature is None and cfg.json_mode == "none"


def test_temperature_dropped_when_rejected(opts):
    client = LLMClient.from_options({**opts, "llm_model": "reject-temperature", "llm_cache": False})
    assert client.chat([{"role": "user", "content": "hi"}]) == "Free-form answer."
    assert "temperature" not in REQUESTS[-1]["body"]
    client.chat([{"role": "user", "content": "again"}])
    assert "temperature" not in REQUESTS[-1]["body"]


def test_no_temperature_sent_when_none(opts):
    client = LLMClient.from_options({**opts, "llm_temperature": None, "llm_cache": False})
    client.chat([{"role": "user", "content": "hi"}])
    assert "temperature" not in REQUESTS[-1]["body"]


def test_morph_alignment_by_token():
    from turkicnlp.processors.llm_backend import LLMMorphAnalyzer

    tokens = ["Қасым-Жомарт", "Тоқаев", "кеше", "келді", "."]
    # the model split the first token into two entries and forgot the full stop
    answer = {"words": [
        {"i": 0, "token": "Қасым", "lemma": "қасым", "upos": "PROPN", "feats": "_"},
        {"i": 1, "token": "-Жомарт", "lemma": "жомарт", "upos": "PROPN", "feats": "_"},
        {"i": 2, "token": "Тоқаев", "lemma": "тоқаев", "upos": "PROPN", "feats": "_"},
        {"i": 3, "token": "кеше", "lemma": "кеше", "upos": "ADV", "feats": "_"},
        {"i": 4, "token": "келді", "lemma": "кел", "upos": "VERB", "feats": "Tense=Past"}]}
    with pytest.raises(ValueError, match="Қасым-Жомарт"):
        LLMMorphAnalyzer._validate(answer, tokens)  # no shifted analyses: retry instead
    answer["words"][0:2] = [{"i": 0, "token": "Қасым-Жомарт", "lemma": "Қасым-Жомарт", "upos": "PROPN", "feats": "_"}]
    out = LLMMorphAnalyzer._validate(answer, tokens)
    assert [o[1] for o in out] == ["PROPN", "PROPN", "ADV", "VERB", "PUNCT"]
    assert out[3][0] == "кел"


def test_spellcheck_keeps_only_spelling_changes():
    from turkicnlp.processors.llm_backend import LLMSpellcheckProcessor

    proc = LLMSpellcheckProcessor(lang="kaz", config={})
    src = "Мен кеше китаптар окыдым инвесторлармен өте пайдалы"
    hyp = "Мен кеше кітаптар оқыдым инвесторлар меніңізді өткөп пайда"
    fixed, changes, dropped = proc._safe_changes(src, hyp)
    assert ("китаптар окыдым", "кітаптар оқыдым") in changes or ("китаптар", "кітаптар") in changes
    assert "меніңізді" not in fixed and "өткөп" not in fixed
    assert fixed.startswith("Мен кеше кітаптар оқыдым инвесторлармен")
    assert dropped >= 1


# --------------------------------------------------------------------------------------
# C23: only validated answers are cached
# --------------------------------------------------------------------------------------

def test_invalid_answers_are_not_cached(opts, monkeypatch):
    client = LLMClient.from_options(opts)
    answers = iter(["not json", "still not json", '{"label": "neutral"}'])
    calls = []

    def fake_request(self, messages, json_schema, temperature, max_tokens):
        calls.append(1)
        return next(answers)

    monkeypatch.setattr(LLMClient, "_request", fake_request)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    with pytest.raises(LLMError):
        client.chat_json(msgs, attempts=2)
    # nothing invalid was stored: the next call asks the model again
    assert client.chat_json(msgs, attempts=1) == {"label": "neutral"}
    assert len(calls) == 3
    assert client.chat_json(msgs, attempts=1) == {"label": "neutral"}  # now from the cache
    assert len(calls) == 3


def test_cached_invalid_answer_is_never_served(opts, monkeypatch):
    client = LLMClient.from_options(opts)
    msgs = [{"role": "user", "content": "u"}]
    key = client._cache_key(msgs, None, client.config.temperature, client.config.max_tokens)
    client._cache.put(key, '{"label": "INVALID"}')  # e.g. written by an older version
    monkeypatch.setattr(LLMClient, "_request", lambda self, *a: '{"label": "ok"}')

    def validate(d):
        if d["label"] != "ok":
            raise ValueError("bad label")
        return d["label"]

    assert client.chat_json(msgs, validate=validate) == "ok"
    assert client._cache.get(key) == '{"label": "ok"}'


# --------------------------------------------------------------------------------------
# C24: JSON mode fallback
# --------------------------------------------------------------------------------------

def test_json_mode_falls_back_to_plain_prompt(opts):
    _Handler.reject_schema = _Handler.reject_object = True
    client = LLMClient.from_options({**opts, "llm_cache": False})
    schema = {"type": "object", "properties": {"label": {"type": "string"}}}
    msgs = [{"role": "system", "content": "sentiment"}, {"role": "user", "content": "x"}]
    assert client.chat_json(msgs, json_schema=schema) == {"label": "neutral"}
    assert "response_format" not in REQUESTS[-1]["body"]
    assert client._json_mode == "none"


def test_context_length_error_does_not_downgrade(opts):
    client = LLMClient.from_options({**opts, "llm_model": "too-long", "llm_cache": False})
    with pytest.raises(LLMError, match="maximum context length"):
        client.chat([{"role": "user", "content": "x"}], json_schema={"type": "object"})
    assert client._json_mode == "schema"


def test_json_mode_fallback_is_thread_safe(opts):
    _Handler.reject_schema = _Handler.reject_object = True
    client = LLMClient.from_options({**opts, "llm_cache": False, "llm_max_workers": 8})
    schema = {"type": "object", "properties": {"label": {"type": "string"}}}
    reqs = [[{"role": "system", "content": "sentiment"}, {"role": "user", "content": f"x{i}"}]
            for i in range(16)]
    results = client.map_json(reqs, json_schema=schema)
    assert results == [{"label": "neutral"}] * 16


# --------------------------------------------------------------------------------------
# C25: one bad response never breaks a batch
# --------------------------------------------------------------------------------------

def test_malformed_responses_fail_only_their_item(opts):
    client = LLMClient.from_options({**opts, "llm_cache": False, "llm_max_retries": 0})
    reqs = [[{"role": "system", "content": "sentiment"}, {"role": "user", "content": u}]
            for u in ["ok", "SEND-HTML", "SEND-TRUNCATED", "fine"]]
    results = client.map_json(reqs)
    assert results[0] == {"label": "neutral"} and results[3] == {"label": "neutral"}
    assert isinstance(results[1], LLMError) and "did not return JSON" in str(results[1])
    assert isinstance(results[2], LLMError)


def test_unexpected_exception_becomes_llm_error(opts):
    client = LLMClient.from_options({**opts, "llm_cache": False})
    reqs = [[{"role": "system", "content": "sentiment"}, {"role": "user", "content": u}] for u in "ab"]

    def boom(d):
        raise RuntimeError("boom")

    results = client.map_json(reqs, validate=[boom, None])
    assert isinstance(results[0], LLMError) and "boom" in str(results[0])
    assert results[1] == {"label": "neutral"}


def test_locked_cache_does_not_fail_requests(opts, monkeypatch):
    import sqlite3

    from turkicnlp.llm.client import _DiskCache

    client = LLMClient.from_options(opts)

    def locked(self, *a):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(_DiskCache, "get", locked)
    monkeypatch.setattr(_DiskCache, "put", locked)
    reqs = [[{"role": "system", "content": "sentiment"}, {"role": "user", "content": u}] for u in "abc"]
    assert client.map_json(reqs) == [{"label": "neutral"}] * 3


def test_unwritable_cache_dir_disables_cache(opts, tmp_path, caplog):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    with caplog.at_level("WARNING", logger="turkicnlp.llm.client"):
        client = LLMClient.from_options({**opts, "llm_cache_dir": str(blocker / "sub")})
    assert client._cache is None
    assert "cache disabled" in caplog.text
    assert client.chat([{"role": "user", "content": "hi"}]) == "Free-form answer."


def test_disk_cache_uses_wal_and_threads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from turkicnlp.llm.client import _DiskCache

    cache = _DiskCache(tmp_path / "c.sqlite")
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: cache.put(str(i), str(i)), range(50)))
    assert all(cache.get(str(i)) == str(i) for i in range(50))
    mode = cache._conn().execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


# --------------------------------------------------------------------------------------
# C26: API keys
# --------------------------------------------------------------------------------------

def test_anthropic_never_uses_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TURKICNLP_LLM_API_KEY", raising=False)
    cfg = LLMConfig.from_options({"llm_provider": "anthropic", "llm_model": "claude"})
    assert cfg.api_key is None
    monkeypatch.setenv("TURKICNLP_LLM_API_KEY", "generic")
    cfg = LLMConfig.from_options({"llm_provider": "anthropic", "llm_model": "claude"})
    assert cfg.api_key == "generic"


def test_openai_key_only_for_openai_host(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("TURKICNLP_LLM_API_KEY", raising=False)
    monkeypatch.delenv("TURKICNLP_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert LLMConfig.from_options({"llm_model": "m"}).api_key == "sk-openai"  # api.openai.com
    third = LLMConfig.from_options({"llm_model": "m", "llm_base_url": "https://openrouter.ai/api/v1"})
    assert third.api_key is None
    evil = LLMConfig.from_options({"llm_model": "m", "llm_base_url": "https://api.openai.com.evil.io/v1"})
    assert evil.api_key is None
    explicit = LLMConfig.from_options({"llm_model": "m", "llm_base_url": "http://localhost:8000/v1",
                                       "llm_api_key": "mine"})
    assert explicit.api_key == "mine"
    monkeypatch.setenv("TURKICNLP_LLM_API_KEY", "generic")
    assert LLMConfig.from_options({"llm_model": "m", "llm_base_url": "http://x/v1"}).api_key == "generic"


def test_config_repr_masks_secrets():
    cfg = LLMConfig(model="m", api_key="sk-secret", headers={"api-key": "azure-secret"})
    for text in (repr(cfg), str(cfg), f"{cfg}"):
        assert "sk-secret" not in text and "azure-secret" not in text
        assert "api-key" in text and "model='m'" in text


# --------------------------------------------------------------------------------------
# C27: spellcheck guard
# --------------------------------------------------------------------------------------

def _speller():
    from turkicnlp.processors.llm_backend import LLMSpellcheckProcessor

    return LLMSpellcheckProcessor(lang="kaz", config={})


def test_spellcheck_ignores_attached_punctuation():
    fixed, changes, dropped = _speller()._safe_changes("Мен кітап окыдым", "Мен кітап оқыдым.")
    assert fixed == "Мен кітап оқыдым"  # spelling fixed, the added full stop is not
    assert changes == [("окыдым", "оқыдым")] and dropped == 0
    fixed, changes, _ = _speller()._safe_changes("Мен окыдым.", "Мен оқыдым")
    assert fixed == "Мен оқыдым." and changes == [("окыдым", "оқыдым")]


def test_spellcheck_does_not_accept_punctuation_changes():
    fixed, changes, _ = _speller()._safe_changes("Мен келдім бірақ ол кетті.", "Мен келдім, бірақ ол кетті!")
    assert fixed == "Мен келдім бірақ ол кетті." and changes == []


def test_spellcheck_keeps_whitespace_and_newlines():
    src = "Бұл  китап\nжақсы ,  өте!"
    fixed, changes, _ = _speller()._safe_changes(src, "Бұл кітап жақсы, өте!")
    assert fixed == "Бұл  кітап\nжақсы ,  өте!"
    assert changes == [("китап", "кітап")]


def test_spellcheck_pipeline_accepts_added_full_stop(opts, monkeypatch):
    import turkicnlp.llm.client as client_mod

    monkeypatch.setattr(client_mod.LLMClient, "_request",
                        lambda self, *a: json.dumps({"corrected": "Мен кітап оқыдым."}, ensure_ascii=False))
    nlp = _pipeline("kaz", ["tokenize", "spellcheck"], {**opts, "llm_cache": False})
    doc = nlp("Мен кітап окыдым")
    assert doc.sentences[0].correction == "Мен кітап оқыдым"
    assert doc.sentences[0].corrections == [("окыдым", "оқыдым")]


# --------------------------------------------------------------------------------------
# C28
# --------------------------------------------------------------------------------------

def test_fill_leaves_json_braces_alone(tmp_path):
    from turkicnlp.llm.prompts import _fill

    assert _fill('Return {"a": {"b": [1]}} for {language} {missing} {0} { x }', {"language": "Kazakh"}) == \
        'Return {"a": {"b": [1]}} for Kazakh {missing} {0} { x }'
    assert _fill("{text}", {"text": "{language}", "language": "X"}) == "{language}"  # single pass
    custom = tmp_path / "p.json"
    custom.write_text(json.dumps({"system": 'Answer {"label": "..."} in {language}.', "user": "{text}"}))
    msgs = PromptTemplate.load("sentiment", str(custom)).render({"language": "Kazakh", "text": "t"})
    assert msgs[0]["content"] == 'Answer {"label": "..."} in Kazakh.'
    # packaged prompts show single braces to the model
    msgs = PromptTemplate.load("translate").render({"language": "Kazakh", "text": "t", "target_language": "English"})
    assert '{"translation": "..."}' in msgs[0]["content"] and "{{" not in msgs[0]["content"]


def test_document_context_respects_max_words():
    from turkicnlp.llm.ask import document_context
    from turkicnlp.models.document import Document, Sentence, Word

    sents = []
    for k in range(20):
        words = [Word(id=i + 1, text=f"w{k}_{i}") for i in range(10)]
        sents.append(Sentence(text=" ".join(w.text for w in words), words=words))
    doc = Document(text=" ".join(s.text for s in sents), sentences=sents, lang="tur")
    ctx = document_context(doc, max_words=25)
    word_lines = [ln for ln in ctx.splitlines() if ln.startswith("  w") or re.match(r"^  \d+ w", ln)]
    assert len(word_lines) == 25
    assert len(ctx.split()) < 25 * 6  # the text lines are cut as well
    assert "Sentence 4" not in ctx


def test_target_language_name_with_script():
    from turkicnlp.llm.prompts import target_language_name

    assert target_language_name("uzb_Latn") == "Uzbek (Latin script)"
    assert target_language_name("uzb_Cyrl") == "Uzbek (Cyrillic script)"
    assert target_language_name("uzb-Cyrl") == "Uzbek (Cyrillic script)"
    assert target_language_name("uzb", "Cyrl") == "Uzbek (Cyrillic script)"
    assert target_language_name("uzb") == "Uzbek (Latin script)"  # primary script
    assert target_language_name("eng") == "English"
    assert target_language_name("English") == "English"


def test_translate_target_script_in_prompt(opts):
    nlp = _pipeline("kaz", ["tokenize", "translate"], opts, translate_backend="llm",
                    translate_tgt_lang="uzb_Cyrl")
    doc = nlp("Бұл кітап.")
    assert doc.sentences[0].translation.startswith("[Uzbek (Cyrillic script)]")


def test_ner_keeps_valid_entities_when_some_are_malformed():
    from turkicnlp.processors.llm_backend import LLMNERProcessor

    proc = LLMNERProcessor(lang="kaz", config={})
    proc.labels = ["PER", "LOC", "ORG"]
    data = {"entities": [{"start": 1, "end": 2, "label": "LOC"}, {"start": "x", "end": 2, "label": "PER"},
                         {"end": 3, "label": "ORG"}, "garbage", {"start": 3, "end": 4, "label": "per"},
                         {"start": 0, "end": 1, "label": "WHAT"}]}
    assert proc._validate(data, 5) == [(1, 2, "LOC"), (3, 4, "PER")]
    with pytest.raises(ValueError):
        proc._validate({"entities": [{"start": "x"}]}, 5)  # nothing usable: retry
    assert proc._validate({"entities": []}, 5) == []


def _apertium_proc(opts, **config):
    from turkicnlp.processors.llm_backend import ApertiumLLMMorphProcessor
    from turkicnlp.resources.tag_mappings import load_tag_map

    class FakeFST:
        def lookup(self, s):
            return {"yarın": [("yar<n><gen>", 0.0), ("yarın<adv>", 0.0)],
                    "geldim": [("gel<v><iv><ifi><p1><sg>", 0.0)]}.get(s, [])

    proc = ApertiumLLMMorphProcessor(lang="tur", config={**opts, **config})
    proc._analyzer = FakeFST()
    proc._tag_mapper = load_tag_map("tur")
    proc._lexicon = {}
    proc._needs_translit = False
    proc.client = LLMClient.from_options(opts)
    proc.prompt = PromptTemplate.load("morph_disambiguate")
    proc.extra_examples, proc.n_examples, proc.max_options, proc.on_error = None, 3, 8, "warn"
    proc._llm_calls = proc._llm_failures = 0
    return proc


def _apertium_doc(n_sents=1):
    from turkicnlp.models.document import Document, Sentence, Token, Word

    sents = []
    for _ in range(n_sents):
        words = [Word(id=1, text="yarın"), Word(id=2, text="geldim")]
        sents.append(Sentence(text="yarın geldim", tokens=[
            Token(id=(1,), text="yarın", words=[words[0]]), Token(id=(2,), text="geldim", words=[words[1]])],
            words=words))
    return Document(text=" ".join(s.text for s in sents), sentences=sents, script="Latn")


def test_apertium_llm_batches_sentences(opts, monkeypatch):
    proc = _apertium_proc(opts)
    batches = []
    orig = LLMClient.map_json

    def spy(self, requests, **kw):
        batches.append(len(requests))
        return orig(self, requests, **kw)

    monkeypatch.setattr(LLMClient, "map_json", spy)
    doc = proc.process(_apertium_doc(3))
    assert batches == [3]
    assert all(s.words[0].upos == "ADV" for s in doc.sentences)


def test_apertium_llm_keeps_configured_tagger(opts, monkeypatch):
    from turkicnlp.processors.morphology import ApertiumMorphProcessor

    proc = _apertium_proc(opts, apertium_tagger_cmd="apertium-tagger", apertium_tagger_args=["-g", "x.prob"])
    tagger_calls = []

    def fake_tagger(self, words, readings):
        tagger_calls.append(1)
        return [r[0] if r else None for r in readings]  # tagger picks "yar<n><gen>"

    monkeypatch.setattr(ApertiumMorphProcessor, "_tagger_disambiguate", fake_tagger)
    REQUESTS.clear()
    doc = proc.process(_apertium_doc())
    assert doc.words[0].lemma == "yar"  # the tagger's choice is kept
    assert REQUESTS == []  # nothing left for the LLM
    assert tagger_calls == [1]  # and the tagger ran once per sentence


def test_apertium_llm_shows_lemmas_in_user_script(opts):
    proc = _apertium_proc(opts)

    class Upper:
        def transliterate(self, s):
            return s.upper()

    proc._needs_translit = True
    proc._from_fst_translit = Upper()
    proc._to_fst_translit = type("Id", (), {"transliterate": lambda self, s: s})()
    REQUESTS.clear()
    doc = proc.process(_apertium_doc())
    user = REQUESTS[-1]["body"]["messages"][-1]["content"]
    assert "YARIN ADV" in user and "YAR " in user
    assert doc.words[0].lemma == "YARIN"


def test_retry_after_is_capped():
    import urllib.error
    from email.message import Message

    from turkicnlp.llm.client import _MAX_RETRY_AFTER

    def err(value):
        h = Message()
        if value is not None:
            h["Retry-After"] = value
        return urllib.error.HTTPError("http://x", 429, "slow down", h, None)

    assert LLMClient._retry_delay(err("3600"), 1.0) == _MAX_RETRY_AFTER == 60
    assert LLMClient._retry_delay(err("2.5"), 1.0) == 2.5
    assert LLMClient._retry_delay(err("Wed, 21 Oct 2015 07:28:00 GMT"), 4.0) == 4.0
    assert LLMClient._retry_delay(err(None), 1.0) == 1.0


def test_hf_provider_pad_id_zero_and_no_double_bos(tmp_path):
    torch = pytest.importorskip("torch")

    class Tok:
        pad_token_id = 0
        eos_token_id = 2
        seen = {}

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "<s>[INST] hi [/INST]"

        def __call__(self, text, return_tensors, add_special_tokens=True):
            Tok.seen["add_special_tokens"] = add_special_tokens

            class Enc(dict):
                def to(self, device):
                    return self
            return Enc(input_ids=torch.tensor([[1, 5, 6]]))

        def decode(self, ids, skip_special_tokens):
            return "answer"

    class Model:
        kwargs = {}

        def generate(self, **kw):
            Model.kwargs = kw
            return torch.tensor([[1, 5, 6, 7]])

    client = LLMClient(model="local", provider="hf", cache_dir=str(tmp_path))
    client._hf = (Tok(), Model(), "cpu", torch)
    assert client.chat([{"role": "user", "content": "hi"}]) == "answer"
    assert Model.kwargs["pad_token_id"] == 0
    assert Tok.seen["add_special_tokens"] is False
