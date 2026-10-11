"""LLM client used by all LLM-based TurkicNLP components.

Providers:

``openai``     (default) any server that implements the OpenAI *chat completions* API:
               OpenAI, Azure OpenAI (via ``base_url``), vLLM, SGLang, Ollama, llama.cpp
               server, LM Studio, Together, OpenRouter, Groq, … Uses only the standard
               library (no ``openai`` package needed).
``anthropic``  Claude through Anthropic's OpenAI-compatible endpoint
               (``https://api.anthropic.com/v1``; key from ``ANTHROPIC_API_KEY``).
               ``temperature`` is not sent (recent Claude models reject values below 1).
``hf``         a local Hugging Face chat model run with ``transformers``
               (``pip install "turkicnlp[transformers]"``).

Configuration (all optional except ``model``); the same keys work as ``llm_*`` options
of :class:`turkicnlp.Pipeline`:

=================  ==================================================================
``model``          model name on the server / Hugging Face model id
``base_url``       API root, e.g. ``https://api.openai.com/v1`` or
                   ``http://localhost:8000/v1`` (vLLM) or ``http://localhost:11434/v1``
                   (Ollama). Env: ``TURKICNLP_LLM_BASE_URL``, ``OPENAI_BASE_URL``.
``api_key``        Bearer token. Env: ``TURKICNLP_LLM_API_KEY``; ``OPENAI_API_KEY`` only when
                   ``base_url`` is ``api.openai.com``; ``ANTHROPIC_API_KEY`` for ``anthropic``.
``provider``       ``openai``, ``anthropic`` or ``hf``
``headers``        extra HTTP headers (e.g. ``{"api-key": ...}`` for Azure)
``extra_body``     extra JSON fields for every request (e.g. ``{"top_k": 1}``)
``temperature``    default 0.0 (deterministic, cache-friendly); ``None`` = not sent
``max_tokens``     default 2048
``timeout``        seconds per request, default 120
``max_retries``    retries on network errors / 429 / 5xx, default 3
``max_workers``    parallel requests for batches, default 4
``json_mode``      ``auto`` (JSON schema → JSON object → plain prompt), ``schema``,
                   ``object`` or ``none``
``cache``          cache answers on disk (default True); ``cache_dir`` to relocate it.
                   Only answers that passed validation are stored.
=================  ==================================================================
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

Messages = list[dict[str, str]]


class LLMError(RuntimeError):
    """Raised when the LLM cannot be reached or keeps returning invalid output."""


_MAX_RETRY_AFTER = 60.0


def _host(url: Optional[str]) -> str:
    try:
        return (urlparse(url or "").hostname or "").lower()
    except ValueError:
        return ""


@dataclass(repr=False)
class LLMConfig:
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    provider: str = "openai"
    headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    temperature: Optional[float] = 0.0
    max_tokens: int = 2048
    timeout: float = 120.0
    max_retries: int = 3
    max_workers: int = 4
    json_mode: str = "auto"
    cache: bool = True
    cache_dir: Optional[str] = None
    device: Optional[str] = None  # hf provider only

    @classmethod
    def from_options(cls, options: Optional[dict] = None, prefix: str = "llm_") -> "LLMConfig":
        """Build a config from ``{"llm_model": ..., "llm_base_url": ...}`` or plain keys."""
        options = options or {}
        known = set(cls.__dataclass_fields__)
        kwargs = {}
        for key, value in options.items():
            name = key[len(prefix):] if key.startswith(prefix) else key
            if name in known and (value is not None or name == "temperature"):
                kwargs[name] = value
        cfg = cls(**kwargs)
        cfg.model = cfg.model or os.environ.get("TURKICNLP_LLM_MODEL")
        cfg.base_url = cfg.base_url or os.environ.get("TURKICNLP_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if cfg.provider == "anthropic":
            if not cfg.base_url or "openai.com" in cfg.base_url:
                cfg.base_url = "https://api.anthropic.com/v1"
            if "temperature" not in {k[len(prefix):] if k.startswith(prefix) else k for k in options}:
                cfg.temperature = None
            cfg.json_mode = "none" if cfg.json_mode == "auto" else cfg.json_mode  # response_format is ignored
        if cfg.provider == "openai" and not cfg.base_url:
            cfg.base_url = "https://api.openai.com/v1"
        # Keys: an explicit key or TURKICNLP_LLM_API_KEY is always used. Provider-specific
        # keys only go to their own provider: never send OPENAI_API_KEY to Anthropic or to
        # a third-party server.
        cfg.api_key = cfg.api_key or os.environ.get("TURKICNLP_LLM_API_KEY")
        if not cfg.api_key:
            if cfg.provider == "anthropic":
                cfg.api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif cfg.provider == "openai" and _host(cfg.base_url) == "api.openai.com":
                cfg.api_key = os.environ.get("OPENAI_API_KEY")
        return cfg

    def public_dict(self) -> dict:
        d = asdict(self)
        d.pop("api_key", None)
        d["headers"] = sorted(self.headers)  # names only, values may be secrets
        return d

    def __repr__(self) -> str:
        parts = []
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "api_key" and value:
                value = "***"
            elif name == "headers" and value:
                value = {k: "***" for k in value}
            parts.append(f"{name}={value!r}")
        return f"LLMConfig({', '.join(parts)})"

    __str__ = __repr__


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------

class _DiskCache:
    """SQLite answer cache, safe for threads and for several processes.

    Every thread gets its own connection; ``timeout`` and WAL journaling avoid
    "database is locked" errors when several workers / processes write at once."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._local = threading.local()
        self._lock = threading.Lock()  # serialises writes within this process
        conn = self._conn()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:  # e.g. network file systems without WAL support
            pass
        conn.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, value TEXT)")
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, timeout=30.0, check_same_thread=False)
            self._local.conn = conn
        return conn

    def get(self, key: str) -> Optional[str]:
        row = self._conn().execute("SELECT value FROM answers WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute("INSERT OR REPLACE INTO answers VALUES (?, ?)", (key, value))
            conn.commit()

    def delete(self, key: str) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute("DELETE FROM answers WHERE key = ?", (key,))
            conn.commit()


def _default_cache_path() -> Path:
    try:
        from turkicnlp.resources.registry import ModelRegistry

        return ModelRegistry.default_dir().parent / "llm_cache.sqlite"
    except Exception:  # noqa: BLE001
        return Path.home() / ".turkicnlp" / "llm_cache.sqlite"


# --------------------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in a model answer (tolerates code fences, prose,
    and ``<think>`` blocks of reasoning models)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    stripped = _FENCE.sub("", text).strip()
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except (json.JSONDecodeError, ValueError):
                            break
            start = text.find(opener, start + 1)
    raise ValueError("no JSON found in the model answer")


# --------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------

class LLMClient:
    """Chat-completion client with JSON output, retries, batching and a disk cache.

    Example::

        client = LLMClient(model="gpt-4o-mini", api_key="sk-...")
        client = LLMClient(model="Qwen/Qwen3-8B", base_url="http://gpu-server:8000/v1")
        client = LLMClient(model="Qwen/Qwen2.5-1.5B-Instruct", provider="hf")
        client.chat([{"role": "user", "content": "Сәлем!"}])
    """

    def __init__(self, config: Optional[LLMConfig] = None, **options: Any):
        self.config = config or LLMConfig.from_options(options, prefix="")
        if config is not None and options:
            for k, v in options.items():
                setattr(self.config, k, v)
        if not self.config.model:
            raise ValueError(
                "No LLM model configured. Pass llm_model=... (Pipeline) or model=... (LLMClient), "
                "or set TURKICNLP_LLM_MODEL."
            )
        if self.config.provider not in ("openai", "anthropic", "hf"):
            raise ValueError(f"Unknown LLM provider '{self.config.provider}' (use 'openai', 'anthropic' or 'hf')")
        if self.config.json_mode not in ("auto", "schema", "object", "none"):
            raise ValueError(f"Unknown json_mode '{self.config.json_mode}' (use 'auto', 'schema', 'object' or 'none')")
        self._send_temperature = self.config.temperature is not None
        # ``auto`` starts with a JSON schema and steps down (object → plain prompt) when the
        # server says it does not support the current response_format.
        self._auto_json = self.config.json_mode == "auto"
        self._json_mode = "schema" if self._auto_json else self.config.json_mode
        self._mode_lock = threading.Lock()
        self._cache: Optional[_DiskCache] = None
        if self.config.cache:
            path = (Path(self.config.cache_dir) / "llm_cache.sqlite" if self.config.cache_dir
                    else _default_cache_path())
            try:
                self._cache = _DiskCache(path)
            except (OSError, sqlite3.Error) as exc:
                logger.warning("LLM answer cache disabled: cannot use %s (%s)", path, exc)
        self._hf = None
        self._hf_lock = threading.Lock()

    @classmethod
    def from_options(cls, options: Optional[dict] = None) -> "LLMClient":
        """From pipeline-style options. ``llm_client`` (an existing client) is reused."""
        options = options or {}
        if isinstance(options.get("llm_client"), LLMClient):
            return options["llm_client"]
        return cls(LLMConfig.from_options(options))

    @property
    def model(self) -> str:
        return str(self.config.model)

    def __repr__(self) -> str:
        where = "local" if self.config.provider == "hf" else self.config.base_url
        return f"<LLMClient {self.config.provider}:{self.config.model} @ {where}>"

    # -- cache -------------------------------------------------------------------------

    def _cache_key(self, messages: Messages, json_schema: Optional[dict], temperature: Optional[float],
                   max_tokens: int) -> str:
        return hashlib.sha256(json.dumps(
            [self.config.provider, self.config.model, self.config.base_url, messages, json_schema,
             temperature, max_tokens, self.config.extra_body], ensure_ascii=False, sort_keys=True
        ).encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        if self._cache is None:
            return None
        try:
            return self._cache.get(key)
        except sqlite3.Error as exc:
            logger.warning("LLM cache read failed (%s); asking the model", exc)
            return None

    def _cache_put(self, key: str, value: str) -> None:
        if self._cache is None:
            return
        try:
            self._cache.put(key, value)
        except sqlite3.Error as exc:
            logger.warning("LLM cache write failed (%s); answer not cached", exc)

    def _cache_delete(self, key: str) -> None:
        if self._cache is None:
            return
        try:
            self._cache.delete(key)
        except sqlite3.Error:
            pass

    # -- public API --------------------------------------------------------------------

    def chat(self, messages: Messages, json_schema: Optional[dict] = None,
             temperature: Optional[float] = None, max_tokens: Optional[int] = None,
             use_cache: bool = True) -> str:
        """Return the assistant's text for ``messages`` (cached unless ``use_cache=False``)."""
        temperature = self.config.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.config.max_tokens
        key = None
        if use_cache and self._cache is not None:
            key = self._cache_key(messages, json_schema, temperature, max_tokens)
            hit = self._cache_get(key)
            if hit is not None:
                return hit
        text = self._request(messages, json_schema, temperature, max_tokens)
        if key is not None:
            self._cache_put(key, text)
        return text

    def _request(self, messages: Messages, json_schema: Optional[dict], temperature: Optional[float],
                 max_tokens: int) -> str:
        if self.config.provider == "hf":
            return self._chat_hf(messages, temperature, max_tokens)
        return self._chat_openai(messages, json_schema, temperature, max_tokens)

    def chat_json(self, messages: Messages, json_schema: Optional[dict] = None,
                  validate: Optional[Callable[[Any], Any]] = None, attempts: int = 2) -> Any:
        """Ask for JSON; parse and validate it. On failure the error is shown to the model
        and the request repeated (``attempts`` times in total). Raises :class:`LLMError`.

        Only answers that pass parsing and validation are cached; a cached answer that
        does not validate (e.g. from an older version) is ignored and replaced."""

        def check(text: str) -> Any:
            data = extract_json(text)
            return validate(data) if validate else data

        msgs = list(messages)
        last_err: Optional[Exception] = None
        for attempt in range(attempts):
            # greedy first; a little randomness on API retries (local models stay greedy:
            # sampling in half precision on MPS can produce NaN probabilities)
            retry_temp = (None if attempt == 0 or self.config.temperature is None
                          or self.config.provider == "hf" else 0.3)
            temperature = self.config.temperature if retry_temp is None else retry_temp
            key = (self._cache_key(msgs, json_schema, temperature, self.config.max_tokens)
                   if self._cache is not None else None)
            if key is not None:
                hit = self._cache_get(key)
                if hit is not None:
                    try:
                        return check(hit)
                    except (ValueError, KeyError, TypeError, IndexError):
                        logger.debug("ignoring a cached LLM answer that does not validate")
                        self._cache_delete(key)
            text = self.chat(msgs, json_schema=json_schema, temperature=retry_temp, use_cache=False)
            try:
                result = check(text)
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                last_err = exc
                # The feedback goes into the system message: appending it as a user turn
                # makes some models treat it as the text to translate / correct.
                note = (f"\n\nIMPORTANT: a previous reply to this request could not be used ({exc}). "
                        f"Reply with JSON only, in exactly the requested format.")
                msgs = [dict(m) for m in messages]
                if msgs and msgs[0]["role"] == "system":
                    msgs[0]["content"] += note
                else:
                    msgs.insert(0, {"role": "system", "content": note.strip()})
                continue
            if key is not None:
                self._cache_put(key, text)
            return result
        raise LLMError(f"invalid LLM output after {attempts} attempts: {last_err}")

    def map_json(self, requests: list[Messages], json_schema: Optional[dict] = None,
                 validate: Any = None) -> list[Any]:
        """:meth:`chat_json` over many prompts (in parallel for the openai provider).

        ``validate`` is one callable for all requests or a list with one per request.
        Failed items are returned as :class:`LLMError` instances instead of raising: one
        bad answer (HTML error page, truncated JSON, a locked cache, ...) never breaks
        the rest of the batch."""
        validators = validate if isinstance(validate, list) else [validate] * len(requests)

        def one(job):
            msgs, val = job
            try:
                return self.chat_json(msgs, json_schema=json_schema, validate=val)
            except LLMError as exc:
                return exc
            except Exception as exc:  # noqa: BLE001 — isolate the failure to this item
                err = LLMError(f"LLM request failed: {type(exc).__name__}: {exc}")
                err.__cause__ = exc
                return err

        jobs = list(zip(requests, validators))
        if self.config.provider != "hf" and self.config.max_workers > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                return list(pool.map(one, jobs))
        return [one(j) for j in jobs]

    # -- OpenAI-compatible HTTP --------------------------------------------------------

    @staticmethod
    def _retry_delay(exc: urllib.error.HTTPError, default: float) -> float:
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        try:
            value = float(retry_after) if retry_after else default
        except ValueError:  # an HTTP date: use our own back-off
            value = default
        return max(0.0, min(value, _MAX_RETRY_AFTER))

    def _post(self, path: str, body: dict) -> dict:
        url = self.config.base_url.rstrip("/") + path
        headers = {"Content-Type": "application/json", **self.config.headers}
        if self.config.api_key:
            headers.setdefault("Authorization", f"Bearer {self.config.api_key}")
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        delay = 1.0
        for attempt in range(self.config.max_retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    raw = resp.read()
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", "replace")[:1000]
                except Exception:  # noqa: BLE001
                    detail = ""
                if exc.code in (408, 409, 429, 500, 502, 503, 504) and attempt < self.config.max_retries:
                    time.sleep(self._retry_delay(exc, delay))
                    delay = min(delay * 2, _MAX_RETRY_AFTER)
                    continue
                raise _HTTPError(exc.code, detail) from None
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError,
                    OSError) as exc:
                # HTTPException covers truncated bodies (IncompleteRead) and bad status lines
                if attempt < self.config.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, _MAX_RETRY_AFTER)
                    continue
                raise LLMError(f"cannot reach LLM server at {url}: {exc}") from None
            try:
                return json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                snippet = raw[:200].decode("utf-8", "replace")
                raise LLMError(f"LLM server at {url} did not return JSON: {snippet!r}") from None
        raise LLMError("unreachable")  # pragma: no cover

    @staticmethod
    def _format_unsupported(exc: "_HTTPError") -> bool:
        """Does this error say that the response_format (JSON mode) is not supported?
        Context-length and other request errors do not count."""
        if exc.code not in (400, 422):
            return False
        detail = exc.detail.lower()
        if any(s in detail for s in ("context length", "context_length", "maximum context", "too many tokens",
                                      "max_tokens", "maximum length", "too long")):
            return False
        return any(s in detail for s in ("response_format", "response format", "json_schema", "json schema",
                                          "json_object", "json mode", "structured output", "guided_json"))

    def _chat_openai(self, messages: Messages, json_schema: Optional[dict], temperature: Optional[float],
                     max_tokens: int) -> str:
        for _ in range(6):  # each round removes one unsupported feature
            body: dict[str, Any] = {"model": self.config.model, "messages": messages, "max_tokens": max_tokens}
            if temperature is not None and self._send_temperature:
                body["temperature"] = temperature
            body.update(self.config.extra_body)
            mode = self._json_mode
            if json_schema is not None and mode == "schema":
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": "answer", "schema": json_schema}}
            elif json_schema is not None and mode == "object":
                body["response_format"] = {"type": "json_object"}
            try:
                resp = self._post("/chat/completions", body)
            except _HTTPError as exc:
                if exc.code == 400 and "temperature" in body and "temperature" in exc.detail.lower():
                    # e.g. newer Claude models accept no temperature below 1: stop sending it
                    if self._send_temperature:
                        logger.info("LLM server rejected temperature; not sending it any more")
                    self._send_temperature = False
                    continue
                if "response_format" in body and self._auto_json and self._format_unsupported(exc):
                    # Several threads may hit this at once: only the first one steps down,
                    # the others just retry with the mode that is now current.
                    with self._mode_lock:
                        if self._json_mode == mode:
                            self._json_mode = "object" if mode == "schema" else "none"
                            logger.info("LLM server rejected response_format (%s); falling back to '%s'",
                                        exc.detail[:120], self._json_mode)
                    continue
                raise LLMError(f"LLM request failed (HTTP {exc.code}): {exc.detail}") from None
            try:
                choice = resp["choices"][0]
                content = choice["message"].get("content")
            except (KeyError, IndexError, TypeError, AttributeError):
                raise LLMError(f"unexpected response from LLM server: {str(resp)[:300]}") from None
            if isinstance(content, list):  # some servers return content parts
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return content if isinstance(content, str) else ""
        raise LLMError("LLM request failed: the server rejected every request format")

    # -- local Hugging Face model ------------------------------------------------------

    def _chat_hf(self, messages: Messages, temperature: Optional[float], max_tokens: int) -> str:
        with self._hf_lock:
            if self._hf is None:
                try:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                except ImportError as exc:
                    raise ImportError('The "hf" LLM provider needs transformers + torch: '
                                      'pip install "turkicnlp[transformers]"') from exc
                device = self.config.device or ("cuda" if torch.cuda.is_available() else
                                                "mps" if torch.backends.mps.is_available() else "cpu")
                tok = AutoTokenizer.from_pretrained(self.config.model)
                model = AutoModelForCausalLM.from_pretrained(self.config.model, dtype="auto").to(device).eval()
                self._hf = (tok, model, device, torch)
            tok, model, device, torch = self._hf
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # The chat template already contains BOS etc.: do not add them a second time.
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            gen = {"max_new_tokens": max_tokens, "do_sample": bool(temperature)}
            if temperature:
                gen["temperature"] = temperature
            else:  # greedy: unset the sampling defaults of the model's generation config
                gen.update(temperature=None, top_p=None, top_k=None)
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id  # 0 is valid
            try:
                with torch.no_grad():
                    out = model.generate(**enc, **gen, pad_token_id=pad_id)
            except RuntimeError as exc:  # e.g. NaN probabilities, out of memory
                raise LLMError(f"local generation failed: {exc}") from None
            return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


class _HTTPError(Exception):
    def __init__(self, code: int, detail: str):
        super().__init__(f"HTTP {code}: {detail}")
        self.code = code
        self.detail = detail
