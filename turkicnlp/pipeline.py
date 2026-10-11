"""
Pipeline orchestrator for TurkicNLP.

The :class:`Pipeline` resolves processor dependencies, loads script-appropriate
models, optionally inserts transliteration steps, and chains processors
in canonical order.
"""

from __future__ import annotations

from typing import Optional

from turkicnlp.models.document import Document
from turkicnlp.processors.base import Processor
from turkicnlp.scripts import Script, get_script_config
from turkicnlp.scripts.detector import detect_script
from turkicnlp.scripts.transliterator import Transliterator

# Canonical processor execution order
PROCESSOR_ORDER: list[str] = [
    "asr",
    "script_detect",
    "transliterate",
    "tokenize",
    "spellcheck",
    "mwt",
    "morph",
    "morph_neural",
    "pos",
    "feats",
    "lemma",
    "depparse",
    "ner",
    "embeddings",
    "sentiment",
    "transliterate_back",
]


# Processors that ``Pipeline(lang)`` (``processors=None``) does not load by
# default: they download large models or need optional extras. Request them
# explicitly, e.g. ``Pipeline("kaz", processors=["tokenize", "translate"])``.
OPTIONAL_PROCESSORS: frozenset[str] = frozenset(
    {"asr", "translate", "embeddings", "sentiment", "sp_tokenize", "morpheme_tokenize", "langid",
     "spellcheck"}
)


# Word fields and the annotation layer (``Processor.PROVIDES``) that writes them.
FIELD_LAYERS: dict[str, str] = {
    "lemma": "lemma",
    "upos": "pos",
    "xpos": "pos",
    "feats": "feats",
    "head": "depparse",
    "deprel": "depparse",
}
_FIELDS = tuple(FIELD_LAYERS)


def _catalog_backend(proc_info: object) -> Optional[str]:
    """The default backend a catalog entry names (``None`` if it names none)."""
    if not isinstance(proc_info, dict):
        return None
    backend = proc_info.get("default")
    if backend is None and proc_info.get("backends"):
        backend = next(iter(proc_info["backends"].keys()), None)
    return backend


def default_processors(proc_catalog: dict) -> list[str]:
    """Processors loaded when the user does not name any.

    Takes the processors the catalog lists for the language/script, drops the
    optional heavy ones (:data:`OPTIONAL_PROCESSORS`) and, when both an Apertium
    ``morph`` and the neural ``morph_neural`` analyser exist, keeps only
    ``morph`` (also dropping ``feats``/``lemma`` when they come from the
    same Glot500 morph model). A processor is kept only when the backend the catalog names for
    it is registered and implemented (e.g. Glot500 backends need torch).
    """
    from turkicnlp.resources.registry import ProcessorRegistry

    names = [p for p in proc_catalog if p not in OPTIONAL_PROCESSORS]
    if "morph" in names:
        # The Apertium analyser and the Glot500 morph model provide the same
        # fields; running both would mix two analyses per word.
        if "morph_neural" in names:
            names.remove("morph_neural")
        for p in ("feats", "lemma"):
            if p in names and _catalog_backend(proc_catalog.get(p)) == "multilingual_glot500_morph":
                names.remove(p)
    kept = []
    for p in names:
        if p == "tokenize":
            kept.append(p)
            continue
        implemented = ProcessorRegistry.implemented_backends(p)
        backend = _catalog_backend(proc_catalog.get(p))
        if backend is not None:
            if backend in implemented:
                kept.append(p)
        elif implemented:
            kept.append(p)
    if "tokenize" not in kept:
        kept.insert(0, "tokenize")
    return kept


class Pipeline:
    """Main entry point. Constructs a chain of Processors for a language
    and runs documents through them.

    Usage::

        nlp = Pipeline("kaz", processors=["tokenize", "pos", "lemma", "depparse"])
        doc = nlp("Мен мектепке бардым")

    Field ownership: every word field has exactly one owner per pipeline, the
    last processor in execution order that provides it (``pos`` owns
    UPOS/XPOS, ``feats`` FEATS, ``lemma`` LEMMA, ``depparse`` HEAD/DEPREL; a
    full analyser such as ``morph`` owns the fields no later step provides).
    Values written by other processors are kept in ``word._provenance`` but not
    in the word fields, and FEATS from a different processor than UPOS are
    dropped for a word when that processor's own UPOS disagrees.

    Args:
        lang: ISO 639-3 language code.
        processors: Processor names to use. ``None`` loads the default set for
            the language (catalog processors without the optional heavy ones in
            :data:`OPTIONAL_PROCESSORS`, see :func:`default_processors`).
        script: Script code (``Cyrl``, ``Latn``, ``Arab``) or ``auto``. When
            the language has no models for this script, the input is
            transliterated to the script that has them and back.
        transliterate_to: Target script for transliteration bridging.
        model_dir: Override default model directory.
        use_gpu: Whether to use GPU for neural processors.
        **processor_configs: Per-processor config overrides
            (e.g. ``morph_backend="apertium"``).
    """

    def __init__(
        self,
        lang: str,
        processors: Optional[list[str]] = None,
        script: Optional[str] = None,
        transliterate_to: Optional[str] = None,
        model_dir: Optional[str] = None,
        use_gpu: bool = False,
        **processor_configs: object,
    ) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self.model_dir = model_dir
        self._processors: list[Processor] = []
        self._owners: dict[str, Processor] = {}
        self._script_config = get_script_config(lang)
        self._requested_processors = processors
        self._processor_configs = processor_configs
        self._autoload = processors is not None
        self._asr = None  # SpeechRecognizer, created on first from_audio()

        if script is None or script == "auto":
            self._script: Optional[Script] = None
            self._explicit_script = False
        else:
            self._script = Script(script)
            self._explicit_script = True
            if self._script not in self._script_config.available:
                raise ValueError(
                    f"Script '{script}' not available for {lang}. "
                    f"Available: {[str(s) for s in self._script_config.available]}"
                )

        self._transliterator: Optional[Transliterator] = None
        self._transliterate_back_enabled = False
        source_script = self._script or self._script_config.primary

        if transliterate_to:
            target_script = Script(transliterate_to)
            if target_script != source_script:
                self._set_bridge(source_script, target_script)
            else:
                self._model_script = source_script
        else:
            self._model_script = source_script
            model_script = self._catalog_model_script(source_script)
            if model_script is not None and model_script != source_script:
                # No models for this script: bridge through the script that has them.
                try:
                    self._set_bridge(source_script, model_script)
                except ValueError:
                    pass

        if self._autoload:
            self._build_processors()

    # ------------------------------------------------------------------
    # Script handling

    def _set_bridge(self, source: Script, target: Script) -> None:
        self._transliterator = Transliterator(self.lang, source, target)
        try:
            Transliterator(self.lang, target, source)
            self._transliterate_back_enabled = True
        except ValueError:
            self._transliterate_back_enabled = False
        self._model_script = target

    def _lang_catalog(self) -> dict:
        from turkicnlp.resources.registry import ModelRegistry

        info = ModelRegistry.load_catalog().get(self.lang, {})
        return info if isinstance(info, dict) else {}

    def _catalog_model_script(self, script: Script) -> Optional[Script]:
        """The script whose catalog models serve input in ``script``."""
        info = self._lang_catalog()
        section = info.get("processors", {})
        if not section or str(script) in section:
            return script
        primary = info.get("scripts", {}).get("primary")
        if primary in section:
            return Script(primary)
        return Script(next(iter(section)))

    def _proc_catalog(self) -> dict:
        info = self._lang_catalog()
        section = info.get("processors", {})
        script_key = str(self._model_script)
        if script_key in section:
            return section[script_key]
        if section:
            primary = info.get("scripts", {}).get("primary")
            return section.get(primary, next(iter(section.values())))
        return {}

    # ------------------------------------------------------------------
    # Processor resolution

    def _choose_backend(self, proc_name: str, proc_catalog: dict) -> Optional[str]:
        """Backend for ``proc_name``: user option, catalog default, then fallbacks.

        Placeholder (unimplemented) backends are never chosen implicitly.
        """
        from turkicnlp.resources.registry import ProcessorRegistry

        backend_key = f"{proc_name}_backend"
        if self._processor_configs.get(backend_key) is not None:
            return str(self._processor_configs[backend_key])
        backend = _catalog_backend(proc_catalog.get(proc_name))
        if backend is not None:
            return backend
        if proc_name == "tokenize":
            return "rule_arabic" if self._model_script == Script.PERSO_ARABIC else "rule"
        implemented = ProcessorRegistry.implemented_backends(proc_name)
        if "llm" in implemented:
            return "llm"
        return implemented[0] if implemented else None

    def _plan(self) -> list[tuple[str, str]]:
        """Resolve processors and backends without loading anything."""
        from turkicnlp.resources.registry import ProcessorRegistry

        proc_catalog = self._proc_catalog()
        requested = self._requested_processors
        explicit = requested is not None
        if requested is None:
            requested = default_processors(proc_catalog)
        names = self._resolve_dependencies(list(requested), proc_catalog)

        plan: list[tuple[str, str]] = []
        for proc_name in names:
            if proc_name in ("script_detect", "transliterate", "transliterate_back", "asr"):
                # "asr" runs before the text pipeline, see from_audio()
                continue
            backend = self._choose_backend(proc_name, proc_catalog)
            if backend is None:
                if explicit and proc_name in requested:
                    raise ValueError(
                        f"No backend available for processor '{proc_name}' "
                        f"({self.lang}, {self._model_script}). Pass "
                        f"{proc_name}_backend=... to choose one."
                    )
                raise ValueError(
                    f"Processor '{proc_name}' is needed by "
                    f"{[p for p in requested]} but has no backend for "
                    f"{self.lang} ({self._model_script})."
                )
            cls = ProcessorRegistry.get(proc_name, backend)
            if not getattr(cls, "IMPLEMENTED", True) and backend_key_missing(
                self._processor_configs, proc_name
            ):
                raise ValueError(
                    f"Backend '{backend}' of '{proc_name}' is not implemented yet."
                )
            plan.append((proc_name, backend))
        return plan

    def _resolve_dependencies(
        self, requested: list[str], proc_catalog: Optional[dict] = None
    ) -> list[str]:
        """Ensure all processor dependencies are satisfied.

        Requirements are read from the backend that will actually run, e.g.
        requesting ``["depparse"]`` adds ``tokenize`` and ``pos``.
        """
        from turkicnlp.resources.registry import ProcessorRegistry

        if proc_catalog is None:
            proc_catalog = self._proc_catalog()
        needed: set[str] = set()
        queue = list(requested)
        while queue:
            proc_name = queue.pop()
            if proc_name in needed:
                continue
            needed.add(proc_name)
            backend = self._choose_backend(proc_name, proc_catalog)
            try:
                proc_class = (
                    ProcessorRegistry.get(proc_name, backend)
                    if backend is not None
                    else ProcessorRegistry.get_any(proc_name)
                )
            except ValueError:
                continue
            for req in proc_class.REQUIRES:
                if req not in needed:
                    queue.append(req)

        resolved = [p for p in PROCESSOR_ORDER if p in needed]
        for proc_name in requested:
            if proc_name in needed and proc_name not in resolved:
                resolved.append(proc_name)
        return resolved

    def _build_processors(self) -> None:
        """Resolve, download, and load processors."""
        if self._processors:
            return
        from turkicnlp.resources.registry import ModelRegistry

        with ModelRegistry.using_dir(self.model_dir):
            self._load_plan(self._plan())
        self._owners = self._field_owners()

    def _load_plan(self, plan: list[tuple[str, str]]) -> None:
        import importlib.util

        from turkicnlp.resources.downloader import download, hf_snapshot_complete
        from turkicnlp.resources.registry import ModelRegistry, ProcessorRegistry

        proc_catalog = self._proc_catalog()
        for proc_name, backend in plan:
            backend_key = f"{proc_name}_backend"
            config: dict[str, object] = {"use_gpu": self.use_gpu}
            if self.model_dir:
                config["model_dir"] = self.model_dir
            # Shared LLM settings (llm_model, llm_base_url, llm_api_key, llm_client, ...)
            # reach every processor; "<proc>_llm_*" overrides them for one processor.
            for key, value in self._processor_configs.items():
                if key.startswith("llm_"):
                    config[key] = value
            prefix = f"{proc_name}_"
            for key, value in self._processor_configs.items():
                if key.startswith(prefix) and key != backend_key:
                    config[key[len(prefix):]] = value

            proc_class = ProcessorRegistry.get(proc_name, str(backend))
            proc = proc_class(lang=self.lang, script=self._model_script, config=config)

            proc_info = proc_catalog.get(proc_name, {})
            backend_info = {}
            if isinstance(proc_info, dict):
                backend_info = proc_info.get("backends", {}).get(str(backend), {}) or {}

            model_path = None
            backend_type = backend_info.get("type")
            script = str(self._model_script)
            if backend_type == "huggingface_seq2seq":
                model_name = backend_info.get("model_name")
                shared_dir = (
                    ModelRegistry.default_dir() / "huggingface" / str(model_name).replace("/", "--")
                ) if model_name else None
                if shared_dir is None or not hf_snapshot_complete(shared_dir):
                    download(self.lang, processors=[proc_name], script=script)
            elif backend_type in ("stanza", "stanza_custom"):
                if importlib.util.find_spec("stanza") is None:
                    raise ImportError(
                        f"The '{backend}' backend of '{proc_name}' needs Stanza. "
                        'Install it with: pip install "turkicnlp[stanza]"'
                    )
                if backend_type == "stanza_custom":
                    custom_dir = ModelRegistry.default_dir() / "stanza_custom" / self.lang
                    filename = backend_info.get("filename", f"{proc_name}.pt")
                    pretrain_filename = backend_info.get("pretrain_filename")
                    needs_download = not (custom_dir / filename).exists()
                    if not needs_download and pretrain_filename:
                        needs_download = not (custom_dir / pretrain_filename).exists()
                    if needs_download:
                        download(self.lang, processors=[proc_name], script=script)
            elif backend_type in ("apertium_fst", "neural_model"):
                try:
                    model_path = ModelRegistry.get_model_path(
                        self.lang, proc_name, str(backend), script=script
                    )
                except FileNotFoundError:
                    download(self.lang, processors=[proc_name], script=script)
                    model_path = ModelRegistry.get_model_path(
                        self.lang, proc_name, str(backend), script=script
                    )
            # Glot500 backends download their shared model inside load().

            proc.load(str(model_path) if model_path is not None else "")
            self._processors.append(proc)

    def _field_owners(self) -> dict[str, Processor]:
        owners: dict[str, Processor] = {}
        for proc in self._processors:
            provides = set(getattr(proc, "PROVIDES", []) or [])
            for fld, layer in FIELD_LAYERS.items():
                if layer in provides:
                    owners[fld] = proc
        return owners

    # ------------------------------------------------------------------
    # Running

    def _run_processor(self, proc: Processor, doc: Document) -> Document:
        """Run one processor and keep only the fields it owns."""
        owned = {f for f, o in self._owners.items() if o is proc}
        foreign = {f for f, o in self._owners.items() if o is not proc}
        if not foreign and not owned:
            return proc.process(doc)
        before = {
            id(w): tuple(getattr(w, f) for f in _FIELDS) for w in doc.words
        }
        doc = proc.process(doc)
        label = proc.NAME
        for w in doc.words:
            old = before.get(id(w))
            written = {}
            for i, f in enumerate(_FIELDS):
                new = getattr(w, f)
                if (old is None and new is not None) or (old is not None and new != old[i]):
                    written[f] = new
                    if f in foreign and old is not None:
                        setattr(w, f, old[i])
                    elif f in foreign:
                        setattr(w, f, None)
            if written:
                w._provenance.setdefault(label, {}).update(written)
        return doc

    def _reconcile_feats(self, doc: Document) -> None:
        """Drop FEATS that belong to a different UPOS than the final one."""
        feats_owner = self._owners.get("feats")
        upos_owner = self._owners.get("upos")
        if feats_owner is None or upos_owner is None or feats_owner is upos_owner:
            return
        for w in doc.words:
            rec = w._provenance.get(feats_owner.NAME, {})
            own_upos = rec.get("upos")
            if own_upos is not None and w.upos is not None and own_upos != w.upos:
                w._provenance.setdefault(feats_owner.NAME, {})["feats_dropped"] = w.feats
                w.feats = None

    def _bridge_for(self, script: Optional[Script]) -> Optional[Transliterator]:
        """Transliterator from the input script to the model script, if needed."""
        if self._transliterator is not None:
            return self._transliterator
        if script is None or script == self._model_script:
            return None
        if script not in self._script_config.available:
            return None
        try:
            return Transliterator(self.lang, script, self._model_script)
        except ValueError:
            return None

    def __call__(self, text: str) -> Document:
        """Process a single text string.

        Args:
            text: Input text.

        Returns:
            Annotated :class:`Document`.
        """
        if not self._processors:
            self._build_processors()

        doc = Document(text=text, lang=self.lang)

        input_script: Optional[Script]
        if self._explicit_script:
            input_script = self._script
        else:
            try:
                input_script = detect_script(text)
            except ValueError:
                input_script = None
        if not text.strip():
            doc.script = str(input_script or self._script or self._script_config.primary)
            return doc

        bridge = self._bridge_for(input_script)
        if input_script is None:
            input_script = self._script or self._model_script
        doc.script = str(input_script)

        original_text = text
        if bridge is not None:
            doc._original_text = original_text
            doc.text = bridge.transliterate(text)
            doc.script = str(bridge.target)

        for processor in self._processors:
            doc = self._run_processor(processor, doc)
        self._reconcile_feats(doc)

        if bridge is not None:
            try:
                reverse = Transliterator(self.lang, bridge.target, bridge.source)
            except ValueError:
                reverse = None
            if reverse is not None:
                _restore_original(doc, original_text, reverse)
                doc.script = str(bridge.source)

        return doc

    def _get_asr(self):
        """Create the speech recognizer from ``asr_*`` pipeline options."""
        if self._asr is None:
            from turkicnlp.asr import DEFAULT_MODEL_CARD, DEFAULT_SEGMENT_SECONDS, SpeechRecognizer

            cfg = self._processor_configs
            device = cfg.get("asr_device")
            if device is None and self.use_gpu:
                device = "cuda"
            self._asr = SpeechRecognizer(
                lang=str(cfg.get("asr_lang", self.lang)),
                model_card=str(cfg.get("asr_model_card", DEFAULT_MODEL_CARD)),
                script=str(self._script or self._script_config.primary),
                device=device,
                dtype=str(cfg.get("asr_dtype", "auto")),
                batch_size=int(cfg.get("asr_batch_size", 2)),
                max_segment_seconds=float(
                    cfg.get("asr_max_segment_seconds", DEFAULT_SEGMENT_SECONDS)
                ),
            )
        return self._asr

    def from_audio(self, audio, sample_rate: Optional[int] = None):
        """Transcribe speech and run the text processors on the transcript.

        The transcript is produced in the pipeline's input script (``script``
        or the language's primary script) by Omnilingual ASR
        (:class:`turkicnlp.asr.SpeechRecognizer`), then processed exactly like
        text passed to :meth:`__call__`. Options are passed with the ``asr_``
        prefix, e.g. ``Pipeline("kaz", processors=["asr", "pos"],
        asr_model_card="omniASR_LLM_300M", asr_device="cpu")``.

        Args:
            audio: An audio file path, encoded bytes, a waveform array (with
                ``sample_rate``), a dict with ``waveform``/``sample_rate``,
                or a list of these.
            sample_rate: Sample rate of raw waveform arrays.

        Returns:
            A :class:`Document` (or a list for list input) whose
            ``audio_segments`` hold the time-aligned segment transcripts.
        """
        single = not isinstance(audio, (list, tuple))
        results = self._get_asr().transcribe(
            [audio] if single else list(audio),
            sample_rate=sample_rate,
            return_details=True,
        )
        docs = []
        for res in results:
            doc = self(res.text)
            doc.audio_segments = [
                {"start": s.start, "end": s.end, "text": s.text} for s in res.segments
            ]
            doc._processor_log.insert(0, f"asr:omniasr:{self._asr.model_card}:{res.model_lang}")
            docs.append(doc)
        return docs[0] if single else docs

    def batch(self, texts: list[str], batch_size: int = 32) -> list[Document]:
        """Process multiple texts.

        Texts are processed in chunks of ``batch_size``; within a chunk every
        processor gets all documents at once through
        :meth:`Processor.process_batch`, which neural and LLM backends use to
        batch their model calls.

        Args:
            texts: List of input strings.
            batch_size: Number of documents handed to each processor at once.

        Returns:
            List of annotated documents.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not self._processors:
            self._build_processors()
        out: list[Document] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            if self._transliterator is not None or not self._explicit_script:
                # Script handling is per document.
                out.extend(self(t) for t in chunk)
                continue
            docs = []
            for t in chunk:
                d = Document(text=t, lang=self.lang)
                d.script = str(self._script)
                docs.append(d)
            work = [d for d in docs if d.text.strip()]
            for proc in self._processors:
                if self._owners:
                    work = [self._run_processor(proc, d) for d in work]
                else:
                    work = proc.process_batch(work)
            for d in work:
                self._reconcile_feats(d)
            out.extend(docs)
        return out

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        output_format: str = "conllu",
    ) -> Optional[str]:
        """Process a text file, optionally writing output.

        Args:
            input_path: Path to input text file.
            output_path: Path to write output. ``None`` returns as string.
            output_format: ``conllu`` or ``json``.

        Returns:
            Formatted output string, or ``None`` if written to file.
        """
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        doc = self(text)

        if output_format == "conllu":
            result = doc.to_conllu()
        elif output_format == "json":
            import json

            result = json.dumps(doc.to_dict(full=True), ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unknown format: {output_format}")

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            return None
        return result

    @property
    def processors(self) -> list[str]:
        """List active processor names."""
        return [p.NAME for p in self._processors]

    @property
    def license_info(self) -> dict:
        """Return license information for all loaded components."""
        from turkicnlp.processors.morphology import ApertiumMorphProcessor

        info: dict = {"library": "Apache-2.0", "components": {}}
        for proc in self._processors:
            if isinstance(proc, ApertiumMorphProcessor):
                info["components"][proc.NAME] = {
                    "license": "GPL-3.0",
                    "note": "Apertium FST data downloaded separately; not bundled with library",
                }
            else:
                info["components"][proc.NAME] = {"license": "Apache-2.0"}
        return info

    def __repr__(self) -> str:
        if self._processors:
            procs = " -> ".join(self.processors)
            state = ""
        else:
            try:
                procs = " -> ".join(f"{n}:{b}" for n, b in self._plan())
            except Exception:  # noqa: BLE001 - repr must not fail
                procs = "?"
            state = ", not loaded yet"
        return (
            f"<Pipeline(lang={self.lang}, script={self._model_script}, "
            f"processors=[{procs}]{state})>"
        )


def backend_key_missing(configs: dict, proc_name: str) -> bool:
    """``True`` when the user did not choose a backend for ``proc_name``."""
    return configs.get(f"{proc_name}_backend") is None


def _restore_original(doc: Document, original: str, reverse: Transliterator) -> None:
    """Map a document processed in the model script back to the input text.

    Word and token texts become the matching substrings of ``original`` and
    their offsets index ``original``; lemmas are transliterated back. Tokens
    are located with a running cursor: first by searching the back-
    transliterated token text, otherwise by aligning whitespace-separated
    chunks of the two texts.
    """
    model_text = doc.text
    doc._model_text = model_text  # type: ignore[attr-defined]

    # Chunk alignment (whitespace runs are kept by every transliterator).
    import re as _re

    m_chunks = [(m.start(), m.end()) for m in _re.finditer(r"\S+", model_text)]
    o_chunks = [(m.start(), m.end()) for m in _re.finditer(r"\S+", original)]
    aligned = len(m_chunks) == len(o_chunks)

    def chunk_map(start: int, end: int) -> tuple[Optional[int], Optional[int]]:
        if not aligned:
            return None, None
        for (ms, me), (os_, oe) in zip(m_chunks, o_chunks):
            if ms <= start < me:
                s = os_ if start == ms else None
                e = oe if end == me else None
                if s is None or e is None:
                    # Inside a chunk: proportional position.
                    span_m = max(me - ms, 1)
                    span_o = oe - os_
                    if s is None:
                        s = os_ + round((start - ms) * span_o / span_m)
                    if e is None:
                        e = os_ + round((end - ms) * span_o / span_m)
                return s, max(e, s)
        return None, None

    low = original.lower()
    cursor = 0

    def locate(text: str, start: Optional[int], end: Optional[int]):
        nonlocal cursor
        back = reverse.transliterate(text) if text else text
        if back:
            pos = original.find(back, cursor)
            if pos < 0:
                pos = low.find(back.lower(), cursor)
            if pos >= 0 and (start is None or pos - cursor < 40 + len(back) * 2):
                cursor = pos + len(back)
                return pos, pos + len(back)
        if start is not None and end is not None:
            s, e = chunk_map(start, end)
            if s is not None:
                cursor = max(cursor, e)
                return s, e
        return None, None

    for sent in doc.sentences:
        s_start = s_end = None
        for tok in sent.tokens:
            ts, te = locate(tok.text, tok.start_char, tok.end_char)
            if ts is not None:
                tok.text = original[ts:te]
                tok.start_char, tok.end_char = ts, te
                s_start = ts if s_start is None else s_start
                s_end = te
            else:
                tok.text = reverse.transliterate(tok.text)
            for w in tok.words:
                if len(tok.words) == 1 and ts is not None:
                    w.text = tok.text
                    w.start_char, w.end_char = ts, te
                else:
                    w.text = reverse.transliterate(w.text)
                    w.start_char, w.end_char = tok.start_char, tok.end_char
        # words not covered by tokens (processors that only fill sentence.words)
        if not sent.tokens:
            for w in sent.words:
                ws, we = locate(w.text, w.start_char, w.end_char)
                if ws is not None:
                    w.text = original[ws:we]
                    w.start_char, w.end_char = ws, we
                    s_start = ws if s_start is None else s_start
                    s_end = we
                else:
                    w.text = reverse.transliterate(w.text)
        for w in sent.words:
            if w.lemma:
                w.lemma = reverse.transliterate(w.lemma)
        if s_start is not None:
            sent.text = original[s_start:s_end]
        else:
            sent.text = reverse.transliterate(sent.text)
        for ent in sent.entities:
            if ent.words and ent.words[0].start_char is not None and ent.words[-1].end_char is not None:
                ent.start_char = ent.words[0].start_char
                ent.end_char = ent.words[-1].end_char
                ent.text = original[ent.start_char:ent.end_char]
    doc.text = original
