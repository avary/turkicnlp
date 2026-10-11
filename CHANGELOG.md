# Changelog

All notable changes to TurkicNLP. Versions follow the numbers published on [PyPI](https://pypi.org/project/turkicnlp/).

## 0.5.0 (unreleased)

### Added

- **LLM backend**: `morph=llm`, `morph=apertium_llm`, `ner=llm`, `translate=llm`, `sentiment=llm` and
  `spellcheck=llm` for all languages, through an OpenAI-compatible server (vLLM, Ollama, OpenAI, …), Claude, or an
  in-process Hugging Face model; `turkicnlp.llm.ask()`; `scripts/start_vllm_server.sh` and
  `examples/llm_backend_demo.py`.
- Automatic transliteration for input scripts that have no models (e.g. `Pipeline("uzb", script="Cyrl")`,
  Kazakh Latin): results are mapped back to the input text and offsets.
- `Pipeline.batch(texts, batch_size)` hands `batch_size` documents to each processor (`Processor.process_batch`).
- `Document.to_dict(full=True)` with sentence- and document-level annotations.
- `Pipeline(model_dir=...)` is used; `ModelRegistry.using_dir()`; `TURKICNLP_CATALOG` environment variable.
- Tokenizer: abbreviation lists (`tokenize_abbreviations=[...]` adds more), ordinals, bracket and quote splitting.
- `CHANGELOG.md`; CI runs on pushes and pull requests and runs `pytest -m "not slow"`.

### Changed

- **Python ≥ 3.10** is required (3.9 is no longer supported).
- **Field ownership**: each word field has one owner per pipeline (the last processor that provides it); other
  processors' values are kept in `word._provenance`, and FEATS that belong to a different UPOS are dropped. The
  default set no longer combines the Apertium analyser with the Glot500 `feats`/`lemma` steps.
- Placeholder backends are marked `IMPLEMENTED = False` and never selected automatically; `sentiment` defaults to
  `llm`. Requirements are read from the backend that actually runs.
- The packaged catalog is always used; a `catalog.json` in the model folder no longer hides it after an upgrade.
- Stanza NER always runs on the toolkit's tokens and spreads token tags over multi-word tokens.
- Tag mapping: `aor` → `Aspect=Hab|Tense=Pres`; `ind` → `Mood=Ind` on verbs and `PronType=Ind` on pronouns;
  POS tags inside derivations are no longer read as features; ota, otk and klj use the common Turkic mapping.
- Transliteration: Uzbek `сҳ` ↔ `sʼh`, apostrophe variants, Azerbaijani `ј`, Uyghur case and punctuation,
  Turkmen `е` → `ýe`, CTS output restricted to the Common Turkic Alphabet, all-caps words.
- NLLB translation and embeddings share one model in memory and honour `use_gpu`.
- Glot500: sentences longer than 512 subwords are processed in windows; parse trees are valid (one root, no
  cycles); lemmas keep proper-noun capitals and use Turkic-aware lowercasing; one shared backbone.
- Packaging: `setuptools>=77`, `setup.cfg`, `requirements.txt` and `requirements-all.txt` removed, tests excluded
  from wheels, development status Alpha.

### Fixed

- `Pipeline(lang)` crashed with `NotImplementedError` for crh, chv, gag, alt, tyv, kjh and klj.
- Empty or whitespace-only input crashed every processor after tokenization.
- Old Turkic letters from U+10C14 onwards were shifted (𐱅𐰇𐰼𐰚 now gives `törk`).
- Character offsets were wrong after the first sentence or with leading or repeated whitespace; the Arabic
  tokenizer computed them on kashida-stripped text and split on ZWNJ and in decimals.
- 89 corrupt multi-word-token rules and the placeholder rule in every rules file.
- Ottoman Turkish used the Latin tokenizer.
- The external Apertium tagger output was never read; FST generation always returned `None`.
- Uzbek words with o‘/g‘ were analysed as proper names.
- LLM backend: invalid answers were cached; the JSON-mode fallback stopped too early and downgraded on any
  HTTP 400; one bad response broke a batch; `OPENAI_API_KEY` could be sent to other servers; keys appeared in
  `repr`; the spelling guard rejected corrections next to punctuation and collapsed whitespace; JSON braces in
  custom prompts crashed.
- GlotLID failed on text with line breaks.
- Morpheme tokenizer: `I` was lowercased to `ı` in Uzbek, Turkmen and Karakalpak; `segment(" ")` crashed;
  unbounded cache; `Kur'an'ı` split after `Kur`.
- Downloads: interrupted downloads counted as complete; Stanza models were downloaded before checking that Stanza
  is installed.
- CoNLL-U: `Script=` was duplicated; newlines in sentences produced invalid files; the reader failed on a
  non-integer HEAD, dropped a BOM line and leaked an unclosed multi-word token into the next sentence.
- Stanza changed the global `stanza` logging level; model caches had no locks.
- `split_audio` returned a zero-length segment for empty audio.

## 0.4.0

- Speech recognition with Omnilingual ASR (`SpeechRecognizer`, `Pipeline.from_audio`).
