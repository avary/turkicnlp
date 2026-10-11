"""Free-form questions to an LLM with TurkicNLP analyses as context.

    import turkicnlp
    nlp = turkicnlp.Pipeline("kaz", processors=["tokenize", "morph"])
    doc = nlp("Мен кеше мектепке бардым.")
    turkicnlp.llm.ask("Explain the grammar of each verb.", doc=doc,
                      llm_model="gpt-4o-mini", llm_api_key="sk-...")

Here the user writes the prompt; the toolkit adds what it knows about the text
(tokens, lemmas, POS, features, dependencies, entities, translation, Common Turkic
Alphabet transliteration) to the system message.
"""

from __future__ import annotations

from typing import Any, Optional

from turkicnlp.llm.client import LLMClient
from turkicnlp.llm.prompts import PromptTemplate, base_variables


def _clip(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def document_context(doc, include_cts: bool = True, max_words: int = 400) -> str:
    """Render a document's annotations as compact text for an LLM prompt.

    At most ``max_words`` words are rendered: the text lines (text, CTS, translation)
    are cut to ``max_words`` words and word annotations stop after ``max_words`` words
    in total; ``…`` marks the cut."""
    text = doc.text or ""
    lines = [f"Text: {_clip(text, max_words)}"]
    if include_cts and doc.lang and doc.script and doc.script not in ("CTS",):
        try:
            from turkicnlp.scripts import Script
            from turkicnlp.scripts.transliterator import Transliterator

            cts = Transliterator(doc.lang, Script(doc.script), Script.COMMON_TURKIC).transliterate(text)
            if cts != text:
                lines.append(f"Common Turkic Alphabet: {_clip(cts, max_words)}")
        except Exception:  # noqa: BLE001 — not every language/script pair has CTS
            pass
    if doc.translation:
        lines.append(f"Translation: {_clip(doc.translation, max_words)}")
    n = 0
    for si, sent in enumerate(doc.sentences, 1):
        if not sent.words:
            continue
        if n >= max_words:
            lines.append("…")
            break
        lines.append(f"Sentence {si}: {_clip(sent.text, max_words - n)}")
        for w in sent.words:
            if n >= max_words:
                lines.append("  …")
                break
            cols = [f"{w.id}", w.text]
            if w.lemma:
                cols.append(f"lemma={w.lemma}")
            if w.upos:
                cols.append(w.upos)
            if w.feats and w.feats != "_":
                cols.append(w.feats)
            if w.head is not None and w.deprel:
                cols.append(f"{w.deprel}->{w.head}")
            if w.ner and w.ner != "O":
                cols.append(w.ner)
            lines.append("  " + " ".join(cols))
            n += 1
        if sent.correction and sent.correction != sent.text:
            lines.append(f"  Spelling-corrected: {_clip(sent.correction, max_words)}")
        if sent.sentiment:
            lines.append(f"  Sentiment: {sent.sentiment}")
    return "\n".join(lines)


def ask(question: str, doc=None, text: Optional[str] = None, lang: Optional[str] = None,
        client: Optional[LLMClient] = None, prompt_file: Optional[str] = None, max_words: int = 400,
        **llm_options: Any) -> str:
    """Ask ``question`` about ``doc`` (or raw ``text``). LLM options as for the pipeline:
    ``llm_model``, ``llm_base_url``, ``llm_api_key``, … or pass a ``client``.
    ``max_words`` limits the size of the context sent with the question."""
    client = client or LLMClient.from_options(llm_options)
    lang = lang or (doc.lang if doc is not None else None)
    script = doc.script if doc is not None else None
    if doc is not None:
        context = document_context(doc, max_words=max_words)
    elif text:
        context = f"Text: {_clip(text, max_words)}"
    else:
        context = "(no text given)"
    prompt = PromptTemplate.load("ask", prompt_file)
    msgs = prompt.render({**base_variables(lang, script), "context": context, "question": question},
                         n_examples=0)
    return client.chat(msgs).strip()
