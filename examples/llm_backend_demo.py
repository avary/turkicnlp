#!/usr/bin/env python3
"""Run the TurkicNLP LLM backends against a local or hosted LLM.

    # 1) local vLLM server (bash scripts/start_vllm_server.sh)
    python examples/llm_backend_demo.py --backend vllm
    # 2) OpenAI                        (export OPENAI_API_KEY=sk-...)
    python examples/llm_backend_demo.py --backend openai --model gpt-4.1-mini
    # 3) Claude                        (export ANTHROPIC_API_KEY=sk-ant-...)
    python examples/llm_backend_demo.py --backend claude --model claude-sonnet-4-6
    # 4) Ollama on a laptop            (ollama pull qwen2.5:7b)
    python examples/llm_backend_demo.py --backend ollama --model qwen2.5:7b
    # 5) Hugging Face model in-process (no server; small models only)
    python examples/llm_backend_demo.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct

    # other languages / sentences
    python examples/llm_backend_demo.py --backend vllm --lang kjh --text "Пу прай наа чуртас."
"""

from __future__ import annotations

import argparse
import os
import time

import turkicnlp

BACKENDS = {
    "vllm":   dict(llm_base_url="http://localhost:8000/v1", llm_api_key="turkicnlp-local",
                   model="Qwen/Qwen2.5-1.5B-Instruct"),
    "ollama": dict(llm_base_url="http://localhost:11434/v1", llm_api_key="ollama", model="qwen2.5:7b"),
    "openai": dict(llm_base_url="https://api.openai.com/v1", model="gpt-4.1-mini"),  # key: OPENAI_API_KEY
    "claude": dict(llm_provider="anthropic", model="claude-sonnet-4-6"),             # key: ANTHROPIC_API_KEY
    "hf":     dict(llm_provider="hf", model="Qwen/Qwen2.5-1.5B-Instruct"),
}
EXAMPLES = {
    "kaz": "Қасым-Жомарт Тоқаев кеше Астанада шетелдік инвесторлармен кездесті. Бұл кездесу өте пайдалы болды.",
    "tur": "Mustafa Kemal 1919'da Samsun'a çıktı. Bugun hava cok guzel.",
    "uzb": "Shavkat Mirziyoyev Toshkentda yangi maktabni ochdi. Bu juda yaxshi yangilik.",
    "kjh": "Пу прай наа чуртастың алызығлары.",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=BACKENDS, default="vllm")
    ap.add_argument("--model", help="model name (default depends on --backend)")
    ap.add_argument("--base-url", help="override the server URL")
    ap.add_argument("--api-key", help="API key (default: from the environment)")
    ap.add_argument("--lang", default="kaz")
    ap.add_argument("--text")
    ap.add_argument("--target", default="eng", help="translation target (ISO 639-3)")
    args = ap.parse_args()

    cfg = dict(BACKENDS[args.backend])
    model = args.model or cfg.pop("model")
    cfg.pop("model", None)
    cfg["llm_model"] = model
    if args.base_url:
        cfg["llm_base_url"] = args.base_url
    if args.api_key:
        cfg["llm_api_key"] = args.api_key
    if args.backend == "openai" and not (args.api_key or os.environ.get("OPENAI_API_KEY")):
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key")
    if args.backend == "claude" and not (args.api_key or os.environ.get("ANTHROPIC_API_KEY")):
        raise SystemExit("Set ANTHROPIC_API_KEY or pass --api-key")

    text = args.text or EXAMPLES.get(args.lang, EXAMPLES["kaz"])
    t0 = time.time()
    nlp = turkicnlp.Pipeline(
        args.lang,
        processors=["tokenize", "morph", "ner", "sentiment", "spellcheck", "translate"],
        morph_backend="llm", ner_backend="llm", sentiment_backend="llm", translate_backend="llm",
        translate_tgt_lang=args.target,
        **cfg,
    )
    doc = nlp(text)

    print(f"Model: {model} ({args.backend})\nText:  {text}\n")
    print("Morphology:")
    for w in doc.words:
        print(f"  {w.text:22} {w.lemma or '':18} {w.upos or '':6} {w.feats or ''}")
    print("\nEntities:   ", [(e.text, e.type) for e in doc.entities])
    print("Sentiment:  ", [s.sentiment for s in doc.sentences])
    print("Spelling:   ", doc.corrected_text, [c for s in doc.sentences for c in s.corrections])
    print("Translation:", doc.translation)
    print("\nProvenance: ", doc._processor_log)
    print(f"Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
