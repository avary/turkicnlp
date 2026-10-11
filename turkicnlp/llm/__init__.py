"""LLM integration for TurkicNLP.

* :class:`LLMClient` — one client for OpenAI-compatible servers (OpenAI, vLLM, Ollama,
  llama.cpp, …) and local Hugging Face models.
* LLM backends of the pipeline processors (``morph``, ``ner``, ``translate``,
  ``sentiment``, ``spellcheck``) — see :mod:`turkicnlp.processors.llm_backend`.
* :func:`ask` — a free-form question about a text, with TurkicNLP's analyses as context.
"""

from turkicnlp.llm.client import LLMClient, LLMConfig, LLMError, extract_json
from turkicnlp.llm.prompts import PromptTemplate, language_name
from turkicnlp.llm.ask import ask, document_context

__all__ = ["LLMClient", "LLMConfig", "LLMError", "PromptTemplate", "ask", "document_context",
           "extract_json", "language_name"]
