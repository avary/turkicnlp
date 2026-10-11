"""Prompt templates of the LLM components.

Each component has a JSON template in ``turkicnlp/resources/prompts/<name>.json``::

    {
      "version": 1,
      "system": "... {language} ... {script_name} ...",
      "user": "... {tokens_json} ...",
      "examples": {"default": [{"vars": {...}, "answer": {...}}], "kaz": [...]}
    }

``{name}`` placeholders (a Python identifier in single braces) are replaced by the
variable of that name; placeholders without a variable and all other braces — e.g. a
JSON example such as ``{"translation": "..."}`` — are left exactly as written, so
braces need no escaping. Few-shot examples are rendered with the same ``user`` template and
sent as user/assistant turns before the real input. Language-specific examples are
used when present, otherwise ``default``.

Users can replace a template with ``<processor>_prompt_file="my_prompt.json"`` and add
examples with ``<processor>_examples=[{"vars": {...}, "answer": {...}}, ...]``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Optional

SCRIPT_NAMES = {
    "Latn": "Latin", "Cyrl": "Cyrillic", "Arab": "Perso-Arabic", "Orkh": "Old Turkic runic",
    "CTS": "Common Turkic Alphabet (Latin)",
}

# Non-Turkic target languages for translation prompts (Turkic names come from the catalog).
OTHER_LANGUAGES = {
    "eng": "English", "rus": "Russian", "deu": "German", "fra": "French", "spa": "Spanish",
    "ara": "Arabic", "fas": "Persian", "zho": "Chinese", "ukr": "Ukrainian", "ita": "Italian",
    "jpn": "Japanese", "kor": "Korean", "hin": "Hindi", "mon": "Mongolian", "tgk": "Tajik",
}


def language_name(code: str) -> str:
    """Human-readable name of a language code (Turkic languages from the model catalog)."""
    if code in OTHER_LANGUAGES:
        return OTHER_LANGUAGES[code]
    try:
        from turkicnlp.resources.registry import ModelRegistry

        info = ModelRegistry.load_catalog().get(code)
        if isinstance(info, dict) and info.get("name"):
            return str(info["name"])
    except Exception:  # noqa: BLE001
        pass
    return code


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _fill(template: str, variables: dict) -> str:
    """Replace ``{name}`` for every ``name`` in ``variables`` (one pass: values are not
    searched for placeholders again); leave every other brace untouched."""
    def sub(m: "re.Match[str]") -> str:
        key = m.group(1)
        return str(variables[key]) if key in variables else m.group(0)

    return _PLACEHOLDER.sub(sub, template)


def target_language_name(target: str, script: Optional[str] = None) -> str:
    """Name of a translation target for the prompt, with its script when known.

    Accepts ``uzb``, ``uzb_Latn``, ``uzb-Cyrl``, ``kaz_Cyrl`` or a plain language name;
    ``script`` (e.g. ``"Latn"``) overrides the script part of the code. For a Turkic
    language written in several scripts the catalog's primary script is assumed.
    """
    target = str(target).strip()
    parts = re.split(r"[_-]", target, maxsplit=1)
    code = parts[0]
    code_script = parts[1] if len(parts) > 1 else None
    if not (len(code) in (2, 3) and code.isalpha() and code.islower()):
        return target  # a language name such as "English"
    name = language_name(code)
    script = script or code_script
    if not script:
        try:
            from turkicnlp.resources.registry import ModelRegistry

            scripts = (ModelRegistry.load_catalog().get(code) or {}).get("scripts") or {}
            if len(scripts.get("available") or []) > 1:
                script = scripts.get("primary")
        except Exception:  # noqa: BLE001
            script = None
    if script:
        script = str(script)
        script_name = SCRIPT_NAMES.get(script) or SCRIPT_NAMES.get(script.title(), script)
        return f"{name} ({script_name} script)"
    return name


@dataclass
class PromptTemplate:
    name: str
    version: int
    system: str
    user: str
    examples: dict[str, list[dict]] = field(default_factory=dict)

    @classmethod
    def load(cls, name: str, override: Optional[str] = None) -> "PromptTemplate":
        if override:
            data = json.loads(Path(override).read_text(encoding="utf-8"))
        else:
            data = json.loads(
                resources.files("turkicnlp.resources").joinpath("prompts", f"{name}.json").read_text("utf-8"))
        return cls(name=name, version=int(data.get("version", 1)), system=data["system"],
                   user=data["user"], examples=data.get("examples", {}))

    def render(self, variables: dict[str, Any], lang: Optional[str] = None,
               extra_examples: Optional[list[dict]] = None, n_examples: int = 3) -> list[dict[str, str]]:
        """Build the chat messages: system, few-shot user/assistant turns, user input."""
        msgs = [{"role": "system", "content": _fill(self.system, variables)}]
        examples = list(extra_examples or [])
        examples += self.examples.get(lang or "", []) or self.examples.get("default", [])
        for ex in examples[:n_examples]:
            ex_vars = {**variables, **_example_vars(ex.get("vars", {}))}
            msgs.append({"role": "user", "content": _fill(self.user, ex_vars)})
            answer = ex["answer"]
            msgs.append({"role": "assistant",
                         "content": answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)})
        msgs.append({"role": "user", "content": _fill(self.user, variables)})
        return msgs

    @property
    def tag(self) -> str:
        return f"{self.name}/v{self.version}"


def _example_vars(v: dict) -> dict:
    out = dict(v)
    if "tokens" in out and "tokens_json" not in out:
        out["tokens_json"] = tokens_json(out["tokens"])
    return out


def tokens_json(tokens: list[str]) -> str:
    """Tokens with their indices, one JSON list: ``[[0, "Мен"], [1, "кеше"], ...]``."""
    return json.dumps([[i, t] for i, t in enumerate(tokens)], ensure_ascii=False)


def base_variables(lang: Optional[str], script: Optional[str]) -> dict[str, str]:
    script = str(script) if script else ""
    return {
        "lang": lang or "",
        "language": language_name(lang) if lang else "a Turkic language",
        "script": script,
        "script_name": SCRIPT_NAMES.get(script, script or "its usual"),
    }
