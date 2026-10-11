#!/usr/bin/env python3
"""Extract morphological suffix rules from Wiktionary JSONL dictionaries.

Reads Wiktionary-derived JSONL files (one entry per line) and extracts
suffix allomorph rules by comparing lemma → inflected forms. Produces
a single JSON file (morpheme_rules.json) consumed by MorphemeTokenizer.

Usage:
    python scripts/extract_suffix_rules.py \
        --dict-dir /path/to/dictionaries \
        --output turkicnlp/resources/morpheme_rules.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Language mapping: Wiktionary filename → ISO 639-3 code, script, harmony type
# ═══════════════════════════════════════════════════════════════════════════════

LANG_META: dict[str, dict] = {
    "Turkish":        {"code": "tur", "script": "Latn", "harmony": "4way",
                       "infinitive_suffix": ("mak", "mek"),
                       "stem_alt": {"p": "b", "ç": "c", "t": "d", "k": "ğ"}},
    "Azerbaijani":    {"code": "aze", "script": "Latn", "harmony": "4way",
                       "infinitive_suffix": ("maq", "mək"),
                       "stem_alt": {"p": "b", "ç": "c", "t": "d", "k": "y"}},
    "Turkmen":        {"code": "tuk", "script": "Latn", "harmony": "4way",
                       "infinitive_suffix": ("mak", "mek"),
                       "stem_alt": {"p": "b", "ç": "c", "t": "d", "k": "g"}},
    "Uzbek":          {"code": "uzb", "script": "Latn", "harmony": "none",
                       "infinitive_suffix": ("moq",),
                       "stem_alt": {"p": "b", "k": "g"}},
    "Crimean_Tatar":  {"code": "crh", "script": "Latn", "harmony": "4way",
                       "infinitive_suffix": ("maq", "mek"),
                       "stem_alt": {"p": "b", "ç": "c", "t": "d", "k": "g"}},
    "Gagauz":         {"code": "gag", "script": "Latn", "harmony": "2way",
                       "infinitive_suffix": ("maa", "mää"),
                       "stem_alt": {"p": "b", "ç": "c", "t": "d", "k": "g"}},
    "Kazakh":         {"code": "kaz", "script": "Cyrl", "harmony": "2way",
                       "infinitive_suffix": ("у",),
                       "stem_alt": {"п": "б", "к": "г", "қ": "ғ"}},
    "Kyrgyz":         {"code": "kir", "script": "Cyrl", "harmony": "2way",
                       "infinitive_suffix": ("уу", "үү"),
                       "stem_alt": {"п": "б", "к": "г"}},
    "Tatar":          {"code": "tat", "script": "Cyrl", "harmony": "2way",
                       "infinitive_suffix": ("у", "ү"),
                       "stem_alt": {"п": "б", "к": "г"}},
    "Bashkir":        {"code": "bak", "script": "Cyrl", "harmony": "2way",
                       "infinitive_suffix": ("у", "ү"),
                       "stem_alt": {"п": "б", "к": "г"}},
    "Chuvash":        {"code": "chv", "script": "Cyrl", "harmony": "2way",
                       "infinitive_suffix": (),
                       "stem_alt": {}},
    "Yakut":          {"code": "sah", "script": "Cyrl", "harmony": "2way",
                       "infinitive_suffix": (),
                       "stem_alt": {}},
    "Kumyk":          {"code": "kum", "script": "Cyrl", "harmony": "2way",
                       "infinitive_suffix": ("макъ", "мек"),
                       "stem_alt": {"п": "б", "к": "г"}},
    "Khalaj":         {"code": "klj", "script": "Latn", "harmony": "2way",
                       "infinitive_suffix": ("mak", "mek"),
                       "stem_alt": {"p": "b", "k": "g"}},
    "Tuvan":          {"code": "tyv", "script": "Cyrl", "harmony": "2way",
                       "infinitive_suffix": (),
                       "stem_alt": {}},
    "Khakas":         {"code": "kjh", "script": "Cyrl", "harmony": "2way",
                       "infinitive_suffix": (),
                       "stem_alt": {}},
    "Southern_Altai":  {"code": "alt", "script": "Cyrl", "harmony": "2way",
                       "infinitive_suffix": (),
                       "stem_alt": {}},
    "Northern_Altai":  {"code": "atv", "script": "Cyrl", "harmony": "2way",
                       "infinitive_suffix": (),
                       "stem_alt": {}},
    "Uyghur":         {"code": "uig", "script": "Cyrl", "harmony": "2way",
                       "infinitive_suffix": (),
                       "stem_alt": {}},
    "Ottoman_Turkish": {"code": "ota", "script": "Latn", "harmony": "4way",
                       "infinitive_suffix": ("mak", "mek"),
                       "stem_alt": {"p": "b", "ç": "c", "t": "d", "k": "ğ"}},
}

# ═══════════════════════════════════════════════════════════════════════════════
# Phonological helpers (mirror morpheme_tokenizer.py)
# ═══════════════════════════════════════════════════════════════════════════════

_CY_BACK_V = set("аоұыуАОҰЫУ")
_CY_FRONT_V = set("еөүіэЕӨҮІЭәӘ")
_CY_VOWELS = _CY_BACK_V | _CY_FRONT_V
_CY_SONORANTS = set("рлйРЛЙ")
_CY_NASALS = set("мнңМНҢ")
_CY_VOICED_OBS = set("бвгғджзБВГҒДЖЗ")
_CY_VOICELESS = set("пфкқтсшщцчхПФКҚТСШЩЦЧХ")

_LA_BACK_V = set("aıouAIOUāĀ")
_LA_FRONT_V = set("eiöüEİÖÜəƏäÄ")
_LA_VOWELS = _LA_BACK_V | _LA_FRONT_V
_LA_SONORANTS = set("rlyRLY")
_LA_NASALS = set("mnMNñÑňŇŋŊ")
_LA_VOICED_OBS = set("bdgğjvzBDGĞJVZǵǦ")
_LA_VOICELESS = set("çfhkpstşÇFHKPSTŞqQ")


def _turkic_lower(text: str) -> str:
    result = []
    for ch in text:
        if ch == "I":
            result.append("ı")
        elif ch == "\u0130":  # İ
            result.append("i")
        else:
            result.append(ch.lower())
    return "".join(result)


def _get_vowels(script: str):
    return _CY_VOWELS if script == "Cyrl" else _LA_VOWELS


def _get_back_vowels(script: str):
    return _CY_BACK_V if script == "Cyrl" else _LA_BACK_V


def _get_front_vowels(script: str):
    return _CY_FRONT_V if script == "Cyrl" else _LA_FRONT_V


def _get_harmony(text: str, script: str) -> str:
    """B(ack) or F(ront) from last vowel."""
    back = _get_back_vowels(script)
    front = _get_front_vowels(script)
    for ch in reversed(text):
        if ch in back:
            return "B"
        if ch in front:
            return "F"
    return "B"


_ROUND_CY = set("оөұүОӨҰҮ")
_ROUND_LA = set("oöuüOÖUÜ")


def _get_harmony4(text: str, script: str) -> str:
    """BU/FU/BR/FR from last vowel."""
    back = _get_back_vowels(script)
    front = _get_front_vowels(script)
    rounds = _ROUND_CY if script == "Cyrl" else _ROUND_LA
    for ch in reversed(text):
        if ch in back:
            return "BR" if ch in rounds else "BU"
        if ch in front:
            return "FR" if ch in rounds else "FU"
    return "BU"


def _char_class(ch: str, script: str) -> str:
    if script == "Cyrl":
        if ch in _CY_VOWELS: return "V"
        if ch in _CY_SONORANTS: return "S"
        if ch in _CY_NASALS: return "N"
        if ch in _CY_VOICED_OBS: return "D"
        return "T"
    else:
        if ch in _LA_VOWELS: return "V"
        if ch in _LA_SONORANTS: return "S"
        if ch in _LA_NASALS: return "N"
        if ch in _LA_VOICED_OBS: return "D"
        return "T"


# ═══════════════════════════════════════════════════════════════════════════════
# Wiktionary tag → internal suffix key mapping
# ═══════════════════════════════════════════════════════════════════════════════

# For nouns: we extract suffix from singular forms only (tag combo → suffix key)
NOUN_TAG_MAP: dict[frozenset, str] = {
    # Plural
    frozenset({"nominative", "plural"}): "pl",
    frozenset({"plural"}): "pl",
    # Case (singular)
    frozenset({"accusative", "singular"}): "acc",
    frozenset({"accusative", "definite", "singular"}): "acc",
    frozenset({"accusative", "definite"}): "acc",
    frozenset({"accusative"}): "acc",
    frozenset({"dative", "singular"}): "dat",
    frozenset({"dative"}): "dat",
    frozenset({"locative", "singular"}): "loc",
    frozenset({"locative"}): "loc",
    frozenset({"ablative", "singular"}): "abl",
    frozenset({"ablative"}): "abl",
    frozenset({"genitive", "singular"}): "gen",
    frozenset({"genitive"}): "gen",
    frozenset({"instrumental", "singular"}): "ins",
    frozenset({"instrumental"}): "ins",
    # Compound: plural + case
    frozenset({"dative", "plural"}): "pl+dat",
    frozenset({"accusative", "plural"}): "pl+acc",
    frozenset({"accusative", "definite", "plural"}): "pl+acc",
    frozenset({"locative", "plural"}): "pl+loc",
    frozenset({"ablative", "plural"}): "pl+abl",
    frozenset({"genitive", "plural"}): "pl+gen",
    frozenset({"instrumental", "plural"}): "pl+ins",
}

# For verbs: We use multiple passes.
# First, extract tense markers from 3rd-person singular (zero person marker).
# Then extract person markers by subtracting tense from 1sg/2sg/etc.

# Tags that indicate "3rd person singular" (used to isolate tense)
_3SG_TAGS = {"third-person", "singular"}
_1SG_TAGS = {"first-person", "singular"}
_2SG_TAGS = {"second-person", "singular"}
_1PL_TAGS = {"first-person", "plural"}
_2PL_TAGS = {"second-person", "plural"}
_3PL_TAGS = {"third-person", "plural"}

# Tense tags we care about
VERB_TENSE_MAP = {
    "aorist": "aor",
    "past": "past",
    "preterite": "past",
    "present": "pres",
    "continuative": "pres",     # Turkish -ıyor
    "future": "fut",
    "inferential": "evid",
    "renarrative": "evid",
    "perfect": "evid",
    "conditional": "cond",
    "necessitative": "neces",
    "obligative": "neces",
    "progressive": "prog",
    "optative": "opt",
    "habitual": "hab",
    "intentive": "intent",
    "presumptive": "presump",
    "evidential": "evid",
}

VERB_NONFINITE_MAP = {
    frozenset({"participle", "non-subject"}): "prc_perf",
    frozenset({"participle", "non-prospective"}): "prc_perf",
    frozenset({"participle", "prospective"}): "prc_fut",
    frozenset({"converb"}): "gna_perf",
    frozenset({"verbal-noun"}): "ger",
}

# Person tags
PERSON_MAP = {
    frozenset({"first-person", "singular"}): "p1sg",
    frozenset({"second-person", "singular"}): "p2sg",
    frozenset({"first-person", "plural"}): "p1pl",
    frozenset({"second-person", "plural"}): "p2pl",
    frozenset({"third-person", "plural"}): "p3pl",
    # Formal 2sg in Kazakh
    frozenset({"formal", "second-person", "singular"}): "p2sg_frm",
    frozenset({"informal", "second-person", "singular"}): "p2sg",
    frozenset({"formal", "second-person", "plural"}): "p2pl_frm",
    frozenset({"informal", "second-person", "plural"}): "p2pl",
    frozenset({"formal", "plural", "second-person"}): "p2pl_frm",
    frozenset({"informal", "plural", "second-person"}): "p2pl",
}

# Tags to ignore entirely
SKIP_TAGS = {
    "romanization", "table-tags", "inflection-template", "alternative",
    "error-unknown-tag", "stem", "canonical",
    "sentence-case", "uppercase", "broken-form",
}
# For nouns we still skip error-unrecognized-form; for verbs we allow it
# because many languages (Kyrgyz, Bashkir, etc.) have useful tense data inside.
NOUN_SKIP_TAGS = SKIP_TAGS | {"error-unrecognized-form"}
VERB_SKIP_TAGS = SKIP_TAGS  # allow error-unrecognized-form through


# ═══════════════════════════════════════════════════════════════════════════════
# Suffix extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_infinitive(word: str, inf_suffixes: tuple[str, ...]) -> str:
    """Remove infinitive suffix to get verb stem (e.g., ekmek → ek)."""
    w = _turkic_lower(word)
    for s in sorted(inf_suffixes, key=len, reverse=True):
        if w.endswith(s) and len(w) > len(s):
            return w[:-len(s)]
    return w


def _extract_suffix(form: str, lemma_stem: str, stem_alt: dict) -> str | None:
    """Extract suffix by comparing form to lemma stem.

    Returns the suffix string, or None if extraction fails.
    """
    fl = _turkic_lower(form)
    sl = _turkic_lower(lemma_stem)

    # Direct prefix match
    if fl.startswith(sl):
        return fl[len(sl):]

    # Try consonant alternation: lemma ends in X, form starts with stem where X→Y
    if sl and stem_alt:
        last = sl[-1]
        if last in stem_alt:
            alt_stem = sl[:-1] + stem_alt[last]
            if fl.startswith(alt_stem):
                return fl[len(alt_stem):]

    # Reverse alternation: sometimes the dictionary lemma has the voiced form
    if sl and stem_alt:
        rev_alt = {v: k for k, v in stem_alt.items()}
        last = sl[-1]
        if last in rev_alt:
            alt_stem = sl[:-1] + rev_alt[last]
            if fl.startswith(alt_stem):
                return fl[len(alt_stem):]

    # Longest common prefix fallback
    lcp = 0
    for a, b in zip(fl, sl):
        if a == b:
            lcp += 1
        else:
            break

    if lcp >= len(sl) - 1 and lcp >= 2:
        return fl[lcp:]

    return None


def _process_noun_entries(entries: list[dict], meta: dict) -> dict:
    """Process noun entries and extract suffix allomorphs."""
    script = meta["script"]
    stem_alt = meta["stem_alt"]
    suffix_data = defaultdict(list)  # key → [(suffix, stem, harmony, char_class)]

    for entry in entries:
        word = entry["word"]
        word_lower = _turkic_lower(word)

        for form_entry in entry.get("forms", []):
            tags = set(form_entry.get("tags", []))
            form = form_entry.get("form", "")

            # Skip bad entries
            if tags & NOUN_SKIP_TAGS:
                continue
            if " " in form or form == "-" or not form:
                continue
            if form == word:
                continue
            # Skip possessive/3sg possessed forms (complex)
            if "possessive" in tags or "possessed-single" in tags or "possessed-many" in tags:
                continue

            tag_key_raw = tags - {"definite"}  # strip "definite" since Wikt uses it inconsistently
            tag_key = frozenset(tag_key_raw)

            suffix_key = NOUN_TAG_MAP.get(tag_key)
            if not suffix_key:
                continue

            # For compound keys like "pl+dat", we need the plural form first
            if "+" in suffix_key:
                # We'll handle compound suffix extraction in a second pass
                continue

            suffix = _extract_suffix(form, word, stem_alt)
            if suffix is None or len(suffix) == 0:
                continue
            if len(suffix) > 8:
                continue  # too long — probably bad extraction

            harmony = _get_harmony(word_lower, script)
            last_char = word_lower[-1] if word_lower else ""
            cc = _char_class(last_char, script)
            h4 = _get_harmony4(word_lower, script)

            suffix_data[suffix_key].append({
                "suffix": suffix,
                "harmony": harmony,
                "harmony4": h4,
                "char_class": cc,
                "lemma": word,
            })

    return dict(suffix_data)


def _process_verb_entries(entries: list[dict], meta: dict) -> dict:
    """Process verb entries and extract verb suffix allomorphs.

    Strategy — person/number tags in Wiktionary verb tables are often SHIFTED
    (off-by-one or off-by-two), so we DO NOT rely on them.  Instead:

    1. Group all forms for a given verb by tense tag (aorist, past, …).
    2. Within each tense group, the **shortest** form is assumed to be 3sg
       (which typically has a zero person marker in Turkic).
    3. The tense suffix is extracted by comparing the shortest form to the verb stem.
    4. Person suffixes are extracted by diffing longer forms against the shortest.
    5. Non-finite forms (participle, converb) are extracted directly from the stem.
    6. Negative suffix is extracted from forms tagged ``negative``.
    """
    script = meta["script"]
    stem_alt = meta["stem_alt"]
    inf_suffixes = meta["infinitive_suffix"]
    suffix_data = defaultdict(list)

    for entry in entries:
        word = entry["word"]
        verb_stem = _strip_infinitive(word, inf_suffixes)

        forms = entry.get("forms", [])
        if len(forms) < 3:
            continue  # not enough data

        # ------------------------------------------------------------------
        # Pass 1: bucket forms by tense tag (ignore person tags completely)
        # ------------------------------------------------------------------
        tense_buckets: dict[str, list[str]] = defaultdict(list)
        neg_forms: list[str] = []

        for form_entry in forms:
            tags = set(form_entry.get("tags", []))
            form = form_entry.get("form", "")

            if tags & VERB_SKIP_TAGS:
                continue
            if " " in form or form == "-" or not form:
                continue

            is_negative = "negative" in tags

            # Non-finite forms — extract directly from stem
            for nf_tags, nf_key in VERB_NONFINITE_MAP.items():
                if nf_tags <= tags and not is_negative:
                    suffix = _extract_suffix(form, verb_stem, stem_alt)
                    if suffix and 0 < len(suffix) <= 10:
                        _record_suffix(suffix_data, nf_key, suffix, verb_stem, script)
                    break

            if is_negative:
                neg_forms.append(form)
                continue

            # Identify tense from tags
            tense_key = None
            for tag in tags:
                if tag in VERB_TENSE_MAP:
                    tense_key = VERB_TENSE_MAP[tag]
                    break
            if not tense_key and "imperative" in tags:
                tense_key = "imp"
            if tense_key:
                tense_buckets[tense_key].append(form)

        # ------------------------------------------------------------------
        # Pass 2: for each tense, extract tense suffix from shortest form (3sg)
        #         and person suffixes by diffing longer forms against it.
        # ------------------------------------------------------------------
        for tense_key, tense_forms in tense_buckets.items():
            if not tense_forms:
                continue

            # De-duplicate and lowercase
            tense_forms_lower = sorted(set(_turkic_lower(f) for f in tense_forms), key=len)
            shortest = tense_forms_lower[0]

            # Extract tense suffix from shortest form (= 3sg)
            tense_suffix = _extract_suffix(shortest, verb_stem, stem_alt)
            if tense_suffix is not None and 0 < len(tense_suffix) <= 12:
                _record_suffix(suffix_data, tense_key, tense_suffix, verb_stem, script)

            if tense_key == "imp" or not tense_suffix:
                continue

            # Extract person suffixes by diffing other forms against shortest
            for form_lower in tense_forms_lower[1:]:
                if form_lower == shortest:
                    continue
                if form_lower.startswith(shortest):
                    psuffix = form_lower[len(shortest):]
                    if psuffix and len(psuffix) <= 8:
                        _record_person_suffix(suffix_data, psuffix, shortest, script)
                else:
                    # Try LCP approach
                    lcp = 0
                    for a, b in zip(form_lower, shortest):
                        if a == b:
                            lcp += 1
                        else:
                            break
                    if lcp >= len(shortest) - 2 and lcp >= 2:
                        psuffix = form_lower[lcp:]
                        if psuffix and len(psuffix) <= 8:
                            _record_person_suffix(suffix_data, psuffix, form_lower[:lcp], script)

        # ------------------------------------------------------------------
        # Pass 3: extract negative marker from negative forms
        # ------------------------------------------------------------------
        for form in neg_forms:
            suffix = _extract_suffix(form, verb_stem, stem_alt)
            if suffix and 1 < len(suffix) <= 15:
                fl = _turkic_lower(suffix)
                # Negative morpheme candidates
                for neg in ("ма", "ме", "мә", "мА", "ба", "бе", "бә",
                            "па", "пе", "пә",
                            "ma", "me", "mə", "mä"):
                    if fl.startswith(neg):
                        _record_suffix(suffix_data, "neg", neg, verb_stem, script)
                        break

    return dict(suffix_data)


def _record_person_suffix(suffix_data, psuffix, preceding, script):
    """Try to identify and record which person slot a suffix belongs to."""
    # Common person endings (language-agnostic patterns)
    # We record ALL of them under a generic key; clustering will sort them out.
    # Use heuristic matching based on well-known Turkic person markers:
    p = psuffix
    # 1sg: -m, -(ı/i/u/ü)m, -(ы/і)м
    if p in ("m", "ım", "im", "um", "üm", "ym", "ам", "ем",
             "м", "ым", "ім", "ұм", "үм"):
        _record_suffix(suffix_data, "p1sg", p, preceding, script)
    # 2sg: -n, -sın/sin/sun/sün, -ң, -сың/сің
    elif p in ("n", "sın", "sin", "sun", "sün", "ň",
               "ң", "сың", "сің", "сұң", "сүң",
               "ñ", "siñ", "sıñ", "suñ", "süñ",
               "sан", "sən"):
        _record_suffix(suffix_data, "p2sg", p, preceding, script)
    # 1pl: -k, -(ı/i/u/ü)z, -mız/miz, -қ/-к
    elif p in ("k", "ız", "iz", "uz", "üz", "ıq", "ik",
               "қ", "к", "мыз", "міз", "быз", "біз",
               "mız", "miz", "muz", "müz"):
        _record_suffix(suffix_data, "p1pl", p, preceding, script)
    # 2pl: -sınız, -ңыз, -ñız, etc.
    elif any(p.startswith(x) for x in ("sınız", "siniz", "sunuz", "sünüz",
                                        "сыңыз", "сіңіз",
                                        "sıñız", "siñiz", "suñuz", "süñüz",
                                        "ңыз", "ңіз", "ñız", "ñiz")):
        _record_suffix(suffix_data, "p2pl", p, preceding, script)
    elif any(p.startswith(x) for x in ("lar", "ler", "лар", "лер",
                                        "лАр", "lär")):
        _record_suffix(suffix_data, "p3pl", p, preceding, script)
    # Catch common 2sg patterns with -s- onset
    elif p.startswith("s") and len(p) <= 5:
        _record_suffix(suffix_data, "p2sg", p, preceding, script)
    elif p.startswith("с") and len(p) <= 5:
        _record_suffix(suffix_data, "p2sg", p, preceding, script)


def _record_suffix(suffix_data, key, suffix, stem, script):
    """Record a suffix observation."""
    stem_lower = _turkic_lower(stem)
    harmony = _get_harmony(stem_lower, script)
    h4 = _get_harmony4(stem_lower, script)
    last_char = stem_lower[-1] if stem_lower else ""
    cc = _char_class(last_char, script)
    suffix_data[key].append({
        "suffix": _turkic_lower(suffix),
        "harmony": harmony,
        "harmony4": h4,
        "char_class": cc,
        "lemma": stem,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Allomorph clustering: group extracted suffixes into rules
# ═══════════════════════════════════════════════════════════════════════════════

def _cluster_allomorphs(
    suffix_observations: list[dict],
    harmony_type: str,
) -> list[dict]:
    """Cluster suffix observations into condition-based allomorph rules.

    Returns a list of conditions in the JSON format:
    [{"context": "VS", "forms": {"B": "lar", "F": "ler"}}]
    """
    if not suffix_observations:
        return []

    # Group by (char_class, harmony/harmony4) → Counter of suffix forms
    if harmony_type == "4way":
        groups = defaultdict(Counter)  # char_class → Counter of (suffix, harmony4)
        for obs in suffix_observations:
            cc = obs["char_class"]
            h4 = obs["harmony4"]
            groups[cc][(obs["suffix"], h4)] += 1
    elif harmony_type == "none":
        groups = defaultdict(Counter)
        for obs in suffix_observations:
            cc = obs["char_class"]
            groups[cc][(obs["suffix"], "B")] += 1  # treat all as back
    else:  # 2way
        groups = defaultdict(Counter)
        for obs in suffix_observations:
            cc = obs["char_class"]
            h = obs["harmony"]
            groups[cc][(obs["suffix"], h)] += 1

    # For each char_class, find the most common suffix per harmony class
    cc_rules = {}  # char_class → harmony_dict
    for cc, counter in groups.items():
        if harmony_type == "4way":
            best = {}
            for h4_key in ["BU", "FU", "BR", "FR"]:
                candidates = [(suf, cnt) for (suf, h4), cnt in counter.items() if h4 == h4_key]
                if candidates:
                    best[h4_key] = max(candidates, key=lambda x: x[1])[0]
            if best:
                cc_rules[cc] = best
        elif harmony_type == "none":
            candidates = [(suf, cnt) for (suf, _), cnt in counter.items()]
            if candidates:
                top = max(candidates, key=lambda x: x[1])[0]
                cc_rules[cc] = {"B": top, "F": top}
        else:  # 2way
            best = {}
            for h_key in ["B", "F"]:
                candidates = [(suf, cnt) for (suf, h), cnt in counter.items() if h == h_key]
                if candidates:
                    best[h_key] = max(candidates, key=lambda x: x[1])[0]
            if best:
                cc_rules[cc] = best

    if not cc_rules:
        return []

    # Merge character classes that have the same allomorph set
    form_groups = defaultdict(set)  # frozen(forms) → set of char classes
    for cc, forms in cc_rules.items():
        frozen_forms = tuple(sorted(forms.items()))
        form_groups[frozen_forms].add(cc)

    # Build the condition list (ordered by number of classes, largest first)
    conditions = []
    for frozen_forms, classes in sorted(form_groups.items(), key=lambda x: -len(x[1])):
        context = "".join(sorted(classes, key=lambda c: "VSNDT".index(c) if c in "VSNDT" else 9))
        forms = dict(frozen_forms)
        conditions.append({"context": context, "forms": forms})

    return conditions


def _fill_missing_harmony(conditions: list[dict], harmony_type: str) -> list[dict]:
    """Fill in missing harmony keys by copying from available ones."""
    for cond in conditions:
        forms = cond["forms"]
        if harmony_type == "4way":
            # Fill missing 4way keys
            if "BU" not in forms and "B" in forms:
                forms["BU"] = forms.pop("B")
            if "FU" not in forms and "F" in forms:
                forms["FU"] = forms.pop("F")
            # Copy rounded from unrounded if missing
            if "BR" not in forms and "BU" in forms:
                forms["BR"] = forms["BU"]
            if "FR" not in forms and "FU" in forms:
                forms["FR"] = forms["FU"]
            if "BU" not in forms and "BR" in forms:
                forms["BU"] = forms["BR"]
            if "FU" not in forms and "FR" in forms:
                forms["FU"] = forms["FR"]
        else:
            # 2way: fill missing
            if "B" not in forms and "F" in forms:
                forms["B"] = forms["F"]
            if "F" not in forms and "B" in forms:
                forms["F"] = forms["B"]
    return conditions


# ═══════════════════════════════════════════════════════════════════════════════
# Main extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def extract_language(dict_path: str, lang_name: str, meta: dict) -> dict:
    """Extract all suffix rules for one language."""
    logger.info(f"Processing {lang_name} ({meta['code']})...")

    # Read entries
    nouns = []
    verbs = []
    adjectives = []
    total = 0

    with open(dict_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            total += 1
            pos = entry.get("pos", "")
            if pos == "noun" and entry.get("forms"):
                nouns.append(entry)
            elif pos == "verb" and entry.get("forms"):
                verbs.append(entry)
            elif pos == "adj" and entry.get("forms"):
                adjectives.append(entry)

    logger.info(f"  {lang_name}: {total} entries, {len(nouns)} nouns, {len(verbs)} verbs, {len(adjectives)} adjectives")

    # Extract raw suffix observations
    noun_suffixes = _process_noun_entries(nouns + adjectives, meta)
    verb_suffixes = _process_verb_entries(verbs, meta)

    harmony_type = meta["harmony"]

    # Cluster into rules
    noun_rules = {}
    for key, observations in noun_suffixes.items():
        conditions = _cluster_allomorphs(observations, harmony_type)
        conditions = _fill_missing_harmony(conditions, harmony_type)
        if conditions:
            noun_rules[key] = {"conditions": conditions, "source": "wiktionary",
                               "n_observations": len(observations)}

    verb_rules = {}
    for key, observations in verb_suffixes.items():
        conditions = _cluster_allomorphs(observations, harmony_type)
        conditions = _fill_missing_harmony(conditions, harmony_type)
        if conditions:
            verb_rules[key] = {"conditions": conditions, "source": "wiktionary",
                               "n_observations": len(observations)}

    return {
        "code": meta["code"],
        "script": meta["script"],
        "harmony_type": harmony_type,
        "stem_alternations": meta["stem_alt"],
        "noun_suffixes": noun_rules,
        "verb_suffixes": verb_rules,
        "stats": {
            "total_entries": total,
            "nouns_with_forms": len(nouns),
            "verbs_with_forms": len(verbs),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Additional languages not in Wiktionary (Cyrillic fallback copies)
# ═══════════════════════════════════════════════════════════════════════════════

# Languages that don't have Wiktionary dictionaries but need entries
# These will be added as stubs with source="fallback"
ADDITIONAL_LANGS = {
    "nog": {"script": "Cyrl", "harmony_type": "2way", "fallback_from": "kaz",
            "stem_alternations": {"п": "б", "к": "г"}},
    "krc": {"script": "Cyrl", "harmony_type": "2way", "fallback_from": "kaz",
            "stem_alternations": {"п": "б", "к": "г"}},
    "kaa": {"script": "Latn", "harmony_type": "2way", "fallback_from": "kaz",
            "stem_alternations": {"p": "b", "k": "g"}},
    "azb": {"script": "Arab", "harmony_type": "2way", "fallback_from": "aze",
            "stem_alternations": {}},
    "atv": {"script": "Cyrl", "harmony_type": "2way", "fallback_from": "alt",
            "stem_alternations": {}},
}


def _export_hardcoded_tables() -> dict[str, dict]:
    """Export the existing hardcoded suffix tables from morpheme_tokenizer.py to JSON format.

    These serve as the baseline / manual rules. Wiktionary-extracted rules
    are merged on top where they have enough observations.
    """
    # Import the hardcoded tables directly
    from turkicnlp.processors.morpheme_tokenizer import _LANG_SUFFIX_MAP, _STEM_ALTERNATIONS

    def _entries_to_json(entries: list[tuple[str, tuple]]) -> list[dict]:
        """Convert internal (context, forms_tuple) to JSON conditions."""
        conditions = []
        for ctx, forms in entries:
            if len(forms) == 2:
                conditions.append({"context": ctx, "forms": {"B": forms[0], "F": forms[1]}})
            elif len(forms) == 4:
                conditions.append({"context": ctx, "forms": {
                    "BU": forms[0], "FU": forms[1], "BR": forms[2], "FR": forms[3],
                }})
        return conditions

    # Determine harmony type per language
    _HARMONY_TYPES = {
        "tur": "4way", "aze": "4way", "tuk": "4way", "crh": "4way", "gag": "2way",
        "kaz": "2way", "kir": "2way", "tat": "2way", "bak": "2way",
        "kaa": "2way", "kum": "2way", "uzb": "none", "klj": "2way",
        "uig": "2way", "sah": "2way", "kjh": "2way", "tyv": "2way",
        "alt": "2way", "chv": "2way", "ota": "4way",
        "azb": "2way", "nog": "2way", "krc": "2way",
    }

    # Classify keys into noun vs verb
    _NOUN_KEYS = {"pl", "px1sg", "px2sg", "px2sg_frm", "px3sg", "px3sp",
                   "px1pl", "px2pl", "px3pl",
                   "acc", "dat", "loc", "abl", "gen", "ins", "equ",
                   "subst", "attr", "ly", "siz", "lyk", "shi"}
    _VERB_KEYS = {"neg", "pass", "caus", "coop",
                   "past", "pres", "fut", "aor", "evid", "imp", "opt", "cond", "neces",
                   "p1sg", "p2sg", "p3sg", "p1pl", "p2pl", "p3pl",
                   "gna_perf", "gna_impf", "prc_perf", "prc_impf", "prc_fut",
                   "inf", "ger", "ger_past", "cop", "qst",
                   "prog", "hab", "intent", "presump"}

    exported = {}
    for lang_code, (table, script) in _LANG_SUFFIX_MAP.items():
        noun_suffixes = {}
        verb_suffixes = {}

        for key, entries in table.items():
            if not entries:  # skip empty entries like imp: []
                continue
            conditions = _entries_to_json(entries)
            rule = {"conditions": conditions, "source": "manual"}

            if key in _NOUN_KEYS:
                noun_suffixes[key] = rule
            elif key in _VERB_KEYS:
                verb_suffixes[key] = rule
            else:
                # Default to verb for unknown keys
                verb_suffixes[key] = rule

        stem_alt = _STEM_ALTERNATIONS.get(lang_code, {})

        exported[lang_code] = {
            "code": lang_code,
            "script": script,
            "harmony_type": _HARMONY_TYPES.get(lang_code, "2way"),
            "stem_alternations": stem_alt,
            "noun_suffixes": noun_suffixes,
            "verb_suffixes": verb_suffixes,
            "stats": {
                "total_entries": 0,
                "nouns_with_forms": 0,
                "verbs_with_forms": 0,
            },
        }

    return exported


def _merge_wiktionary_into_manual(manual: dict, wiktionary: dict, min_obs: int = 5) -> dict:
    """Merge Wiktionary-extracted rules into manual baseline.

    - Manual rules are the baseline (source: "manual")
    - Wiktionary rules override manual IF they have >= min_obs observations
    - Keys only in Wiktionary are added
    - Keys only in manual are kept
    """
    merged = json.loads(json.dumps(manual))  # deep copy

    for section in ("noun_suffixes", "verb_suffixes"):
        manual_section = merged.get(section, {})
        wikt_section = wiktionary.get(section, {})

        for key, wikt_rule in wikt_section.items():
            n_obs = wikt_rule.get("n_observations", 0)
            if n_obs >= min_obs:
                # Wiktionary data is good enough — use it
                manual_section[key] = wikt_rule
            elif key not in manual_section:
                # No manual rule exists — use Wiktionary even if low obs
                manual_section[key] = wikt_rule

        merged[section] = manual_section

    # Update stats from Wiktionary
    if "stats" in wiktionary:
        merged["stats"] = wiktionary["stats"]

    return merged


def _inject_linking_rules(languages: dict) -> None:
    """Inject linking consonant rules (n/y/s) for suffix chaining."""
    case_keys = {"acc", "dat", "loc", "abl", "gen", "ins", "equ"}
    prev_suffixes = ["px3sg", "px3pl", "px3sp"]
    for lang_data in languages.values():
        noun = lang_data.get("noun_suffixes", {})
        for key in case_keys:
            rule = noun.get(key)
            if not rule:
                continue
            for cond in rule.get("conditions", []):
                cond.setdefault("linking", {"chars": ["n"], "prev_suffixes": prev_suffixes})

def build_json(dict_dir: str, output_path: str):
    """Build the complete morpheme_rules.json.

    1. Export hardcoded suffix tables as baseline ("manual" source)
    2. Extract rules from Wiktionary dictionaries ("wiktionary" source)
    3. Merge: Wiktionary overrides manual where it has enough data
    4. Write the combined result
    """
    # Step 1: Export hardcoded tables
    logger.info("Exporting hardcoded suffix tables as baseline...")
    manual_tables = _export_hardcoded_tables()
    logger.info(f"  Exported {len(manual_tables)} languages from hardcoded tables")

    # Step 2: Extract from Wiktionary
    wiktionary_data = {}
    for lang_name, meta in LANG_META.items():
        dict_path = os.path.join(dict_dir, f"{lang_name}.jsonl")
        if not os.path.exists(dict_path):
            logger.warning(f"  No dictionary file for {lang_name}, skipping")
            continue
        lang_data = extract_language(dict_path, lang_name, meta)
        code = meta["code"]
        wiktionary_data[code] = lang_data

    # Step 3: Merge — for each language, overlay Wiktionary on top of manual
    languages = {}
    for lang_code, manual in manual_tables.items():
        if lang_code in wiktionary_data:
            merged = _merge_wiktionary_into_manual(manual, wiktionary_data[lang_code])
            languages[lang_code] = merged
        else:
            languages[lang_code] = manual

    # Add any languages from Wiktionary that aren't in the hardcoded tables
    for code, wikt_data in wiktionary_data.items():
        if code not in languages:
            languages[code] = wikt_data

    # Add fallback languages (empty or missing get rules from a related language)
    for code, info in ADDITIONAL_LANGS.items():
        existing = languages.get(code)
        n_rules = 0
        if existing:
            n_rules = (len(existing.get("noun_suffixes", {}))
                       + len(existing.get("verb_suffixes", {})))
        if n_rules > 5:
            continue  # already has enough rules
        fallback = info["fallback_from"]
        if fallback in languages:
            logger.info(f"Creating fallback for {code} from {fallback}")
            fb_data = json.loads(json.dumps(languages[fallback]))
            fb_data["code"] = code
            fb_data["script"] = info["script"]
            fb_data["harmony_type"] = info["harmony_type"]
            fb_data["stem_alternations"] = info["stem_alternations"]
            for section in ("noun_suffixes", "verb_suffixes"):
                for key in fb_data.get(section, {}):
                    fb_data[section][key]["source"] = "fallback"
            languages[code] = fb_data

    output = {
        "version": "2.0",
        "generated": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Morpheme suffix rules for Turkic languages. "
            "Combines manually curated rules (source: manual) with "
            "Wiktionary-extracted rules (source: wiktionary). "
            "Consumed by MorphemeTokenizer."
        ),
        "languages": languages,
    }
    _inject_linking_rules(output["languages"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\nGenerated {output_path}")
    print(f"Languages: {len(languages)}")
    for code, data in sorted(languages.items()):
        n_noun = len(data.get("noun_suffixes", {}))
        n_verb = len(data.get("verb_suffixes", {}))
        sources = set()
        for section in ("noun_suffixes", "verb_suffixes"):
            for v in data.get(section, {}).values():
                sources.add(v.get("source", "?"))
        src_str = "+".join(sorted(sources))
        print(f"  {code}: {n_noun} noun, {n_verb} verb rules  [{src_str}]")


def main():
    parser = argparse.ArgumentParser(description="Extract suffix rules from Wiktionary dictionaries")
    parser.add_argument("--dict-dir", type=str,
                        default="/Users/sherzodhakimov/Work/Projects/turkic-nlp/train-unified-models/dictionaries",
                        help="Path to directory containing Wiktionary JSONL files")
    parser.add_argument("--output", type=str,
                        default="turkicnlp/resources/morpheme_rules.json",
                        help="Output JSON file path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    build_json(args.dict_dir, args.output)


if __name__ == "__main__":
    main()
