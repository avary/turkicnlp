"""
Tokenizers for Latin and Cyrillic script Turkic languages.

Provides :class:`RegexTokenizer` (rule-based, handles ~80% of cases)
and :class:`NeuralTokenizer` (BiLSTM character-level, for edge cases).

Character offsets (``start_char`` / ``end_char``) always index the
original ``doc.text``: tokens are located in the original string, so any
amount of whitespace (leading, between words, between sentences, newlines)
is handled exactly.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

from turkicnlp.models.document import Document, Sentence, Token, Word
from turkicnlp.processors.base import Processor


# ---------------------------------------------------------------------------
# Abbreviations
# ---------------------------------------------------------------------------
# All entries are lower-cased and include the final period.  Two classes:
#
# * TITLE abbreviations precede a name/number and never end a sentence
#   ("Prof. Aliyev keldi.").
# * TERMINAL abbreviations ("etc."-type, units, years) may legitimately end a
#   sentence: the sentence is closed only when the next word is capitalised.
#
# Multi-word entries ("va b.", "и т.д.") are matched against the previous
# token plus the current one.  Any token of the shape ``x.y.`` / ``x.y.z.``
# (e.g. "т.б.", "ж.т.б.", "M.Ö.") is treated as a TERMINAL abbreviation too.

TITLE_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        # Latin script (tur, aze, uzb, tuk, crh, kaa, gag, ...)
        "prof.", "dr.", "doç.", "doc.", "dots.", "akad.", "yrd.", "doç.dr.",
        "müh.", "uzm.", "öğr.", "arş.", "alb.", "kpt.", "bşk.", "sn.",
        "st.", "mr.", "mrs.", "ms.", "jr.", "bkz.", "bk.", "örn.", "msl.",
        "nr.", "no.", "ko'ch.", "ko‘ch.", "koʻch.", "ред.",
        # Cyrillic script (kaz, kir, tat, bak, uzb-Cyrl, sah, chv, ...)
        "проф.", "др.", "доц.", "акад.", "ген.", "полк.", "пр.", "просп.",
        "пер.", "гр.", "тов.", "см.", "ср.", "табл.", "стр.", "ст.", "обл.",
        "р-н.", "ауд.",
    }
)

TERMINAL_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        # "and so on" style (Latin)
        "vs.", "vb.", "vd.", "ve s.", "va b.", "va h.k.", "h.k.", "v.b.",
        "va boshq.", "boshq.", "vəs.", "və s.", "və b.", "we ş.m.", "ş.m.",
        "we başg.", "başg.", "ve b.", "etc.", "h.b.",
        # words that are also ordinary words: only treated as abbreviations
        # when the next word is lower-case/numeric
        "av.", "gen.", "org.", "kur.", "ak.", "tel.", "vil.", "mah.", "cad.",
        "sok.", "ul.", "gör.", "red.", "nəşr.", "bet.", "sah.", "say.",
        "mas.", "s.", "sh.", "v.", "t.", "c.", "j.", "ş.",
        "им.", "ул.", "көш.", "кӧч.", "кўч.", "мыс.", "мис.", "мас.", "мәс.",
        "қар.", "кар.", "рис.", "б.", "бет.", "с.", "т.", "ш.",
        # "and so on" style (Cyrillic)
        "т.б.", "т.с.", "ж.б.", "ж.т.б.", "у.б.", "һ.б.", "в.б.", "б.а.",
        "и т.д.", "и т.п.", "т.д.", "т.п.", "ва б.", "ва ҳ.к.", "ҳ.к.",
        "һ.б.", "һәм б.", "һ.к.", "ж.к.", "б.у.", "уо.д.",
        # years / eras / centuries
        "y.", "yy.", "yıl.", "yil.", "ж.", "жж.", "жыл.", "г.", "гг.", "й.",
        "йй.", "ҫ.", "сыл.", "ғ.", "ғғ.", "в.", "вв.",
        "m.ö.", "m.s.", "mö.", "ms.", "b.c.", "a.d.", "м.э.", "б.з.д.", "б.з.",
        "б.д.ж.", "з.б.", "к.ә.", "м.ө.", "м.с.",
        # numbers / units commonly abbreviated with a period
        "mln.", "mlrd.", "млн.", "млрд.", "тыс.", "мың.", "trln.", "трлн.",
        "min.", "sn.", "sek.", "saat.", "мин.", "сек.", "сағ.", "саат.",
        "km.", "sm.", "kg.", "gr.", "кг.", "км.",
        "som.", "сом.", "тг.", "руб.", "тнг.", "manat.",
        # months (Latin)
        "oca.", "şub.", "mar.", "nis.", "may.", "haz.", "tem.", "ağu.", "eyl.",
        "eki.", "kas.", "ara.", "yan.", "fev.", "apr.", "iyn.", "iyl.", "avg.",
        "sen.", "sent.", "okt.", "noy.", "dek.", "yanv.", "fevr.", "avq.",
        "sentyabr.", "jan.", "feb.", "aug.", "sep.", "oct.", "nov.", "dec.",
        # months (Cyrillic)
        "янв.", "фев.", "февр.", "мар.", "апр.", "июн.", "июл.", "авг.",
        "сен.", "сент.", "окт.", "ноя.", "нояб.", "дек.", "қаң.", "ақп.",
        "нау.", "сәу.", "мам.", "мау.", "шіл.", "там.", "қыр.", "қаз.",
        "қар.", "жел.",
    }
)

# Characters split off the *start* of a whitespace-delimited chunk.
OPENING_PUNCT = "([{\"'«»„“”‘’‚‹›¿¡"
# Characters split off the *end* of a chunk (plus "." which is special-cased).
CLOSING_PUNCT = ")]}\"'«»“”‘’‹›,:;!?…"
# Characters that close a quotation/parenthesis after sentence-final punct.
SENT_CLOSERS = frozenset(")]}\"'»”’›“")
SENT_FINAL = frozenset(".!?…")
# Apostrophe-like characters that are letters in some orthographies when they
# follow o/g (Uzbek o‘ g‘, Karakalpak, ...).
_APOSTROPHES = "'‘’ʼ`"
_APOSTROPHE_LETTER_LANGS = frozenset({"uzb", "kaa"})

_MULTI_DOT_ABBR = re.compile(r"^(?:[^\W\d_]{1,4}\.){2,}$")
_ORDINAL = re.compile(r"^\d+\.$")


def _lower(text: str) -> str:
    """Lower-case with Turkish dotted capital I handled sanely."""
    return text.replace("İ", "i").replace("I", "i").lower()


def _is_punct(text: str) -> bool:
    return bool(text) and all(unicodedata.category(ch).startswith("P") for ch in text)


@dataclass
class _Tok:
    """A token located in the original text."""

    text: str
    start: int
    end: int
    punct: bool = False
    abbr: Optional[str] = None  # "title" / "terminal" / None
    form: Optional[str] = None  # normalized form (defaults to text)


def _starts_upper(text: str) -> bool:
    for ch in text:
        if ch.isalpha():
            return ch.isupper()
        if ch.isdigit():
            return False
    return False


def _starts_lower(text: str) -> bool:
    for ch in text:
        if ch.isalpha():
            return ch.islower()
        if ch.isdigit():
            return False
    return False


def _is_sentence_final(tok: _Tok) -> bool:
    return tok.punct and all(ch in SENT_FINAL or ch in "؟۔" for ch in tok.text)


def _is_ellipsis(tok: _Tok) -> bool:
    return tok.text == "…" or (len(tok.text) > 1 and set(tok.text) == {"."})


def split_sentences(toks: list[_Tok], case_sensitive: bool = True) -> list[list[_Tok]]:
    """Group located tokens into sentences.

    Args:
        toks: Tokens in text order.
        case_sensitive: Whether the script distinguishes letter case (used for
            ellipses and terminal abbreviations; Arabic script does not).
    """
    sentences: list[list[_Tok]] = []
    current: list[_Tok] = []
    i = 0
    n = len(toks)
    while i < n:
        tok = toks[i]
        current.append(tok)
        i += 1
        nxt = toks[i] if i < n else None
        end_here = False

        if nxt is None:
            end_here = True
        elif _is_sentence_final(tok):
            if _is_sentence_final(nxt) and nxt.start == tok.end:
                end_here = False  # "?!" / "!!" -> wait for the last one
            elif _is_ellipsis(tok) and case_sensitive:
                end_here = _starts_upper(nxt.text) or (
                    nxt.punct and nxt.text[0] in OPENING_PUNCT + "-–—"
                )
            else:
                end_here = True
        elif tok.abbr == "terminal":
            end_here = case_sensitive and _starts_upper(nxt.text)

        if end_here and nxt is not None:
            # Absorb closing quotes/brackets attached to the final punctuation.
            j = i
            last = tok
            while j < n and toks[j].punct and toks[j].start == last.end and all(
                ch in SENT_CLOSERS for ch in toks[j].text
            ):
                last = toks[j]
                j += 1
            follower = toks[j] if j < n else None
            if (
                case_sensitive
                and follower is not None
                and _starts_lower(follower.text)
            ):
                # «Salom!» dedi. / "Geldim." diye -> the sentence continues.
                continue
            current.extend(toks[i:j])
            i = j
            sentences.append(current)
            current = []
        elif end_here:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def build_sentences(doc: Document, sentences: Iterable[list[_Tok]]) -> None:
    """Append :class:`Sentence` objects built from located tokens to ``doc``."""
    text = doc.text
    for toks in sentences:
        if not toks:
            continue
        sentence = Sentence(text=text[toks[0].start : toks[-1].end])
        for word_id, tok in enumerate(toks, start=1):
            form = tok.form if tok.form is not None else tok.text
            word = Word(
                id=word_id,
                text=form,
                upos="PUNCT" if tok.punct else None,
                lemma=form if tok.punct else None,
                start_char=tok.start,
                end_char=tok.end,
            )
            token = Token(
                id=(word_id,),
                text=form,
                words=[word],
                start_char=tok.start,
                end_char=tok.end,
            )
            sentence.tokens.append(token)
            sentence.words.append(word)
        doc.sentences.append(sentence)


class RegexTokenizer(Processor):
    """Rule-based tokenizer for Latin/Cyrillic Turkic languages.

    Most Turkic languages are space-delimited with predictable punctuation.
    This handles the majority of cases:

    * opening/closing brackets and quotes (including ``«»`` and ``„“``) are
      split from words;
    * ``...`` and ``…`` are single tokens;
    * abbreviations (``Prof.``, ``т.б.``, ``va b.``, initials such as ``A.``)
      keep their period and do not end a sentence;
    * ordinal numbers (``1. sınıf``) keep their period and do not end a
      sentence;
    * decimals and word-internal apostrophes/hyphens are preserved.
    """

    NAME = "tokenize"
    PROVIDES = ["tokenize"]
    REQUIRES = []

    CHUNK = re.compile(r"\S+")

    def __init__(
        self,
        lang: str,
        script: Optional[object] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(lang, script, config)
        self._abbreviations: set[str] = set()
        self._title_abbreviations: set[str] = set()

    def load(self, model_path: Optional[str] = None) -> None:
        """Load abbreviation lists used to avoid false sentence splits.

        Extra abbreviations can be passed via ``config["abbreviations"]``
        (treated as terminal-capable, like "etc.").
        """
        extra = {_lower(a if a.endswith(".") else a + ".") for a in self.config.get("abbreviations", [])}
        self._title_abbreviations = {_lower(a) for a in TITLE_ABBREVIATIONS}
        self._abbreviations = {_lower(a) for a in TERMINAL_ABBREVIATIONS} | extra
        self._loaded = True

    # ------------------------------------------------------------------
    def _abbreviation_type(
        self, core: str, prev: Optional[str], next_chunk: Optional[str]
    ) -> Optional[str]:
        """Classify ``core`` (ending in '.') as an abbreviation, or None."""
        if not self._title_abbreviations and not self._abbreviations:
            self.load()
        low = _lower(core)
        if low in self._title_abbreviations:
            return "title"
        if _MULTI_DOT_ABBR.match(core):
            return "terminal"
        if low in self._abbreviations or (
            prev is not None and f"{_lower(prev)} {low}" in self._abbreviations
        ):
            # Single-word "etc."/unit/month abbreviations are often ordinary
            # words too ("som.", "ara."): keep the period only when the text
            # continues with a lower-case word or a number.
            follow = (next_chunk or "").lstrip(OPENING_PUNCT)
            if follow and not _starts_upper(follow):
                return "terminal"
            return None
        # Single capital initial followed by a capitalised word: "A. Navoiy".
        if (
            len(core) == 2
            and core[0].isalpha()
            and core[0].isupper()
            and next_chunk is not None
            and _starts_upper(next_chunk.lstrip(OPENING_PUNCT))
        ):
            return "title"
        return None

    def _split_chunk(
        self,
        chunk: str,
        start: int,
        prev: Optional[str],
        next_chunk: Optional[str],
        at_start: bool = False,
    ) -> list[_Tok]:
        """Split one whitespace-delimited chunk into located tokens."""
        out: list[_Tok] = []
        i, j = 0, len(chunk)

        # Leading opening punctuation.
        while i < j and chunk[i] in OPENING_PUNCT:
            out.append(_Tok(chunk[i], start + i, start + i + 1, punct=True))
            i += 1

        trailing: list[_Tok] = []
        abbr: Optional[str] = None
        while j > i:
            ch = chunk[j - 1]
            if ch == ".":
                k = j
                while k > i and chunk[k - 1] == ".":
                    k -= 1
                if j - k >= 2:  # "..." -> one token
                    trailing.append(_Tok(chunk[k:j], start + k, start + j, punct=True))
                    j = k
                    continue
                core = chunk[i:j]
                if len(core) > 1:
                    abbr = self._abbreviation_type(core, prev, next_chunk)
                    if abbr is None and _ORDINAL.match(core) and not trailing:
                        # "1. Birinchi", "2. Dünya Savaşı", "1990. yılda":
                        # an ordinal, unless it looks like a number (e.g. a
                        # year) ending a sentence before a capitalised word.
                        follow = (next_chunk or "").lstrip(OPENING_PUNCT)
                        if follow and follow[0].isalnum() and (
                            not _starts_upper(follow) or at_start or len(core) <= 4
                        ):
                            abbr = "ordinal"
                    if abbr is not None:
                        break
                trailing.append(_Tok(".", start + j - 1, start + j, punct=True))
                j -= 1
            elif ch in CLOSING_PUNCT:
                if (
                    ch in _APOSTROPHES
                    and self.lang in _APOSTROPHE_LETTER_LANGS
                    and j - 1 > i
                    and chunk[j - 2] in "oOgG"
                ):
                    break
                trailing.append(_Tok(ch, start + j - 1, start + j, punct=True))
                j -= 1
            else:
                break

        if j > i:
            core = chunk[i:j]
            out.append(
                _Tok(
                    core,
                    start + i,
                    start + j,
                    punct=_is_punct(core) and abbr is None,
                    abbr=abbr if abbr in ("title", "terminal") else None,
                )
            )
        out.extend(reversed(trailing))
        return out

    def tokenize(self, text: str) -> list[_Tok]:
        """Return located tokens for ``text`` (offsets index ``text``)."""
        chunks = [(m.group(), m.start()) for m in self.CHUNK.finditer(text)]
        toks: list[_Tok] = []
        prev: Optional[str] = None
        for idx, (chunk, start) in enumerate(chunks):
            next_chunk = chunks[idx + 1][0] if idx + 1 < len(chunks) else None
            at_start = not toks or _is_sentence_final(toks[-1]) or (
                toks[-1].punct and all(ch in SENT_CLOSERS for ch in toks[-1].text)
            )
            pieces = self._split_chunk(chunk, start, prev, next_chunk, at_start)
            toks.extend(pieces)
            words = [p.text for p in pieces if not p.punct]
            prev = words[-1] if words else None
        return toks

    def process(self, doc: Document) -> Document:
        """Split text into sentences and tokens."""
        toks = self.tokenize(doc.text or "")
        build_sentences(doc, split_sentences(toks, case_sensitive=True))
        doc._processor_log.append("tokenize:regex")
        return doc


class NeuralTokenizer(Processor):
    """Neural tokenizer based on a character-level BiLSTM.

    For languages where rule-based splitting is insufficient.
    Architecture follows Stanza's tokenizer.
    """

    IMPLEMENTED = False  # placeholder, see ROADMAP.md
    NAME = "tokenize"
    PROVIDES = ["tokenize"]
    REQUIRES = []

    def load(self, model_path: str) -> None:
        """Load trained character-level BiLSTM tokenizer model."""
        raise NotImplementedError("NeuralTokenizer.load not yet implemented.")

    def process(self, doc: Document) -> Document:
        """Neural sentence splitting and tokenization."""
        raise NotImplementedError("NeuralTokenizer.process not yet implemented.")
