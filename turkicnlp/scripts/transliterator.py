"""
Bidirectional script transliteration for Turkic languages.

Each language has its own mapping because the same Cyrillic letter
maps to different Latin letters across languages. Transliteration
tables are defined in :data:`TRANSLITERATION_TABLES`.
"""

from __future__ import annotations

import unicodedata

from turkicnlp.scripts import Script


class Transliterator:
    """Convert text between writing scripts for a given Turkic language.

    Args:
        lang: ISO 639-3 language code.
        source: Source :class:`Script`.
        target: Target :class:`Script`.

    Raises:
        ValueError: If no transliteration table exists for the given combination.
    """

    def __init__(self, lang: str, source: Script, target: Script) -> None:
        self.lang = lang
        self.source = source
        self.target = target
        self._forward_map, self._reverse_map = self._load_mapping(lang, source, target)

    # Placeholders (Unicode private-use area) inserted by context rules before
    # table lookup and replaced afterwards. They never occur in real text.
    _PUA_OUT: dict[str, str] = {
        "\ue000": "е", "\ue001": "Е",    # Cyrillic e (from Latin ye / e)
        "\ue002": "э", "\ue003": "Э",    # Cyrillic э (word-initial Latin e)
        "\ue004": "й", "\ue005": "Й",    # Kazakh i after a vowel
        "\ue006": "ё", "\ue007": "Ё",    # Crimean Tatar ö after a consonant
        "\ue008": "ю", "\ue009": "Ю",    # Crimean Tatar ü after a consonant
        "\ue00a": "ye", "\ue00b": "Ye",  # Tatar е -> ye
        "\ue00c": "ö", "\ue00d": "Ö",    # Crimean Tatar ё after a consonant
        "\ue00e": "ü", "\ue00f": "Ü",    # Crimean Tatar ю after a consonant
        "\ue010": "ýe", "\ue011": "Ýe",  # Turkmen е -> ýe
    }
    # Uppercase placeholders (odd code points) - used for all-caps detection.
    _PUA_UPPER = frozenset(ch for ch in _PUA_OUT if ord(ch) % 2 == 1)
    _LAT_VOWELS = set("aeiouäöüıəáóúíéâAEIOUÄÖÜIƏÁÓÚÍÉÂİ")
    _CYR_VOWELS = set("аеёиоуыэюяәөүіұӑӗӳАЕЁИОУЫЭЮЯӘӨҮІҰӐӖӲ")
    _CYR_SIGNS = set("ъьЪЬ")
    # Uzbek: Cyrillic е after these (or word-initially) is read "ye".
    _UZB_YE_PREV = _CYR_VOWELS | _CYR_SIGNS | set("йЙўЎ")

    @staticmethod
    def _boundary(text: str, i: int) -> bool:
        return i == 0 or not text[i - 1].isalpha()

    def _latn_e_pre(self, text: str, ye: bool) -> str:
        """Latin -> Cyrillic: word-initial / post-vocalic ``e`` -> ``э`` and
        (if *ye*) ``ye`` -> ``е`` (Uzbek, Tatar, Turkmen)."""
        out: list[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            ctx = self._boundary(text, i) or text[i - 1] in self._LAT_VOWELS
            if ye and ctx and ch in "yY" and i + 1 < len(text) and text[i + 1] in "eE":
                out.append("\ue001" if ch == "Y" else "\ue000")
                i += 2
                continue
            if ctx and ch in "eE":
                out.append("\ue003" if ch == "E" else "\ue002")
            else:
                out.append(ch)
            i += 1
        return "".join(out)

    def _cyrl_ye_pre(self, text: str, lower: str, upper: str, prev_set: set[str]) -> str:
        """Cyrillic -> Latin/CTS: ``е`` word-initially or after a character in
        *prev_set* becomes the placeholder *lower* / *upper* (which expand to a
        ``ye``-type sequence after table lookup)."""
        out: list[str] = []
        for i, ch in enumerate(text):
            if ch in "еЕ" and (self._boundary(text, i) or text[i - 1] in prev_set):
                out.append(upper if ch == "Е" else lower)
            else:
                out.append(ch)
        return "".join(out)

    def _tat_cyrl_pre(self, text: str) -> str:
        """Tatar Cyrillic -> Latin: ``е`` word-initially / after a vowel -> ``ye``."""
        return self._cyrl_ye_pre(text, "", "", self._CYR_VOWELS | self._CYR_SIGNS)

    def _kaz_latn_pre(self, text: str) -> str:
        """Kazakh 2021 Latin -> Cyrillic: ``i`` after a vowel is ``й`` (unless it
        starts io/iu/ia = ё/ю/я)."""
        out: list[str] = []
        for i, ch in enumerate(text):
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if (ch in "iİ" and i > 0 and text[i - 1] in self._LAT_VOWELS
                    and nxt.lower() not in ("a", "o", "u")):
                out.append("" if ch == "İ" else "")
            else:
                out.append(ch)
        return "".join(out)

    def _crh_latn_pre(self, text: str) -> str:
        """Crimean Tatar Latin -> Cyrillic: ö/ü after a consonant -> ё/ю."""
        out: list[str] = []
        for i, ch in enumerate(text):
            prev = text[i - 1] if i > 0 else ""
            after_cons = prev.isalpha() and prev not in self._LAT_VOWELS and prev not in "yY"
            if ch in "öÖüÜ" and after_cons:
                out.append({"ö": "", "Ö": "", "ü": "", "Ü": ""}[ch])
            else:
                out.append(ch)
        return "".join(out)

    def _crh_cyrl_pre(self, text: str) -> str:
        """Crimean Tatar Cyrillic -> Latin: ё/ю after a consonant -> ö/ü."""
        out: list[str] = []
        for i, ch in enumerate(text):
            prev = text[i - 1] if i > 0 else ""
            after_cons = (prev.isalpha() and prev not in self._CYR_VOWELS
                          and prev not in self._CYR_SIGNS)
            if ch in "ёЁюЮ" and after_cons:
                out.append({"ё": "", "Ё": "", "ю": "", "Ю": ""}[ch])
            else:
                out.append(ch)
        return "".join(out)

    def transliterate(self, text: str) -> str:
        """Convert *text* from *source* script to *target* script.

        Applies language-specific context rules, then greedy longest-match
        on the forward mapping table.
        """
        key = (self.lang, self.source, self.target)
        L, C, T = Script.LATIN, Script.CYRILLIC, Script.COMMON_TURKIC
        if key in {("uzb", L, C), ("tat", L, C)}:
            if self.lang == "uzb":
                text = self._normalize_uzb_apostrophes(text)
            text = self._latn_e_pre(text, ye=True)
        elif key == ("uzb", L, T):
            text = self._normalize_uzb_apostrophes(text)
        elif key == ("tuk", L, C):
            text = self._latn_e_pre(text, ye=False)
        elif key == ("tat", C, L):
            text = self._tat_cyrl_pre(text)
        elif key in {("uzb", C, L), ("uzb", C, T)}:
            # Uzbek: е word-initially / after a vowel or sign -> ye
            text = self._cyrl_ye_pre(text, "", "", self._UZB_YE_PREV)
        elif key == ("tuk", C, L):
            # Turkmen: е word-initially / after a vowel or sign -> ýe
            text = self._cyrl_ye_pre(
                text, "", "", self._CYR_VOWELS | self._CYR_SIGNS
            )
        elif key == ("tuk", C, T):
            text = self._cyrl_ye_pre(
                text, "", "", self._CYR_VOWELS | self._CYR_SIGNS
            )
        elif key == ("kaz", L, C):
            text = self._kaz_latn_pre(text)
        elif key == ("crh", L, C):
            text = self._crh_latn_pre(text)
        elif key == ("crh", C, L):
            text = self._crh_cyrl_pre(text)
        out = self._transliterate_core(text)
        if any(ch in self._PUA_OUT for ch in out):
            out = "".join(self._PUA_OUT.get(ch, ch) for ch in out)
        return out

    # ------------------------------------------------------------------
    # Letter-case handling (shared by every table-driven conversion)
    # ------------------------------------------------------------------

    _APOSTROPHES = frozenset("'‘’ʻʼʹ`")
    # Target orthographies whose capital of ``i`` is ``İ`` (dotted).
    _DOTTED_I_LATIN = frozenset({"tur", "aze", "kaz", "tat", "crh", "gag", "klj", "ota"})

    def _caps_mask(self, text: str) -> list[bool]:
        """Per-position flag: is the character inside an all-caps word?

        A word is a maximal run of letters (plus internal apostrophes and the
        context placeholders). It counts as all-caps when it has at least two
        letters and every cased letter is uppercase.
        """
        n = len(text)
        mask = [False] * n
        i = 0
        while i < n:
            ch = text[i]
            if not (ch.isalpha() or ch in self._PUA_OUT):
                i += 1
                continue
            j = i
            while j < n and (
                text[j].isalpha() or text[j] in self._PUA_OUT or text[j] in self._APOSTROPHES
            ):
                j += 1
            letters = 0
            upper = lower = 0
            for c in text[i:j]:
                if c in self._PUA_OUT:
                    letters += 1
                    if c in self._PUA_UPPER:
                        upper += 1
                    else:
                        lower += 1
                elif c.isalpha():
                    letters += 1
                    if c.isupper():
                        upper += 1
                    elif c.islower():
                        lower += 1
            if letters >= 2 and upper and not lower:
                for k in range(i, j):
                    mask[k] = True
            i = j
        return mask

    def _upper(self, s: str) -> str:
        """Uppercase *s* using the target orthography's rule for ``i``."""
        if "i" in s and (
            (self.target == Script.COMMON_TURKIC and self.lang != "uig")
            or (self.target == Script.LATIN and self.lang in self._DOTTED_I_LATIN)
        ):
            s = s.replace("i", "İ")
        return s.upper()

    def _apply_case(self, chunk: str, mapped: str, all_caps: bool, exact: bool) -> str:
        """Give *mapped* (the conversion of source *chunk*) the right case.

        * source word is all-caps -> uppercase the whole output
          (``ШАҲАР`` -> ``SHAHAR``, ``ЩУКА`` -> ``ŞTŞUKA``);
        * otherwise, if the chunk was found only via its lowercase form and
          starts with a capital -> capitalise the first output character.
        """
        if not mapped:
            return mapped
        if all_caps:
            return self._upper(mapped)
        if not exact and chunk[:1].isupper():
            return self._upper(mapped[0]) + mapped[1:]
        return mapped

    def _transliterate_core(self, text: str) -> str:
        """Greedy longest-match conversion (plus language-specific handlers).

        Args:
            text: Input text in the source script.

        Returns:
            Text converted to the target script.
        """
        if self.lang == "uig":
            handler = self._uig_handler()
            if handler is not None:
                return handler(text)
        if (
            self.lang == "azb"
            and self.source == Script.PERSO_ARABIC
            and self.target == Script.COMMON_TURKIC
        ):
            return self._transliterate_azb_arab_to_cts(text)

        fmap = self._forward_map
        caps = self._caps_mask(text)
        result: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            for length in range(min(4, n - i), 0, -1):
                chunk = text[i : i + length]
                if chunk in fmap:
                    mapped, exact = fmap[chunk], True
                else:
                    low = chunk.lower()
                    if low not in fmap:
                        continue
                    mapped, exact = fmap[low], False
                result.append(self._apply_case(chunk, mapped, caps[i], exact))
                i += length
                break
            else:
                ch = text[i]
                if ch in self._PUA_OUT:
                    result.append(self._apply_case(ch, self._PUA_OUT[ch], caps[i], True))
                else:
                    result.append(ch)
                i += 1
        return "".join(result)

    @staticmethod
    def _normalize_uzb_apostrophes(text: str) -> str:
        """Normalize common Uzbek apostrophe variants to ASCII apostrophe."""
        return (
            text.replace("‘", "'")
            .replace("’", "'")
            .replace("ʻ", "'")
            .replace("ʼ", "'")
            .replace("ʹ", "'")
            .replace("`", "'")
        )

    # ------------------------------------------------------------------
    # South Azerbaijani (Perso-Arabic) -> CTS
    # ------------------------------------------------------------------

    _AZB_FULL_VOWELS = frozenset("ۇۆؤۉؽ")          # always vowels (u ü ö ü ı)
    _AZB_YV = frozenset("ویيى")                     # v/o and y/i (context)
    _AZB_SEAT_NEXT = frozenset("یيىؽۇۆؤۉاآ")        # ئ before these is a seat
    _AZB_VOWEL_LETTERS = frozenset("اآوۇۆؤۉیيىؽئ")
    _AZB_DIACRITICS = {
        "َ": "ä", "ِ": "i", "ُ": "u",   # fatha (ə), kasra, damma
        "ْ": "", "ً": "", "ٌ": "", "ٍ": "",
        "ـ": "", "‌": "", "‍": "",        # tatweel, ZWNJ, ZWJ
    }

    @staticmethod
    def _arab_word_char(ch: str) -> bool:
        import unicodedata
        return bool(ch) and unicodedata.category(ch)[0] in "LM"

    def _transliterate_azb_arab_to_cts(self, text: str) -> str:
        """South Azerbaijani Arabic script -> CTS with vowel-letter heuristics.

        The Arabic script under-specifies vowels, so this is best effort:
        ``ۇ ۆ ؤ ۉ ؽ`` are always vowels (u ü ö ü ı); ``و`` / ``ی`` are ``v`` /
        ``y`` word-initially or next to a vowel letter and ``o`` / ``i``
        otherwise; word-initial ``ا`` before a vowel letter and ``ئ`` before
        a vowel letter are silent seats, other ``ئ`` is ``e``; word-final
        ``ه`` is ``ä`` (ə); short-vowel diacritics are kept (fatha -> ``ä``).
        """
        text = text.translate(_ARAB_PUNCT_TO_LATIN)
        fmap = self._forward_map
        word = self._arab_word_char
        out: list[str] = []
        n = len(text)
        for i, ch in enumerate(text):
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < n else ""
            initial = not word(prev)
            final = not word(nxt)
            seat_before = (prev == "ئ" and ch in self._AZB_SEAT_NEXT) or (
                prev == "ا" and (i < 2 or not word(text[i - 2]))
            )
            if ch in self._AZB_DIACRITICS:
                out.append(self._AZB_DIACRITICS[ch])
            elif ch == "ّ":                     # shadda: geminate
                if out and out[-1]:
                    out.append(out[-1][-1])
            elif ch == "ا":
                vowel_next = nxt in self._AZB_VOWEL_LETTERS and nxt not in "اآ"
                out.append("" if initial and vowel_next else "a")
            elif ch == "آ":
                out.append("a")
            elif ch == "ئ":
                out.append("" if nxt in self._AZB_SEAT_NEXT else "e")
            elif ch == "ؤ":
                out.append("ö")
            elif ch == "ۇ":
                out.append("u")
            elif ch in "ۆۉ":
                out.append("ü")
            elif ch == "ؽ":
                out.append("ı")
            elif ch in self._AZB_YV:
                cons = "v" if ch == "و" else "y"
                vow = "o" if ch == "و" else "i"
                if seat_before:
                    out.append(vow)
                elif initial or nxt in self._AZB_VOWEL_LETTERS or (
                    prev in self._AZB_VOWEL_LETTERS and not final
                ):
                    out.append(cons)
                elif prev in self._AZB_VOWEL_LETTERS and final:
                    out.append(cons)
                else:
                    out.append(vow)
            elif ch == "ه":
                out.append("ä" if final and not initial else "h")
            elif ch in fmap:
                out.append(fmap[ch])
            else:
                out.append(ch)
        return "".join(out)

    # ------------------------------------------------------------------
    # Uyghur dispatch: case preservation + Arabic punctuation
    # ------------------------------------------------------------------

    def _uig_handler(self):
        """Return the Uyghur special-case converter for this script pair, or
        ``None`` if the plain table-driven conversion applies.

        Wraps the lowercase-only conversion chains so that
        * input going *to* Arabic is lowercased first and ``, ; ?`` become
          ``، ؛ ؟``;
        * input coming *from* Arabic has ``، ؛ ؟`` turned into ``, ; ?``;
        * conversions between cased scripts (Latin / Cyrillic / CTS) keep the
          capitalisation of each word (lowercase / Title / ALL CAPS).
        """
        A, L, C, T = (Script.PERSO_ARABIC, Script.LATIN, Script.CYRILLIC,
                      Script.COMMON_TURKIC)
        handlers = {
            (L, A): self._transliterate_uig_latn_to_arab,
            (A, L): self._transliterate_uig_arab_to_latn,
            (A, C): self._transliterate_uig_arab_to_cyrl,
            (C, A): self._transliterate_uig_cyrl_to_arab,
            (C, L): self._transliterate_uig_cyrl_to_latn,
            (L, C): self._transliterate_uig_latn_to_cyrl,
            (A, T): self._transliterate_uig_arab_to_cts,
            (T, A): self._transliterate_uig_cts_to_arab,
            (L, T): self._transliterate_uig_latn_to_cts,
            (T, L): self._transliterate_uig_cts_to_latn,
        }
        fn = handlers.get((self.source, self.target))
        if fn is None:
            return None
        if self.target == A:
            return lambda t: fn(t.lower()).translate(_LATIN_PUNCT_TO_ARAB)
        if self.source == A:
            return lambda t: fn(t.translate(_ARAB_PUNCT_TO_LATIN))
        return lambda t: self._case_preserving(t, fn)

    _WORD_RE = None

    @classmethod
    def _case_preserving(cls, text: str, fn) -> str:
        """Apply lowercase-only converter *fn* word by word, restoring each
        word's case pattern (ALL CAPS, Title or lower) on the output."""
        import re
        if cls._WORD_RE is None:
            cls._WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")

        def repl(m):
            word = m.group()
            out = fn(word.lower())
            letters = [c for c in word if c.isalpha()]
            if len(letters) >= 2 and all(c.isupper() for c in letters if c.isupper() or c.islower()):
                return out.upper()
            if word[0].isupper():
                return out[:1].upper() + out[1:]
            return out

        return cls._WORD_RE.sub(repl, text)

    def _transliterate_uig_latn_to_arab(self, text: str) -> str:
        """Uyghur Latin (ULS) → Arabic via CTS pivot with full ئ insertion.

        Follows UMSC ULS2UAS logic: ULS→CTS chain, insert ئ before any vowel
        not preceded by a consonant, replace CTS→UAS, strip apostrophes, revise.
        This handles both word-initial vowels AND vowel-after-vowel sequences
        (e.g. 'radio' → رادىئو) and apostrophe syllable markers (e.g. "inik'ana").
        """
        import regex as re
        text = text.lower()
        text = self._uig_uls_to_cts(text)
        consonants = "bptcçxdrzjsşfñlmhvyqkgnğ"
        text = re.sub(
            rf"(?<=[^{consonants}]|^)[aeéiouöü]",
            lambda m: "\u0626" + m.group(),
            text,
        )
        text = self._uig_replace(text, self._UIG_CTS, self._UIG_UAS)
        text = text.replace("'", "")
        return self._uig_revise_uas(text)

    # ------------------------------------------------------------------
    # Uyghur multi-script helpers
    # Conversion tables and chain logic adapted from:
    # https://github.com/neouyghur/ScriptConverter4Uyghur (Apache-2.0)
    # ------------------------------------------------------------------

    # Positional tables: UAS / UCS / CTS share the same index positions.
    _UIG_UAS: list[str] = [
        "ا", "ە", "ب", "پ", "ت", "ج", "چ", "خ", "د", "ر",
        "ز", "ژ", "س", "ش", "ف", "ڭ", "لا", "ل", "م", "ھ",
        "و", "ۇ", "ۆ", "ۈ", "ۋ", "ې", "ى", "ي", "ق", "ك",
        "گ", "ن", "غ",
    ]
    _UIG_CTS: list[str] = [
        "a", "e", "b", "p", "t", "c", "ç", "x", "d", "r",
        "z", "j", "s", "ş", "f", "ñ", "la", "l", "m", "h",
        "o", "u", "ö", "ü", "v", "é", "i", "y", "q", "k",
        "g", "n", "ğ",
    ]
    _UIG_UCS: list[str] = [
        "а", "ә", "б", "п", "т", "җ", "ч", "х", "д", "р",
        "з", "ж", "с", "ш", "ф", "ң", "ла", "л", "м", "һ",
        "о", "у", "ө", "ү", "в", "е", "и", "й", "қ", "к",
        "г", "н", "ғ",
    ]
    _UIG_CTS_CONSONANTS: str = "bptcçxdrzjsşfñlmhvyqkgnğ"

    @staticmethod
    def _uig_replace(text: str, src: list[str], tgt: list[str]) -> str:
        for s, t in zip(src, tgt):
            text = text.replace(s, t)
        return text

    @staticmethod
    def _uig_revise_cts(text: str) -> str:
        """Remove word-initial ئ; remove post-vowel ئ; replace remaining ئ with apostrophe."""
        import regex as re
        text = re.sub(r"(?<=[^aeuoöübptcçxdrzjsşfñlmhvéiyqkgnğ]|^)\u0626", "", text)
        text = re.sub(r"(([aeéiouöü])\u0626)", lambda m: m.group()[0], text)
        text = text.replace("\u0626", "'")
        return text

    @staticmethod
    def _uig_revise_cts_keep_apos(text: str) -> str:
        """Remove word-initial ئ; keep post-vowel ئ as apostrophe (for Latin/Cyrillic output)."""
        import regex as re
        text = re.sub(r"(?<=[^aeuoöübptcçxdrzjsşfñlmhvéiyqkgnğ]|^)\u0626", "", text)
        # Unlike _uig_revise_cts, we do NOT strip vowel+ئ here — it becomes an apostrophe
        text = text.replace("\u0626", "'")
        return text

    @staticmethod
    def _uig_revise_uas(text: str) -> str:
        """Fix consecutive Arabic vowels by inserting ئ between them."""
        import regex as re
        return re.sub(
            r"(^|-|\s|[اەېىوۇۆۈ])([اەېىوۇۆۈ])",
            lambda m: m.group(1) + "ئ" + m.group(2),
            text,
        )

    @staticmethod
    def _uig_uls_to_cts(text: str) -> str:
        """ULS (Latin) → CTS chain."""
        return (
            text.replace("j", "c")
            .replace("ng", "ñ")
            .replace("n'g", "ng")
            .replace("'ng", "ñ")
            .replace("ch", "ç")
            .replace("zh", "j")
            .replace("sh", "ş")
            .replace("'gh", "ğ")
            .replace("gh", "ğ")
            .replace("w", "v")
        )

    @staticmethod
    def _uig_cts_to_uls(text: str) -> str:
        """CTS → ULS (Latin) chain."""
        return (
            text.replace("ng", "n'g")
            .replace("sh", "s'h")
            .replace("ch", "c'h")
            .replace("zh", "z'h")
            .replace("gh", "g'h")
            .replace("nğ", "n'gh")
            .replace("ñ", "ng")
            .replace("j", "zh")
            .replace("c", "j")
            .replace("ç", "ch")
            .replace("ş", "sh")
            .replace("ğ", "gh")
            .replace("v", "w")
        )

    def _transliterate_uig_arab_to_cts(self, text: str) -> str:
        text = self._uig_replace(text, self._UIG_UAS, self._UIG_CTS)
        return self._uig_revise_cts(text)

    def _transliterate_uig_arab_to_latn(self, text: str) -> str:
        """Arab → CTS (preserving inter-vowel ئ as apostrophe) → ULS chain."""
        text = self._uig_replace(text, self._UIG_UAS, self._UIG_CTS)
        text = self._uig_revise_cts_keep_apos(text)
        return self._uig_cts_to_uls(text.lower())

    def _transliterate_uig_cts_to_arab(self, text: str) -> str:
        import regex as re
        vowels = "aeéiouöü"
        text = re.sub(
            rf"(?<=[^{self._UIG_CTS_CONSONANTS}]|^)[{vowels}]",
            lambda m: "\u0626" + m.group(),
            text,
        )
        text = self._uig_replace(text, self._UIG_CTS, self._UIG_UAS)
        text = text.replace("'", "")
        return self._uig_revise_uas(text)

    def _transliterate_uig_arab_to_cyrl(self, text: str) -> str:
        text = self._uig_replace(text, self._UIG_UAS, self._UIG_CTS)
        text = self._uig_revise_cts_keep_apos(text)
        text = text.replace("ya", "я").replace("yu", "ю")
        text = self._uig_replace(text, self._UIG_CTS, self._UIG_UCS)
        return text

    def _transliterate_uig_cyrl_to_arab(self, text: str) -> str:
        import regex as re
        text = self._uig_replace(text, self._UIG_UCS, self._UIG_CTS)
        text = text.replace("я", "ya").replace("ю", "yu")
        vowels = "aeéiouöü"
        text = re.sub(
            rf"(?<=[^{self._UIG_CTS_CONSONANTS}]|^)[{vowels}]",
            lambda m: "\u0626" + m.group(),
            text,
        )
        text = self._uig_replace(text, self._UIG_CTS, self._UIG_UAS)
        text = text.replace("'", "")
        return self._uig_revise_uas(text)

    def _transliterate_uig_cyrl_to_latn(self, text: str) -> str:
        text = self._uig_replace(text, self._UIG_UCS, self._UIG_CTS)
        text = text.replace("я", "ya").replace("ю", "yu")
        return self._uig_cts_to_uls(text)

    def _transliterate_uig_latn_to_cyrl(self, text: str) -> str:
        text = text.lower()
        text = self._uig_uls_to_cts(text)
        text = text.replace("ya", "я").replace("yu", "ю")
        text = self._uig_replace(text, self._UIG_CTS, self._UIG_UCS)
        return text

    def _transliterate_uig_latn_to_cts(self, text: str) -> str:
        return self._uig_uls_to_cts(text.lower())

    def _transliterate_uig_cts_to_latn(self, text: str) -> str:
        return self._uig_cts_to_uls(text.lower())

    @staticmethod
    def _load_mapping(
        lang: str, source: Script, target: Script
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Load character mapping for a language and script pair."""
        key = f"{lang}_{source.value}_to_{target.value}"
        if key not in TRANSLITERATION_TABLES:
            raise ValueError(
                f"No transliteration table for {lang} {source} -> {target}. "
                f"Available: {list(TRANSLITERATION_TABLES.keys())}"
            )
        forward = TRANSLITERATION_TABLES[key]
        reverse = {v: k for k, v in forward.items()}
        return forward, reverse


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Arabic-script punctuation <-> Latin/Cyrillic punctuation.
_ARAB_PUNCT_TO_LATIN = str.maketrans({"\u060c": ",", "\u061b": ";", "\u061f": "?"})
_LATIN_PUNCT_TO_ARAB = str.maketrans({",": "\u060c", ";": "\u061b", "?": "\u061f"})

#: The 34 letters of the Common Turkic Alphabet (upper and lower case).
CTS_ALPHABET: frozenset[str] = frozenset(
    "AaÄäBbCcÇçDdEeFfGgĞğHhXxIıİiJjKkQqLlMmNnÑñOoÖöPpRrSsŞşTtUuÜüVvWwYyZz"
)
#: Letters this module additionally uses in CTS output: Uyghur closed é
#: (ې / е), which the Uyghur CTS conventions keep distinct from e (ە / ә).
CTS_EXTENSIONS: frozenset[str] = frozenset("Éé")


# Old Turkic (Orkhon / Yenisei) runes, keyed by the final word of the Unicode
# character name ("OLD TURKIC LETTER ORKHON AEB" -> "AEB"). Orkhon and Yenisei
# variants of a letter share one value. Turkological convention: back-/front-
# vowel consonant pairs (AB/AEB, AG/AEG, ...) both give the plain consonant;
# ORKHON O = o/u, OE = ö/ü; ligatures give their consonant cluster.
_OLD_TURKIC_VALUES: dict[str, str] = {
    # vowels
    "A": "a", "AE": "ä", "I": "ı", "E": "e", "O": "o", "OE": "ö",
    # consonants (back / front vowel variants)
    "AB": "b", "AEB": "b", "AG": "g", "AEG": "g", "AD": "d", "AED": "d",
    "EZ": "z", "AY": "y", "AEY": "y", "AEK": "k", "OEK": "k",
    "AL": "l", "AEL": "l", "EM": "m", "AN": "n", "AEN": "n",
    "ANG": "ŋ", "ENG": "ŋ", "AENG": "ŋ", "EP": "p",
    "IC": "ch", "EC": "ch", "AQ": "q", "IQ": "q", "OQ": "q",
    "AR": "r", "AER": "r", "AS": "s", "AES": "s",
    "ASH": "sh", "ESH": "sh", "AT": "t", "AET": "t",
    # ligatures / syllabic signs
    "ELT": "lt", "ENT": "nt", "ENC": "nch", "ENY": "ny",
    "OP": "op", "OT": "ot", "BASH": "bash",
}


def _build_old_turkic_table() -> dict[str, str]:
    """Build the Old Turkic -> Latin table from Unicode character names, so
    every rune is attached to its correct code point (U+10C00-U+10C48)."""
    table: dict[str, str] = {}
    for cp in range(0x10C00, 0x10C50):
        name = unicodedata.name(chr(cp), "")
        if not name.startswith("OLD TURKIC LETTER "):
            continue
        value = _OLD_TURKIC_VALUES.get(name.rsplit(" ", 1)[1])
        if value is not None:
            table[chr(cp)] = value
    # Word separator: inscriptions use a two-dot mark, encoded as ':' or
    # U+205A TWO DOT PUNCTUATION; normalise the latter to ':'. (':' itself is
    # passed through unchanged.)
    table["\u205a"] = ":"
    return table


# ---------------------------------------------------------------------------
# Transliteration tables
# ---------------------------------------------------------------------------

TRANSLITERATION_TABLES: dict[str, dict[str, str]] = {
    # Kazakh Cyrillic → Latin (2021 official Latin alphabet)
    "kaz_Cyrl_to_Latn": {
        # 2021 official alphabet: И/Й -> İ i, І -> I ı, Ы -> Y y, Ш -> Ş ş,
        # Ч -> Tş, Щ -> Ştş, Х/Һ -> H h, Ё/Ю/Я -> io/iu/ia
        "щ": "ştş", "Щ": "Ştş", "ч": "tş", "Ч": "Tş", "ш": "ş", "Ш": "Ş",
        "ц": "ts", "Ц": "Ts",
        "ё": "io", "Ё": "İo", "ю": "iu", "Ю": "İu", "я": "ia", "Я": "İa",
        "ә": "ä", "Ә": "Ä", "ғ": "ğ", "Ғ": "Ğ", "қ": "q", "Қ": "Q",
        "ң": "ñ", "Ң": "Ñ", "ө": "ö", "Ө": "Ö", "ұ": "ū", "Ұ": "Ū",
        "ү": "ü", "Ү": "Ü", "і": "ı", "І": "I", "һ": "h", "Һ": "H",
        "ж": "j", "Ж": "J",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ",
        "й": "i", "Й": "İ", "к": "k", "К": "K", "л": "l", "Л": "L",
        "м": "m", "М": "M", "н": "n", "Н": "N", "о": "o", "О": "O",
        "п": "p", "П": "P", "р": "r", "Р": "R", "с": "s", "С": "S",
        "т": "t", "Т": "T", "у": "u", "У": "U", "ф": "f", "Ф": "F",
        "х": "h", "Х": "H", "э": "e", "Э": "E",
        "ъ": "", "Ъ": "", "ь": "", "Ь": "", "ы": "y", "Ы": "Y",
    },

    # Kazakh Latin → Cyrillic
    "kaz_Latn_to_Cyrl": {
        # 2021 official alphabet (reverse). Ambiguities resolved by default:
        # i -> и (й after a vowel, see Transliterator._kaz_latn_pre), h -> х.
        "ştş": "щ", "Ştş": "Щ", "ŞTŞ": "Щ",
        "tş": "ч", "Tş": "Ч", "TŞ": "Ч",
        "ts": "ц", "Ts": "Ц", "TS": "Ц",
        "io": "ё", "İo": "Ё", "İO": "Ё",
        "iu": "ю", "İu": "Ю", "İU": "Ю",
        "ia": "я", "İa": "Я", "İA": "Я",
        "ş": "ш", "Ş": "Ш",
        "ä": "ә", "Ä": "Ә", "ğ": "ғ", "Ğ": "Ғ", "q": "қ", "Q": "Қ",
        "ñ": "ң", "Ñ": "Ң", "ö": "ө", "Ö": "Ө", "ū": "ұ", "Ū": "Ұ",
        "ü": "ү", "Ü": "Ү", "ı": "і", "I": "І", "i": "и", "İ": "И",
        "a": "а", "A": "А", "b": "б", "B": "Б", "v": "в", "V": "В",
        "g": "г", "G": "Г", "d": "д", "D": "Д", "e": "е", "E": "Е",
        "z": "з", "Z": "З", "j": "ж", "J": "Ж", "k": "к", "K": "К",
        "l": "л", "L": "Л", "m": "м", "M": "М", "n": "н", "N": "Н",
        "o": "о", "O": "О", "p": "п", "P": "П", "r": "р", "R": "Р",
        "s": "с", "S": "С", "t": "т", "T": "Т", "u": "у", "U": "У",
        "f": "ф", "F": "Ф", "h": "х", "H": "Х", "y": "ы", "Y": "Ы",
    },

    # Uzbek Cyrillic → Latin (1995 official Latin alphabet)
    "uzb_Cyrl_to_Latn": {
        # с + ҳ is written sʼh so that it is not read as sh (Исҳоқ -> Isʼhoq)
        "сҳ": "sʼh", "Сҳ": "Sʼh",
        "ш": "sh", "Ш": "Sh", "ч": "ch", "Ч": "Ch", "ғ": "g'", "Ғ": "G'",
        "қ": "q", "Қ": "Q", "ҳ": "h", "Ҳ": "H", "ў": "o'", "Ў": "O'",
        "нг": "ng", "Нг": "Ng", "ж": "j", "Ж": "J",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "з": "z", "З": "Z", "и": "i", "И": "I",
        "й": "y", "Й": "Y", "к": "k", "К": "K", "л": "l", "Л": "L",
        "м": "m", "М": "M", "н": "n", "Н": "N", "о": "o", "О": "O",
        "п": "p", "П": "P", "р": "r", "Р": "R", "с": "s", "С": "S",
        "т": "t", "Т": "T", "у": "u", "У": "U", "ф": "f", "Ф": "F",
        "х": "x", "Х": "X", "ц": "ts", "Ц": "Ts", "э": "e", "Э": "E",
        "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya", "ъ": "'", "ь": "",
    },

    # Uyghur Perso-Arabic → Latin (ULY)
    "uig_Arab_to_Latn": {
        "ئا": "a", "ئە": "e", "ئو": "o", "ئۇ": "u", "ئۆ": "ö", "ئۈ": "ü",
        "ئې": "é", "ئى": "i",
        "ا": "a", "ە": "e", "ب": "b", "پ": "p", "ت": "t", "ج": "j",
        "چ": "ch", "خ": "x", "د": "d", "ر": "r", "ز": "z", "ژ": "zh",
        "س": "s", "ش": "sh", "غ": "gh", "ف": "f", "ق": "q", "ك": "k",
        "گ": "g", "ڭ": "ng", "ل": "l", "م": "m", "ن": "n", "ھ": "h",
        "و": "o", "ۇ": "u", "ۆ": "ö", "ۈ": "ü", "ۋ": "w", "ې": "é",
        "ى": "i", "ي": "y", "ئ": "",
    },

    # Turkmen Cyrillic → Latin (1993 official Latin alphabet)
    "tuk_Cyrl_to_Latn": {
        # Digraphs for loanwords
        "щ": "şç", "Щ": "Şç",
        # Turkmen-specific Cyrillic letters
        "ә": "ä", "Ә": "Ä", "җ": "j", "Җ": "J", "ң": "ň", "Ң": "Ň",
        "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        # Standard Cyrillic
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "w", "В": "W",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "ýo", "Ё": "Ýo", "ж": "ž", "Ж": "Ž", "з": "z", "З": "Z",
        "и": "i", "И": "I", "й": "ý", "Й": "Ý", "к": "k", "К": "K",
        "л": "l", "Л": "L", "м": "m", "М": "M", "н": "n", "Н": "N",
        "о": "o", "О": "O", "п": "p", "П": "P", "р": "r", "Р": "R",
        "с": "s", "С": "S", "т": "t", "Т": "T", "у": "u", "У": "U",
        "ф": "f", "Ф": "F", "х": "h", "Х": "H", "ц": "ts", "Ц": "Ts",
        "ч": "ç", "Ч": "Ç", "ш": "ş", "Ш": "Ş", "ы": "y", "Ы": "Y",
        "э": "e", "Э": "E", "ю": "ýu", "Ю": "Ýu", "я": "ýa", "Я": "Ýa",
        "ъ": "", "ь": "",
    },

    # Turkmen Latin → Cyrillic
    "tuk_Latn_to_Cyrl": {
        # Multi-char sequences first (greedy match)
        "şç": "щ", "Şç": "Щ",
        "ýe": "е", "Ýe": "Е", "ÝE": "Е",
        "ýo": "ё", "Ýo": "Ё", "ÝO": "Ё",
        "ýu": "ю", "Ýu": "Ю", "ÝU": "Ю",
        "ýa": "я", "Ýa": "Я", "ÝA": "Я",
        "ts": "ц", "Ts": "Ц", "TS": "Ц",
        # Turkmen-specific Latin letters
        "ä": "ә", "Ä": "Ә", "ň": "ң", "Ň": "Ң",
        "ö": "ө", "Ö": "Ө", "ü": "ү", "Ü": "Ү",
        "ç": "ч", "Ç": "Ч", "ş": "ш", "Ş": "Ш",
        "ž": "ж", "Ž": "Ж", "ý": "й", "Ý": "Й",
        # Standard Latin → Cyrillic (note: j → җ, w → в)
        "a": "а", "A": "А", "b": "б", "B": "Б", "d": "д", "D": "Д",
        "e": "е", "E": "Е", "f": "ф", "F": "Ф", "g": "г", "G": "Г",
        "h": "х", "H": "Х", "i": "и", "I": "И", "j": "җ", "J": "Җ",
        "k": "к", "K": "К", "l": "л", "L": "Л", "m": "м", "M": "М",
        "n": "н", "N": "Н", "o": "о", "O": "О", "p": "п", "P": "П",
        "r": "р", "R": "Р", "s": "с", "S": "С", "t": "т", "T": "Т",
        "u": "у", "U": "У", "w": "в", "W": "В", "y": "ы", "Y": "Ы",
        "z": "з", "Z": "З",
    },

    # Crimean Tatar Cyrillic → Latin
    "crh_Cyrl_to_Latn": {
        # ё/ю after consonants -> ö/ü (Transliterator._crh_cyrl_pre)
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ы": "ı", "Ы": "I", "Ъ": "", "Ь": "",
        "гъ": "ğ", "Гъ": "Ğ", "дж": "c", "Дж": "C", "къ": "q", "Къ": "Q",
        "нъ": "ñ", "Нъ": "Ñ",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ж": "j", "Ж": "J", "з": "z", "З": "Z", "и": "i", "И": "İ",
        "й": "y", "Й": "Y", "к": "k", "К": "K", "л": "l", "Л": "L",
        "м": "m", "М": "M", "н": "n", "Н": "N", "о": "o", "О": "O",
        "п": "p", "П": "P", "р": "r", "Р": "R", "с": "s", "С": "S",
        "т": "t", "Т": "T", "у": "u", "У": "U", "ф": "f", "Ф": "F",
        "х": "h", "Х": "H", "ц": "ts", "Ц": "Ts", "ч": "ç", "Ч": "Ç",
        "ш": "ş", "Ш": "Ş", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Crimean Tatar Latin → Cyrillic
    "crh_Latn_to_Cyrl": {
        # yo/yu/ya (and yö/yü) -> ё/ю/я; ö/ü after consonants -> ё/ю
        # (Transliterator._crh_latn_pre), elsewhere -> о/у.
        "yo": "ё", "Yo": "Ё", "YO": "Ё", "yö": "ё", "Yö": "Ё", "YÖ": "Ё",
        "yu": "ю", "Yu": "Ю", "YU": "Ю", "yü": "ю", "Yü": "Ю", "YÜ": "Ю",
        "ya": "я", "Ya": "Я", "YA": "Я",
        "ı": "ы", "I": "Ы", "ö": "о", "Ö": "О", "ü": "у", "Ü": "У",
        # Digraphs first (greedy match)
        "ts": "ц", "Ts": "Ц", "TS": "Ц",
        # Special Latin letters → Cyrillic digraphs
        "ğ": "гъ", "Ğ": "Гъ", "q": "къ", "Q": "Къ", "ñ": "нъ", "Ñ": "Нъ",
        "c": "дж", "C": "Дж",
        # Single characters
        "ç": "ч", "Ç": "Ч", "ş": "ш", "Ş": "Ш", "İ": "И", "i": "и",
        "a": "а", "A": "А", "b": "б", "B": "Б", "d": "д", "D": "Д",
        "e": "е", "E": "Е", "f": "ф", "F": "Ф", "g": "г", "G": "Г",
        "h": "х", "H": "Х", "j": "ж", "J": "Ж", "k": "к", "K": "К",
        "l": "л", "L": "Л", "m": "м", "M": "М", "n": "н", "N": "Н",
        "o": "о", "O": "О", "p": "п", "P": "П", "r": "р", "R": "Р",
        "s": "с", "S": "С", "t": "т", "T": "Т", "u": "у", "U": "У",
        "v": "в", "V": "В", "y": "й", "Y": "Й", "z": "з", "Z": "З",
    },

    # Uzbek Latin → Cyrillic (reverse of 1995 official alphabet)
    "uzb_Latn_to_Cyrl": {
        # "yo'" must win over "yo" (ё): yo'q -> йўқ
        "yo'": "йў", "Yo'": "Йў", "YO'": "ЙЎ",
        # s'h / sʼh = с + ҳ (apostrophe variants are normalised to ')
        "s'h": "сҳ", "S'h": "Сҳ", "S'H": "СҲ",
        # Digraphs/trigraphs first (greedy match)
        "sh": "ш", "Sh": "Ш", "SH": "Ш",
        "ch": "ч", "Ch": "Ч", "CH": "Ч",
        "ng": "нг", "Ng": "Нг", "NG": "НГ",
        "g'": "ғ", "G'": "Ғ", "o'": "ў", "O'": "Ў",
        "yo": "ё", "Yo": "Ё", "YO": "Ё",
        "yu": "ю", "Yu": "Ю", "YU": "Ю",
        "ya": "я", "Ya": "Я", "YA": "Я",
        "ts": "ц", "Ts": "Ц", "TS": "Ц",
        # Single characters
        "'": "ъ",
        "a": "а", "A": "А", "b": "б", "B": "Б", "d": "д", "D": "Д",
        "e": "е", "E": "Е", "f": "ф", "F": "Ф", "g": "г", "G": "Г",
        "h": "ҳ", "H": "Ҳ", "i": "и", "I": "И", "j": "ж", "J": "Ж",
        "k": "к", "K": "К", "l": "л", "L": "Л", "m": "м", "M": "М",
        "n": "н", "N": "Н", "o": "о", "O": "О", "p": "п", "P": "П",
        "q": "қ", "Q": "Қ", "r": "р", "R": "Р", "s": "с", "S": "С",
        "t": "т", "T": "Т", "u": "у", "U": "У", "v": "в", "V": "В",
        "x": "х", "X": "Х", "y": "й", "Y": "Й", "z": "з", "Z": "З",
    },

    # Uyghur Latin (ULY) → Perso-Arabic
    "uig_Latn_to_Arab": {
        # Digraphs first (greedy match)
        "ch": "چ", "Ch": "چ", "CH": "چ",
        "sh": "ش", "Sh": "ش", "SH": "ش",
        "zh": "ژ", "Zh": "ژ", "ZH": "ژ",
        "gh": "غ", "Gh": "غ", "GH": "غ",
        "ng": "ڭ", "Ng": "ڭ", "NG": "ڭ",
        # Single characters
        "a": "ا", "e": "ە", "b": "ب", "p": "پ", "t": "ت", "j": "ج",
        "x": "خ", "d": "د", "r": "ر", "z": "ز", "s": "س", "f": "ف",
        "q": "ق", "k": "ك", "g": "گ", "l": "ل", "m": "م", "n": "ن",
        "h": "ھ", "o": "و", "u": "ۇ", "ö": "ۆ", "ü": "ۈ", "w": "ۋ",
        "é": "ې", "i": "ى", "y": "ي",
        # Capitals map to same Arabic letters (Arabic has no case)
        "A": "ا", "E": "ە", "B": "ب", "P": "پ", "T": "ت", "J": "ج",
        "X": "خ", "D": "د", "R": "ر", "Z": "ز", "S": "س", "F": "ف",
        "Q": "ق", "K": "ك", "G": "گ", "L": "ل", "M": "م", "N": "ن",
        "H": "ھ", "O": "و", "U": "ۇ", "Ö": "ۆ", "Ü": "ۈ", "W": "ۋ",
        "É": "ې", "I": "ى", "Y": "ي",
    },

    # Azerbaijani Cyrillic → Latin (1991 official Latin alphabet)
    "aze_Cyrl_to_Latn": {
        # Digraphs for Russian loanwords
        "щ": "şç", "Щ": "Şç",
        # Azerbaijani-specific Cyrillic letters
        "ә": "ə", "Ә": "Ə", "ғ": "ğ", "Ғ": "Ğ", "ө": "ö", "Ө": "Ö",
        "ү": "ü", "Ү": "Ü", "ҹ": "c", "Ҹ": "C", "ҝ": "g", "Ҝ": "G",
        "һ": "h", "Һ": "H",
        # Standard Cyrillic
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "q", "Г": "Q", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "ж": "j", "Ж": "J", "з": "z", "З": "Z",
        "и": "i", "И": "İ", "к": "k", "К": "K",
        # Azerbaijani Cyrillic y is ј (U+0458); й is accepted as a variant
        "ј": "y", "Ј": "Y", "й": "y", "Й": "Y",
        "л": "l", "Л": "L", "м": "m", "М": "M", "н": "n", "Н": "N",
        "о": "o", "О": "O", "п": "p", "П": "P", "р": "r", "Р": "R",
        "с": "s", "С": "S", "т": "t", "Т": "T", "у": "u", "У": "U",
        "ф": "f", "Ф": "F", "х": "x", "Х": "X", "ц": "ts", "Ц": "Ts",
        "ч": "ç", "Ч": "Ç", "ш": "ş", "Ш": "Ş", "ы": "ı", "Ы": "I",
        "э": "e", "Э": "E", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ъ": "", "ь": "",
    },

    # Azerbaijani Latin → Cyrillic
    "aze_Latn_to_Cyrl": {
        # Digraphs first (greedy match). Azerbaijani Cyrillic writes y as ј
        # (јол, јахшы); ё/ю/я occur only in Russian loans, so yo/yu/ya are
        # written јо/ју/ја.
        "şç": "щ", "Şç": "Щ",
        "ts": "ц", "Ts": "Ц", "TS": "Ц",
        # Azerbaijani-specific Latin letters
        "ə": "ә", "Ə": "Ә", "ğ": "ғ", "Ğ": "Ғ", "ö": "ө", "Ö": "Ө",
        "ü": "ү", "Ü": "Ү", "ç": "ч", "Ç": "Ч", "ş": "ш", "Ş": "Ш",
        "İ": "И", "ı": "ы",
        # Standard Latin → Cyrillic (note: g → ҝ, q → г, c → ҹ)
        "a": "а", "A": "А", "b": "б", "B": "Б", "c": "ҹ", "C": "Ҹ",
        "d": "д", "D": "Д", "e": "е", "E": "Е", "f": "ф", "F": "Ф",
        "g": "ҝ", "G": "Ҝ", "h": "һ", "H": "Һ", "i": "и", "I": "Ы",
        "j": "ж", "J": "Ж", "k": "к", "K": "К", "l": "л", "L": "Л",
        "m": "м", "M": "М", "n": "н", "N": "Н", "o": "о", "O": "О",
        "p": "п", "P": "П", "q": "г", "Q": "Г", "r": "р", "R": "Р",
        "s": "с", "S": "С", "t": "т", "T": "Т", "u": "у", "U": "У",
        "v": "в", "V": "В", "x": "х", "X": "Х", "y": "ј", "Y": "Ј",
        "z": "з", "Z": "З",
    },

    # Tatar Cyrillic → Latin (Zamanälif)
    "tat_Cyrl_to_Latn": {
        # Digraphs for loanwords
        "щ": "şç", "Щ": "Şç",
        # Tatar-specific Cyrillic letters
        "ә": "ä", "Ә": "Ä", "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "җ": "c", "Җ": "C", "ң": "ñ", "Ң": "Ñ", "һ": "h", "Һ": "H",
        # Standard Cyrillic
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "w", "В": "W",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "ж": "j", "Ж": "J", "з": "z", "З": "Z",
        "и": "i", "И": "İ", "й": "y", "Й": "Y", "к": "k", "К": "K",
        "л": "l", "Л": "L", "м": "m", "М": "M", "н": "n", "Н": "N",
        "о": "o", "О": "O", "п": "p", "П": "P", "р": "r", "Р": "R",
        "с": "s", "С": "S", "т": "t", "Т": "T", "у": "u", "У": "U",
        "ф": "f", "Ф": "F", "х": "x", "Х": "X", "ц": "ts", "Ц": "Ts",
        "ч": "ç", "Ч": "Ç", "ш": "ş", "Ш": "Ş", "ы": "ı", "Ы": "I",
        "э": "e", "Э": "E", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ъ": "", "ь": "",
    },

    # Tatar Latin (Zamanälif) → Cyrillic
    "tat_Latn_to_Cyrl": {
        # Digraphs first (greedy match)
        "şç": "щ", "Şç": "Щ",
        "yo": "ё", "Yo": "Ё", "YO": "Ё",
        "yu": "ю", "Yu": "Ю", "YU": "Ю",
        "ya": "я", "Ya": "Я", "YA": "Я",
        "ts": "ц", "Ts": "Ц", "TS": "Ц",
        # Tatar-specific Latin letters
        "ä": "ә", "Ä": "Ә", "ö": "ө", "Ö": "Ө", "ü": "ү", "Ü": "Ү",
        "ñ": "ң", "Ñ": "Ң", "ç": "ч", "Ç": "Ч", "ş": "ш", "Ş": "Ш",
        "İ": "И", "ı": "ы",
        # Standard Latin → Cyrillic (note: c → җ, w → в)
        "a": "а", "A": "А", "b": "б", "B": "Б", "c": "җ", "C": "Җ",
        "d": "д", "D": "Д", "e": "е", "E": "Е", "f": "ф", "F": "Ф",
        "g": "г", "G": "Г", "h": "һ", "H": "Һ", "i": "и", "I": "Ы",
        "j": "ж", "J": "Ж", "k": "к", "K": "К", "l": "л", "L": "Л",
        "m": "м", "M": "М", "n": "н", "N": "Н", "o": "о", "O": "О",
        "p": "п", "P": "П", "r": "р", "R": "Р", "s": "с", "S": "С",
        "t": "т", "T": "Т", "u": "у", "U": "У", "w": "в", "W": "В",
        "x": "х", "X": "Х", "y": "й", "Y": "Й", "z": "з", "Z": "З",
    },

    # Karakalpak Cyrillic → Latin (2016 Latin alphabet)
    "kaa_Cyrl_to_Latn": {
        # Digraphs for loanwords
        "щ": "shch", "Щ": "Shch",
        # Karakalpak-specific Cyrillic letters
        "ә": "á", "Ә": "Á", "ғ": "ǵ", "Ғ": "Ǵ", "қ": "q", "Қ": "Q",
        "ң": "ń", "Ң": "Ń", "ө": "ó", "Ө": "Ó", "ү": "ú", "Ү": "Ú",
        "ў": "w", "Ў": "W", "ҳ": "h", "Ҳ": "H",
        # Standard Cyrillic
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "ж": "j", "Ж": "J", "з": "z", "З": "Z",
        "и": "i", "И": "I", "й": "y", "Й": "Y", "к": "k", "К": "K",
        "л": "l", "Л": "L", "м": "m", "М": "M", "н": "n", "Н": "N",
        "о": "o", "О": "O", "п": "p", "П": "P", "р": "r", "Р": "R",
        "с": "s", "С": "S", "т": "t", "Т": "T", "у": "u", "У": "U",
        "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ц": "ts", "Ц": "Ts", "ч": "ch", "Ч": "Ch",
        "ш": "sh", "Ш": "Sh", "ы": "ı", "Ы": "Í",   # 2016: Í ı
        "э": "e", "Э": "E", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ъ": "", "ь": "",
    },

    # Karakalpak Latin → Cyrillic
    "kaa_Latn_to_Cyrl": {
        "ı": "ы",   # 2016 alphabet: Í ı = ы (í accepted as a variant)
        # Multi-char sequences first (greedy match)
        "shch": "щ", "Shch": "Щ",
        "sh": "ш", "Sh": "Ш", "SH": "Ш",
        "ch": "ч", "Ch": "Ч", "CH": "Ч",
        "yo": "ё", "Yo": "Ё", "YO": "Ё",
        "yu": "ю", "Yu": "Ю", "YU": "Ю",
        "ya": "я", "Ya": "Я", "YA": "Я",
        "ts": "ц", "Ts": "Ц", "TS": "Ц",
        # Karakalpak-specific Latin letters
        "á": "ә", "Á": "Ә", "ǵ": "ғ", "Ǵ": "Ғ", "ń": "ң", "Ń": "Ң",
        "ó": "ө", "Ó": "Ө", "ú": "ү", "Ú": "Ү", "í": "ы", "Í": "Ы",
        # Standard Latin → Cyrillic
        "a": "а", "A": "А", "b": "б", "B": "Б", "d": "д", "D": "Д",
        "e": "е", "E": "Е", "f": "ф", "F": "Ф", "g": "г", "G": "Г",
        "h": "ҳ", "H": "Ҳ", "i": "и", "I": "И", "j": "ж", "J": "Ж",
        "k": "к", "K": "К", "l": "л", "L": "Л", "m": "м", "M": "М",
        "n": "н", "N": "Н", "o": "о", "O": "О", "p": "п", "P": "П",
        "q": "қ", "Q": "Қ", "r": "р", "R": "Р", "s": "с", "S": "С",
        "t": "т", "T": "Т", "u": "у", "U": "У", "v": "в", "V": "В",
        "w": "ў", "W": "Ў", "x": "х", "X": "Х", "y": "й", "Y": "Й",
        "z": "з", "Z": "З",
    },

    # Ottoman Turkish Latin (academic transcription) → Perso-Arabic
    # Note: This is a simplified mapping. Ottoman script has many
    # Arabic/Persian letters that share Latin equivalents; this uses
    # the most common correspondence for each letter.
    "ota_Latn_to_Arab": {
        # Digraphs
        "ch": "چ", "Ch": "چ", "CH": "چ",
        "sh": "ش", "Sh": "ش", "SH": "ش",
        # Single characters
        "a": "ا", "A": "ا", "b": "ب", "B": "ب", "c": "ج", "C": "ج",
        "ç": "چ", "Ç": "چ", "d": "د", "D": "د", "e": "ه", "E": "ه",
        "f": "ف", "F": "ف", "g": "گ", "G": "گ", "ğ": "غ", "Ğ": "غ",
        "h": "ح", "H": "ح", "ı": "ی", "i": "ی", "I": "ی", "İ": "ی",
        "j": "ژ", "J": "ژ", "k": "ك", "K": "ك", "l": "ل", "L": "ل",
        "m": "م", "M": "م", "n": "ن", "N": "ن", "o": "و", "O": "و",
        "ö": "و", "Ö": "و", "p": "پ", "P": "پ", "r": "ر", "R": "ر",
        "s": "س", "S": "س", "ş": "ش", "Ş": "ش", "t": "ت", "T": "ت",
        "u": "و", "U": "و", "ü": "و", "Ü": "و", "v": "و", "V": "و",
        "y": "ی", "Y": "ی", "z": "ز", "Z": "ز",
    },

    # Old Turkic Runic (Orkhon) → Latin transliteration
    # Based on standard Turkological conventions for Orkhon inscriptions.
    # Unicode block U+10C00–U+10C4F.
    "otk_Orkh_to_Latn": _build_old_turkic_table(),

    # ---------------------------------------------------------------------------
    # Common Turkic Script (CTS) tables
    # Based on the Common Turkic Alphabet: https://en.wikipedia.org/wiki/Common_Turkic_alphabet
    # ---------------------------------------------------------------------------

    # Turkish Latin → CTS (near-identical; Turkish already uses CTS-compatible letters)
    "tur_Latn_to_CTS": {
        "a": "a", "A": "A", "b": "b", "B": "B", "c": "c", "C": "C",
        "ç": "ç", "Ç": "Ç", "d": "d", "D": "D", "e": "e", "E": "E",
        "f": "f", "F": "F", "g": "g", "G": "G", "ğ": "ğ", "Ğ": "Ğ",
        "h": "h", "H": "H", "ı": "ı", "I": "I", "i": "i", "İ": "İ",
        "j": "j", "J": "J", "k": "k", "K": "K", "l": "l", "L": "L",
        "m": "m", "M": "M", "n": "n", "N": "N", "o": "o", "O": "O",
        "ö": "ö", "Ö": "Ö", "p": "p", "P": "P", "r": "r", "R": "R",
        "s": "s", "S": "S", "ş": "ş", "Ş": "Ş", "t": "t", "T": "T",
        "u": "u", "U": "U", "ü": "ü", "Ü": "Ü", "v": "v", "V": "V",
        "y": "y", "Y": "Y", "z": "z", "Z": "Z",
    },

    # Azerbaijani Latin → CTS (ə → ä is the only non-trivial mapping)
    "aze_Latn_to_CTS": {
        "ə": "ä", "Ə": "Ä",
        "a": "a", "A": "A", "b": "b", "B": "B", "c": "c", "C": "C",
        "ç": "ç", "Ç": "Ç", "d": "d", "D": "D", "e": "e", "E": "E",
        "f": "f", "F": "F", "g": "g", "G": "G", "ğ": "ğ", "Ğ": "Ğ",
        "h": "h", "H": "H", "x": "x", "X": "X", "ı": "ı", "I": "I",
        "i": "i", "İ": "İ", "j": "j", "J": "J", "k": "k", "K": "K",
        "l": "l", "L": "L", "m": "m", "M": "M", "n": "n", "N": "N",
        "o": "o", "O": "O", "ö": "ö", "Ö": "Ö", "p": "p", "P": "P",
        "q": "q", "Q": "Q", "r": "r", "R": "R", "s": "s", "S": "S",
        "ş": "ş", "Ş": "Ş", "t": "t", "T": "T", "u": "u", "U": "U",
        "ü": "ü", "Ü": "Ü", "v": "v", "V": "V", "y": "y", "Y": "Y",
        "z": "z", "Z": "Z",
    },

    # Azerbaijani Cyrillic → CTS
    "aze_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ц": "ts", "Ц": "Ts",
        "ə": "ä", "Ə": "Ä",
        "ә": "ä", "Ә": "Ä", "ғ": "ğ", "Ғ": "Ğ", "ө": "ö", "Ө": "Ö",
        "ү": "ü", "Ү": "Ü", "ҹ": "c", "Ҹ": "C", "ҝ": "g", "Ҝ": "G",
        "һ": "h", "Һ": "H",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "q", "Г": "Q", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ж": "j", "Ж": "J", "з": "z", "З": "Z", "и": "i", "И": "İ",
        "ј": "y", "Ј": "Y", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L",
        "м": "m", "М": "M", "н": "n", "Н": "N", "о": "o", "О": "O",
        "п": "p", "П": "P", "р": "r", "Р": "R", "с": "s", "С": "S",
        "т": "t", "Т": "T", "у": "u", "У": "U", "ф": "f", "Ф": "F",
        "х": "x", "Х": "X", "ч": "ç", "Ч": "Ç", "ш": "ş", "Ш": "Ş",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # South Azerbaijani (Perso-Arabic) → CTS
    "azb_Arab_to_CTS": {
        # Vowel letters (context rules in Transliterator._transliterate_azb_arab_to_cts)
        "آ": "a", "ا": "a", "ع": "", "ه": "h",
        "و": "v", "ی": "y", "ي": "y", "ى": "y",
        "ئ": "e", "ؤ": "ö", "ۇ": "u", "ۆ": "ü", "ۉ": "ü", "ؽ": "ı",
        "ة": "ä",
        # Consonants
        "ب": "b", "پ": "p", "ت": "t", "ث": "s",
        "ج": "c", "چ": "ç", "ح": "h", "خ": "x",
        "د": "d", "ذ": "z", "ر": "r", "ز": "z",
        "ژ": "j", "س": "s", "ش": "ş", "ص": "s",
        "ض": "z", "ط": "t", "ظ": "z", "غ": "ğ",
        "ف": "f", "ق": "q", "ک": "k", "ك": "k",
        "گ": "g", "ل": "l", "م": "m", "ن": "n",
        "ء": "",
    },

    # Kazakh Cyrillic → CTS
    "kaz_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç", "ж": "c", "Ж": "C",
        "ц": "ts", "Ц": "Ts",
        "ә": "ä", "Ә": "Ä", "ғ": "ğ", "Ғ": "Ğ", "қ": "q", "Қ": "Q",
        "ң": "ñ", "Ң": "Ñ", "ө": "ö", "Ө": "Ö", "ұ": "u", "Ұ": "U",
        "ү": "ü", "Ү": "Ü", "і": "i", "І": "İ", "һ": "h", "Һ": "H",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "э": "e", "Э": "E", "ы": "ı", "Ы": "I", "ъ": "", "ь": "",
    },

    # Kazakh Latin (2021 official alphabet) → CTS
    "kaz_Latn_to_CTS": {
        # Input: 2021 official Kazakh Latin alphabet
        "ştş": "şç", "Ştş": "Şç", "tş": "ç", "Tş": "Ç",
        "io": "yo", "İo": "Yo", "iu": "yu", "İu": "Yu", "ia": "ya", "İa": "Ya",
        "ä": "ä", "Ä": "Ä", "ğ": "ğ", "Ğ": "Ğ", "q": "q", "Q": "Q",
        "ñ": "ñ", "Ñ": "Ñ", "ö": "ö", "Ö": "Ö", "ū": "u", "Ū": "U",
        "ü": "ü", "Ü": "Ü", "ş": "ş", "Ş": "Ş",
        "ı": "i", "I": "İ",   # Kazakh 2021 ı (Cyrillic і) -> CTS i
        "i": "i", "İ": "İ",   # Kazakh 2021 i (Cyrillic и/й) -> CTS i
        "y": "ı", "Y": "I",   # Kazakh 2021 y (Cyrillic ы) -> CTS ı
        "h": "x", "H": "X",   # Kazakh h (Cyrillic х/һ) -> CTS x
        "j": "c", "J": "C",
        "a": "a", "A": "A", "b": "b", "B": "B", "d": "d", "D": "D",
        "e": "e", "E": "E", "f": "f", "F": "F", "g": "g", "G": "G",
        "k": "k", "K": "K", "l": "l", "L": "L", "m": "m", "M": "M",
        "n": "n", "N": "N", "o": "o", "O": "O", "p": "p", "P": "P",
        "r": "r", "R": "R", "s": "s", "S": "S", "t": "t", "T": "T",
        "u": "u", "U": "U", "v": "v", "V": "V", "z": "z", "Z": "Z",
    },

    # Uzbek Latin (1995) → CTS
    "uzb_Latn_to_CTS": {
        # apostrophe variants (‘ ʻ ’ ʼ) are normalised to ' before lookup
        "s'h": "sh", "S'h": "Sh", "S'H": "SH",
        "yo'": "yö", "Yo'": "Yö", "YO'": "YÖ",
        "sh": "ş", "Sh": "Ş", "SH": "Ş",
        "ch": "ç", "Ch": "Ç", "CH": "Ç",
        "ng": "ñ", "Ng": "Ñ", "NG": "Ñ",
        "g'": "ğ", "G'": "Ğ",
        "o'": "ö", "O'": "Ö",
        "yo": "yo", "Yo": "Yo",
        "yu": "yu", "Yu": "Yu",
        "ya": "ya", "Ya": "Ya",
        "'": "",
        "a": "a", "A": "A", "b": "b", "B": "B", "d": "d", "D": "D",
        "e": "e", "E": "E", "f": "f", "F": "F", "g": "g", "G": "G",
        "h": "h", "H": "H", "i": "i", "I": "İ", "j": "c", "J": "C",
        "k": "k", "K": "K", "l": "l", "L": "L", "m": "m", "M": "M",
        "n": "n", "N": "N", "o": "o", "O": "O", "p": "p", "P": "P",
        "q": "q", "Q": "Q", "r": "r", "R": "R", "s": "s", "S": "S",
        "t": "t", "T": "T", "u": "u", "U": "U", "v": "v", "V": "V",
        "x": "x", "X": "X", "y": "y", "Y": "Y", "z": "z", "Z": "Z",
    },

    # Uzbek Cyrillic → CTS
    "uzb_Cyrl_to_CTS": {
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "нг": "ñ", "Нг": "Ñ",
        "ғ": "ğ", "Ғ": "Ğ", "қ": "q", "Қ": "Q",
        "ҳ": "h", "Ҳ": "H", "ў": "ö", "Ў": "Ö",
        "ж": "c", "Ж": "C",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ц": "ts", "Ц": "Ts",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Kyrgyz Cyrillic → CTS
    "kir_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç", "ж": "c", "Ж": "C",
        "ц": "ts", "Ц": "Ts",
        "ң": "ñ", "Ң": "Ñ", "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "з": "z", "З": "Z", "и": "i", "И": "İ",
        "й": "y", "Й": "Y", "к": "k", "К": "K", "л": "l", "Л": "L",
        "м": "m", "М": "M", "н": "n", "Н": "N", "о": "o", "О": "O",
        "п": "p", "П": "P", "р": "r", "Р": "R", "с": "s", "С": "S",
        "т": "t", "Т": "T", "у": "u", "У": "U", "ф": "f", "Ф": "F",
        "х": "x", "Х": "X", "ы": "ı", "Ы": "I", "э": "e", "Э": "E",
        "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya", "ъ": "", "ь": "",
    },

    # Turkmen Latin → CTS
    "tuk_Latn_to_CTS": {
        "şç": "şç", "Şç": "Şç",
        "ts": "ts", "Ts": "Ts",
        "ä": "ä", "Ä": "Ä", "ň": "ñ", "Ň": "Ñ",
        "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ç": "ç", "Ç": "Ç", "ş": "ş", "Ş": "Ş",
        "ž": "j", "Ž": "J",
        "ý": "y", "Ý": "Y",   # Turkmen ý = /j/ → CTS y
        "y": "ı", "Y": "I",   # Turkmen y = /ɯ/ → CTS ı
        "j": "c", "J": "C",
        "w": "v", "W": "V",   # Turkmen w = /w/ → CTS V slot (Turkmen has no /v/)
        "a": "a", "A": "A", "b": "b", "B": "B", "d": "d", "D": "D",
        "e": "e", "E": "E", "f": "f", "F": "F", "g": "g", "G": "G",
        "h": "h", "H": "H", "i": "i", "I": "İ", "k": "k", "K": "K",
        "l": "l", "L": "L", "m": "m", "M": "M", "n": "n", "N": "N",
        "o": "o", "O": "O", "p": "p", "P": "P", "r": "r", "R": "R",
        "s": "s", "S": "S", "t": "t", "T": "T", "u": "u", "U": "U",
        "z": "z", "Z": "Z",
    },

    # Turkmen Cyrillic → CTS
    "tuk_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ц": "ts", "Ц": "Ts",
        "ә": "ä", "Ә": "Ä", "ң": "ñ", "Ң": "Ñ",
        "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "ч": "ç", "Ч": "Ç", "ш": "ş", "Ш": "Ş",
        "ж": "j", "Ж": "J", "з": "z", "З": "Z",
        "й": "y", "Й": "Y", "в": "v", "В": "V",   # Turkmen в = /w/ → CTS V slot
        "а": "a", "А": "A", "б": "b", "Б": "B", "г": "g", "Г": "G",
        "д": "d", "Д": "D", "е": "e", "Е": "E", "и": "i", "И": "İ",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "h", "Х": "H",
        "ы": "ı", "Ы": "I",   # Turkmen ы = /ɯ/ → CTS ı
        "э": "e", "Э": "E", "ъ": "", "ь": "",
        "җ": "c", "Җ": "C",
    },

    # Tatar Cyrillic → CTS
    "tat_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "j", "Ж": "J",   # Tatar ж = /ʒ/
        "җ": "c", "Җ": "C",   # Tatar Ж̧ = /dʒ/
        "ц": "ts", "Ц": "Ts",
        "ә": "ä", "Ә": "Ä", "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "ң": "ñ", "Ң": "Ñ", "һ": "h", "Һ": "H",
        "в": "w", "В": "W",   # Tatar в = /w/
        "а": "a", "А": "A", "б": "b", "Б": "B", "г": "g", "Г": "G",
        "д": "d", "Д": "D", "е": "e", "Е": "E", "ё": "yo", "Ё": "Yo",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E",
        "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya", "ъ": "", "ь": "",
    },

    # Tatar Latin (Zamanälif) → CTS
    "tat_Latn_to_CTS": {
        "şç": "şç", "Şç": "Şç",
        "ts": "ts", "Ts": "Ts",
        "ä": "ä", "Ä": "Ä", "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ñ": "ñ", "Ñ": "Ñ", "ç": "ç", "Ç": "Ç", "ş": "ş", "Ş": "Ş",
        "c": "c", "C": "C",   # Zamanälif c = /dʒ/
        "j": "j", "J": "J",   # Zamanälif j = /ʒ/
        "w": "w", "W": "W",   # Tatar w = /w/
        "İ": "İ", "ı": "ı", "I": "I",
        "a": "a", "A": "A", "b": "b", "B": "B", "d": "d", "D": "D",
        "e": "e", "E": "E", "f": "f", "F": "F", "g": "g", "G": "G",
        "h": "h", "H": "H", "i": "i", "k": "k", "K": "K",
        "l": "l", "L": "L", "m": "m", "M": "M", "n": "n", "N": "N",
        "o": "o", "O": "O", "p": "p", "P": "P", "r": "r", "R": "R",
        "s": "s", "S": "S", "t": "t", "T": "T", "u": "u", "U": "U",
        "x": "x", "X": "X", "y": "y", "Y": "Y", "z": "z", "Z": "Z",
    },

    # Uyghur Cyrillic (UCS) → CTS
    "uig_Cyrl_to_CTS": {
        "ла": "la", "Ла": "La",   # multi-char first
        "я": "ya", "Я": "Ya", "ю": "yu", "Ю": "Yu",
        "а": "a", "А": "A", "ә": "e", "Ә": "E",
        "б": "b", "Б": "B", "п": "p", "П": "P",
        "т": "t", "Т": "T", "җ": "c", "Җ": "C",
        "ч": "ç", "Ч": "Ç", "х": "x", "Х": "X",
        "д": "d", "Д": "D", "р": "r", "Р": "R",
        "з": "z", "З": "Z", "ж": "j", "Ж": "J",
        "с": "s", "С": "S", "ш": "ş", "Ш": "Ş",
        "ф": "f", "Ф": "F", "ң": "ñ", "Ң": "Ñ",
        "л": "l", "Л": "L", "м": "m", "М": "M",
        "һ": "h", "Һ": "H", "о": "o", "О": "O",
        "у": "u", "У": "U", "ө": "ö", "Ө": "Ö",
        "ү": "ü", "Ү": "Ü", "в": "v", "В": "V",
        "е": "é", "Е": "É", "и": "i", "И": "I",
        "й": "y", "Й": "Y", "қ": "q", "Қ": "Q",
        "к": "k", "К": "K", "г": "g", "Г": "G",
        "н": "n", "Н": "N", "ғ": "ğ", "Ғ": "Ğ",
    },

    # Uyghur CTS → Cyrillic (UCS)
    "uig_CTS_to_Cyrl": {
        "la": "ла", "La": "Ла",   # multi-char first
        "ya": "я", "Ya": "Я", "yu": "ю", "Yu": "Ю",
        "a": "а", "A": "А", "e": "ә", "E": "Ә",
        "b": "б", "B": "Б", "p": "п", "P": "П",
        "t": "т", "T": "Т", "c": "җ", "C": "Җ",
        "ç": "ч", "Ç": "Ч", "x": "х", "X": "Х",
        "d": "д", "D": "Д", "r": "р", "R": "Р",
        "z": "з", "Z": "З", "j": "ж", "J": "Ж",
        "s": "с", "S": "С", "ş": "ш", "Ş": "Ш",
        "f": "ф", "F": "Ф", "ñ": "ң", "Ñ": "Ң",
        "l": "л", "L": "Л", "m": "м", "M": "М",
        "h": "һ", "H": "Һ", "o": "о", "O": "О",
        "u": "у", "U": "У", "ö": "ө", "Ö": "Ө",
        "ü": "ү", "Ü": "Ү", "v": "в", "V": "В",
        "é": "е", "É": "Е", "i": "и", "I": "И",
        "y": "й", "Y": "Й", "q": "қ", "Q": "Қ",
        "k": "к", "K": "К", "g": "г", "G": "Г",
        "n": "н", "N": "Н", "ğ": "ғ", "Ğ": "Ғ",
    },

    # Uyghur: placeholder entries for pairs handled by special methods
    "uig_Arab_to_CTS": {},
    "uig_CTS_to_Arab": {},
    "uig_Arab_to_Cyrl": {},
    "uig_Cyrl_to_Arab": {},
    "uig_Latn_to_Cyrl": {},
    "uig_Cyrl_to_Latn": {},
    "uig_Latn_to_CTS": {},
    "uig_CTS_to_Latn": {},

    # Bashkir Cyrillic → CTS
    "bak_Cyrl_to_CTS": {
        # Cyrillic ә/ө (U+04D9/U+04E9) as used in Bashkir text, and һ
        "ә": "ä", "Ә": "Ä", "ө": "ö", "Ө": "Ö", "һ": "h", "Һ": "H",
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "j", "Ж": "J",   # Bashkir ж = /ʒ/
        "ц": "ts", "Ц": "Ts",
        "ä": "ä", "Ä": "Ä",   # Bashkir ä (a with breve in some fonts)
        "ă": "ä", "Ă": "Ä",   # alternative Bashkir reduced a
        "ö": "ö", "Ö": "Ö",
        "ü": "ü", "Ü": "Ü",
        # Bashkir /ð/ and /θ/ have no letter among the 34 CTS letters; they map
        # to their Common Turkic cognate consonants z and s (Bashkir ҡыҙ ~
        # Tatar кыз = qız, Bashkir ҫу ~ Tatar су = su).
        "ҙ": "z", "Ҙ": "Z",
        "ҫ": "s", "Ҫ": "S",
        "ғ": "ğ", "Ғ": "Ğ", "қ": "q", "Қ": "Q",
        "ң": "ñ", "Ң": "Ñ", "ő": "ö", "Ő": "Ö",
        "ҡ": "q", "Ҡ": "Q",   # Bashkir q
        "в": "v", "В": "V",   # Bashkir В = /v/ (has both V and W per CTA)
        "а": "a", "А": "A", "б": "b", "Б": "B", "г": "g", "Г": "G",
        "д": "d", "Д": "D", "е": "e", "Е": "E", "ё": "yo", "Ё": "Yo",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E",
        "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya", "ъ": "", "ь": "",
        "ү": "ü", "Ү": "Ü",
    },

    # Crimean Tatar Latin → CTS (already very close)
    "crh_Latn_to_CTS": {
        "ts": "ts", "Ts": "Ts",
        "ğ": "ğ", "Ğ": "Ğ", "q": "q", "Q": "Q",
        "ñ": "ñ", "Ñ": "Ñ", "c": "c", "C": "C",
        "ç": "ç", "Ç": "Ç", "ş": "ş", "Ş": "Ş",
        "İ": "İ", "i": "i", "ı": "ı", "I": "I",
        "a": "a", "A": "A", "b": "b", "B": "B", "d": "d", "D": "D",
        "e": "e", "E": "E", "f": "f", "F": "F", "g": "g", "G": "G",
        "h": "h", "H": "H", "j": "j", "J": "J", "k": "k", "K": "K",
        "l": "l", "L": "L", "m": "m", "M": "M", "n": "n", "N": "N",
        "o": "o", "O": "O", "ö": "ö", "Ö": "Ö", "p": "p", "P": "P",
        "r": "r", "R": "R", "s": "s", "S": "S", "t": "t", "T": "T",
        "u": "u", "U": "U", "ü": "ü", "Ü": "Ü", "v": "v", "V": "V",
        "y": "y", "Y": "Y", "z": "z", "Z": "Z",
    },

    # Crimean Tatar Cyrillic → CTS
    "crh_Cyrl_to_CTS": {
        "гъ": "ğ", "Гъ": "Ğ", "дж": "c", "Дж": "C",
        "къ": "q", "Къ": "Q", "нъ": "ñ", "Нъ": "Ñ",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "j", "Ж": "J", "ц": "ts", "Ц": "Ts",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "h", "Х": "H",
        "э": "e", "Э": "E", "ы": "ı", "Ы": "I", "ъ": "", "ь": "",
        "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
    },

    # Chuvash Cyrillic → CTS
    "chv_Cyrl_to_CTS": {
        # Cyrillic ӑ/ӗ (U+04D1/U+04D7) as used in Chuvash text
        "ӑ": "ä", "Ӑ": "Ä", "ӗ": "e", "Ӗ": "E",
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ă": "ä", "Ă": "Ä",   # Chuvash reduced a → ä
        "ĕ": "e", "Ĕ": "E",   # Chuvash reduced e → e
        "ҫ": "ş", "Ҫ": "Ş",   # Chuvash palatal sibilant → ş
        "ӳ": "ü", "Ӳ": "Ü",   # Chuvash ü
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "з": "z", "З": "Z", "и": "i", "И": "İ",
        "й": "y", "Й": "Y", "к": "k", "К": "K", "л": "l", "Л": "L",
        "м": "m", "М": "M", "н": "n", "Н": "N", "о": "o", "О": "O",
        "п": "p", "П": "P", "р": "r", "Р": "R", "с": "s", "С": "S",
        "т": "t", "Т": "T", "у": "u", "У": "U", "ф": "f", "Ф": "F",
        "х": "x", "Х": "X", "ы": "ı", "Ы": "I", "э": "e", "Э": "E",
        "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya", "ъ": "", "ь": "",
    },

    # Sakha (Yakut) Cyrillic → CTS
    "sah_Cyrl_to_CTS": {
        "һ": "h", "Һ": "H",
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ҕ": "ğ", "Ҕ": "Ğ",   # Sakha voiced uvular fricative
        "ҥ": "ñ", "Ҥ": "Ñ",   # Sakha ng
        "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "з": "z", "З": "Z", "и": "i", "И": "İ",
        "й": "y", "Й": "Y", "к": "k", "К": "K", "л": "l", "Л": "L",
        "м": "m", "М": "M", "н": "n", "Н": "N", "о": "o", "О": "O",
        "п": "p", "П": "P", "р": "r", "Р": "R", "с": "s", "С": "S",
        "т": "t", "Т": "T", "у": "u", "У": "U", "ф": "f", "Ф": "F",
        "х": "x", "Х": "X", "ы": "ı", "Ы": "I", "э": "e", "Э": "E",
        "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya", "ъ": "", "ь": "",
    },

    # Karakalpak Latin → CTS
    "kaa_Latn_to_CTS": {
        "shch": "şç", "Shch": "Şç",
        "sh": "ş", "Sh": "Ş", "SH": "Ş",
        "ch": "ç", "Ch": "Ç", "CH": "Ç",
        "ts": "ts", "Ts": "Ts",
        "á": "ä", "Á": "Ä", "ǵ": "ğ", "Ǵ": "Ğ",
        "ń": "ñ", "Ń": "Ñ", "ó": "ö", "Ó": "Ö",
        "ú": "ü", "Ú": "Ü", "í": "ı", "Í": "I", "ı": "ı",
        "q": "q", "Q": "Q", "h": "h", "H": "H",
        "a": "a", "A": "A", "b": "b", "B": "B", "d": "d", "D": "D",
        "e": "e", "E": "E", "f": "f", "F": "F", "g": "g", "G": "G",
        "i": "i", "I": "İ", "j": "c", "J": "C", "k": "k", "K": "K",
        "l": "l", "L": "L", "m": "m", "M": "M", "n": "n", "N": "N",
        "o": "o", "O": "O", "p": "p", "P": "P", "r": "r", "R": "R",
        "s": "s", "S": "S", "t": "t", "T": "T", "u": "u", "U": "U",
        "v": "v", "V": "V", "w": "w", "W": "W", "x": "x", "X": "X",
        "y": "y", "Y": "Y", "z": "z", "Z": "Z",
    },

    # Karakalpak Cyrillic → CTS
    "kaa_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ә": "ä", "Ә": "Ä", "ғ": "ğ", "Ғ": "Ğ", "қ": "q", "Қ": "Q",
        "ң": "ñ", "Ң": "Ñ", "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "ў": "w", "Ў": "W", "ҳ": "h", "Ҳ": "H",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Gagauz Latin → CTS
    "gag_Latn_to_CTS": {
        "ä": "ä", "Ä": "Ä", "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ş": "ş", "Ş": "Ş", "ç": "ç", "Ç": "Ç",
        # Gagauz has no Ğ per CTA
        "ţ": "ts", "Ţ": "Ts",   # Gagauz Ţ = /ts/ → CTS Ț
        "İ": "İ", "i": "i", "ı": "ı", "I": "I",
        "â": "a", "Â": "A",   # circumflex variants in Gagauz
        "î": "i", "Î": "İ",
        "û": "u", "Û": "U",
        "a": "a", "A": "A", "b": "b", "B": "B", "c": "c", "C": "C",
        "d": "d", "D": "D", "e": "e", "E": "E", "f": "f", "F": "F",
        "g": "g", "G": "G", "h": "h", "H": "H", "j": "j", "J": "J",
        "k": "k", "K": "K", "l": "l", "L": "L", "m": "m", "M": "M",
        "n": "n", "N": "N", "o": "o", "O": "O", "p": "p", "P": "P",
        "r": "r", "R": "R", "s": "s", "S": "S", "t": "t", "T": "T",
        "u": "u", "U": "U", "v": "v", "V": "V", "y": "y", "Y": "Y",
        "z": "z", "Z": "Z",
    },

    # Nogai Cyrillic → CTS
    "nog_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ä": "ä", "Ä": "Ä", "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ң": "ñ", "Ң": "Ñ", "ğ": "ğ", "Ğ": "Ğ", "қ": "q", "Қ": "Q",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "w", "В": "W",   # Nogai has W not V
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Kumyk Cyrillic → CTS
    "kum_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ä": "ä", "Ä": "Ä", "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ң": "ñ", "Ң": "Ñ", "ğ": "ğ", "Ğ": "Ğ", "қ": "q", "Қ": "Q",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "w", "В": "W",   # Kumyk has W not V
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Karachay-Balkar Cyrillic → CTS
    "krc_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ä": "ä", "Ä": "Ä", "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ң": "ñ", "Ң": "Ñ", "ğ": "ğ", "Ğ": "Ğ", "қ": "q", "Қ": "Q",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Altai Cyrillic → CTS
    "alt_Cyrl_to_CTS": {
        "ј": "c", "Ј": "C",   # Altai ј (U+0458) = /dʒ/
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "дж": "c", "Дж": "C",   # Altai /dʒ/ digraph
        "ж": "j", "Ж": "J",     # Altai ж = /ʒ/
        "ц": "ts", "Ц": "Ts",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ӓ": "ä", "Ӓ": "Ä",   # Altai ä
        "ӧ": "ö", "Ӧ": "Ö",   # Altai ö
        "ӱ": "ü", "Ӱ": "Ü",   # Altai ü
        "ҥ": "ñ", "Ҥ": "Ñ",   # Altai ng
        "й": "y", "Й": "Y",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "к": "k", "К": "K",
        "л": "l", "Л": "L", "м": "m", "М": "M", "н": "n", "Н": "N",
        "о": "o", "О": "O", "п": "p", "П": "P", "р": "r", "Р": "R",
        "с": "s", "С": "S", "т": "t", "Т": "T", "у": "u", "У": "U",
        "ф": "f", "Ф": "F", "х": "x", "Х": "X", "ы": "ı", "Ы": "I",
        "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Tuvan Cyrillic → CTS
    "tyv_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "ң": "ñ", "Ң": "Ñ",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Khakas Cyrillic → CTS
    "kjh_Cyrl_to_CTS": {
        # Khakas і, ҷ, and common variants ө/ү (for ӧ/ӱ), қ, ұ
        "і": "i", "І": "İ", "ҷ": "ç", "Ҷ": "Ç",
        "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "қ": "q", "Қ": "Q", "ұ": "u", "Ұ": "U",
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ӧ": "ö", "Ӧ": "Ö",   # Khakas ö
        "ӱ": "ü", "Ӱ": "Ü",   # Khakas ü
        "ң": "ñ", "Ң": "Ñ",
        "ғ": "ğ", "Ғ": "Ğ",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Ottoman Turkish Latin (academic) → CTS
    "ota_Latn_to_CTS": {
        "ch": "ç", "Ch": "Ç", "CH": "Ç",
        "sh": "ş", "Sh": "Ş", "SH": "Ş",
        "gh": "ğ", "Gh": "Ğ", "GH": "Ğ",
        "ç": "ç", "Ç": "Ç", "ş": "ş", "Ş": "Ş",
        "ğ": "ğ", "Ğ": "Ğ", "ö": "ö", "Ö": "Ö",
        "ü": "ü", "Ü": "Ü", "ı": "ı", "İ": "İ", "i": "i",
        "a": "a", "A": "A", "b": "b", "B": "B", "c": "c", "C": "C",
        "d": "d", "D": "D", "e": "e", "E": "E", "f": "f", "F": "F",
        "g": "g", "G": "G", "h": "h", "H": "H", "j": "j", "J": "J",
        "k": "k", "K": "K", "l": "l", "L": "L", "m": "m", "M": "M",
        "n": "n", "N": "N", "o": "o", "O": "O", "p": "p", "P": "P",
        "r": "r", "R": "R", "s": "s", "S": "S", "t": "t", "T": "T",
        "u": "u", "U": "U", "v": "v", "V": "V", "y": "y", "Y": "Y",
        "z": "z", "Z": "Z",
    },

    # Khalaj Latin → CTS (klj; Khalaj is an archaic Turkic language)
    "klj_Latn_to_CTS": {
        "ä": "ä", "Ä": "Ä", "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ş": "ş", "Ş": "Ş", "ç": "ç", "Ç": "Ç", "ğ": "ğ", "Ğ": "Ğ",
        "ñ": "ñ", "Ñ": "Ñ", "q": "q", "Q": "Q", "x": "x", "X": "X",
        "İ": "İ", "i": "i", "ı": "ı", "I": "I",
        "a": "a", "A": "A", "b": "b", "B": "B", "c": "c", "C": "C",
        "d": "d", "D": "D", "e": "e", "E": "E", "f": "f", "F": "F",
        "g": "g", "G": "G", "h": "h", "H": "H", "j": "j", "J": "J",
        "k": "k", "K": "K", "l": "l", "L": "L", "m": "m", "M": "M",
        "n": "n", "N": "N", "o": "o", "O": "O", "p": "p", "P": "P",
        "r": "r", "R": "R", "s": "s", "S": "S", "t": "t", "T": "T",
        "u": "u", "U": "U", "v": "v", "V": "V", "y": "y", "Y": "Y",
        "z": "z", "Z": "Z",
    },
}
