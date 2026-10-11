"""MorphemeAnnotator — converts raw Turkic text into morpheme-boundary-annotated form.

Intended for producing training corpora for SentencePiece tokenizers.
Uses TurkicNLP's MorphemeTokenizer (hybrid Glot500 neural + Apertium HFST
+ suffix allomorph tables) as the segmentation backend.

Example::

    annotator = MorphemeAnnotator(lang="kaz")
    line = annotator.annotate_line("Мен мектептерімізден оқыдым")
    # "Мен мектеп▁тер▁іміз▁ден оқы▁ды▁м"

    annotator.annotate_file(
        input_path=Path("data/raw/fineweb2_kaz.txt"),
        output_path=Path("data/annotated/kaz_annotated.txt"),
        max_lines=20_000_000,
    )
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from turkicnlp.processors.morpheme_tokenizer import MorphemeTokenizer

BOUNDARY = "\u2581"  # ▁ — morpheme boundary marker; must match SP user_defined_symbols
# NOTE: "▁" is also SentencePiece's word-boundary symbol, so morpheme and word
# boundaries become indistinguishable after decoding. It stays the default for
# compatibility with existing models; for new models use a dedicated marker,
# e.g. MorphemeAnnotator(lang, boundary="\u2502"), and the same value for
# ``sp_tokenize_boundary`` at inference. MorphemeTokenizer caches segmentations
# per word form, so repeated words in a corpus are segmented only once.

logger = logging.getLogger(__name__)

# Characters that are very unlikely to be part of a Turkic word
_PUNCT_ONLY_RE = re.compile(r"^[^\w]+$", re.UNICODE)


def basic_clean(line: str) -> str:
    """Remove lines that are likely noise (low alpha ratio, very short)."""
    stripped = line.strip()
    if not stripped:
        return ""
    alpha_ratio = sum(c.isalpha() for c in stripped) / max(1, len(stripped))
    if alpha_ratio < 0.5:
        return ""
    if len(stripped.split()) < 3:
        return ""
    return stripped


class MorphemeAnnotator:
    """Convert raw Turkic text into morpheme-boundary-annotated form.

    Suitable for SentencePiece training.  Designed to handle
    FineWeb2-scale data (billions of words) by streaming line by line.

    Args:
        lang: ISO 639-3 language code supported by MorphemeTokenizer.
        boundary: Boundary character inserted between morpheme segments.
            Must match the ``user_defined_symbols`` value used in SP training.
        clean: If True, skip low-quality lines (low alpha ratio, very short).
    """

    def __init__(
        self,
        lang: str,
        boundary: str = BOUNDARY,
        clean: bool = False,
    ) -> None:
        self.lang = lang
        self.boundary = boundary
        self.clean = clean
        self.tok = MorphemeTokenizer(lang=lang)
        self.tok.load()
        self.stats: defaultdict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_word(self, word: str) -> str:
        """Segment a single word into boundary-marked morphemes.

        Falls back to the raw word on failure so the output is always
        valid (never raises).
        """
        if not word.strip() or len(word) <= 1:
            self.stats["skipped_short"] += 1
            return word

        # Punctuation-only tokens: pass through unchanged
        if _PUNCT_ONLY_RE.match(word):
            self.stats["skipped_punct"] += 1
            return word

        try:
            result = self.tok.segment(word)
            if len(result.segments) > 1:
                self.stats["segmented"] += 1
                return self.boundary.join(result.segments)
            else:
                self.stats["unsegmented"] += 1
                return word
        except Exception as exc:
            logger.debug("Segmentation failed for %r: %s", word, exc)
            self.stats["errors"] += 1
            return word

    def annotate_line(self, line: str) -> str:
        """Annotate all whitespace-delimited words in a single line."""
        words = line.strip().split()
        if not words:
            return ""
        return " ".join(self.annotate_word(w) for w in words)

    def annotate_file(
        self,
        input_path: Path,
        output_path: Path,
        max_lines: Optional[int] = None,
    ) -> dict:
        """Stream-annotate a corpus file and write annotated output.

        Writes an annotation stats JSON alongside the output file.

        Args:
            input_path: Plain-text corpus, one sentence/document per line.
            output_path: Destination for annotated text.
            max_lines: If set, stop after processing this many input lines.

        Returns:
            Annotation statistics dict.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats = defaultdict(int)

        with open(input_path, encoding="utf-8") as fin, open(
            output_path, "w", encoding="utf-8"
        ) as fout:
            for i, line in enumerate(tqdm(fin, desc=f"Annotating [{self.lang}]")):
                if max_lines is not None and i >= max_lines:
                    break

                if self.clean:
                    line = basic_clean(line)
                    if not line:
                        self.stats["total_lines"] += 1
                        continue

                stripped = line.strip()
                if stripped:
                    fout.write(self.annotate_line(stripped) + "\n")
                else:
                    fout.write("\n")

                self.stats["total_lines"] += 1

        stats_path = output_path.with_suffix(".annotation_stats.json")
        with open(stats_path, "w") as f:
            json.dump(dict(self.stats), f, indent=2)

        self._print_summary()
        return dict(self.stats)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        total_words = (
            self.stats["segmented"]
            + self.stats["unsegmented"]
            + self.stats["errors"]
            + self.stats["skipped_short"]
            + self.stats["skipped_punct"]
        )
        if total_words == 0:
            return
        seg_rate = self.stats["segmented"] / max(1, total_words)
        print(f"\n[{self.lang}] Annotation summary:")
        print(f"  Total lines:    {self.stats['total_lines']:>12,}")
        print(f"  Total words:    {total_words:>12,}")
        print(f"  Segmented:      {self.stats['segmented']:>12,}  ({seg_rate:.1%})")
        print(f"  Unsegmented:    {self.stats['unsegmented']:>12,}")
        print(f"  Errors:         {self.stats['errors']:>12,}")
        print("  Segmentation rate target: >80% for production languages")
