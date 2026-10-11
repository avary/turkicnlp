"""Tests for the GlotLID newline fix (B20)."""

from __future__ import annotations

import numpy as np
import pytest

from turkicnlp.language_id import LanguageDetection, _single_line


class _StrictFastText:
    """Mimics fastText: rejects text containing a newline."""

    def __init__(self):
        self.seen = []

    def get_sentence_vector(self, text):
        if "\n" in text:
            raise ValueError("predict processes one line at a time (remove '\\n')")
        self.seen.append(text)
        return np.ones(4, dtype=np.float32)


def _detector(mode: str) -> LanguageDetection:
    det = object.__new__(LanguageDetection)
    det.model = _StrictFastText()
    det.output_matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
    det.labels = ["__label__tur_Latn", "__label__kaz_Cyrl"]
    det.language_indices = [0, 2]
    det.predict = (
        det._predict_limit_after_softmax if mode == "after" else det._predict_limit_before_softmax
    )
    return det


@pytest.mark.parametrize("mode", ["before", "after"])
def test_newlines_replaced(mode):
    det = _detector(mode)
    labels, probs = det.predict("Merhaba\ndünya\r\nnasılsın ?", k=2)
    assert len(labels) == 2
    assert np.all(probs >= 0)
    assert det.model.seen == ["Merhaba dünya  nasılsın ?"]


def test_single_line():
    assert _single_line("a\nb\rc\x0bd") == "a b c d"
    assert _single_line("no breaks") == "no breaks"
