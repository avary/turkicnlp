"""
CoNLL-U format parsing and writing utilities.

Provides functions to read CoNLL-U files into :class:`Document` objects
and write annotated documents back to CoNLL-U format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from turkicnlp.models.document import Document, Sentence, Token, Word


def parse_conllu(text: str) -> Document:
    """Parse a CoNLL-U formatted string into a :class:`Document`.

    Args:
        text: CoNLL-U formatted text.

    Returns:
        A :class:`Document` containing the parsed sentences.
    """
    doc = Document(text="")
    sentences: list[Sentence] = []
    current_words: list[Word] = []
    current_tokens: list[Token] = []
    pending_mwt: Optional[Token] = None
    current_text: Optional[str] = None

    def _close_mwt() -> None:
        nonlocal pending_mwt
        if pending_mwt is not None:
            if pending_mwt.words:
                current_tokens.append(pending_mwt)
            pending_mwt = None

    def _flush_sentence() -> None:
        nonlocal current_words, current_tokens, current_text
        _close_mwt()  # an unclosed range must not leak into the next sentence
        if not current_words and not current_tokens:
            return
        sent_text = current_text or " ".join(w.text for w in current_words)
        sent = Sentence(text=sent_text, tokens=current_tokens, words=current_words)
        sentences.append(sent)
        current_words = []
        current_tokens = []
        current_text = None

    if text.startswith("\ufeff"):
        text = text[1:]
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line:
            _flush_sentence()
            continue
        if line.startswith("#"):
            if line.startswith("# text = "):
                current_text = line.split("=", 1)[1].strip()
            elif line.startswith("# lang = "):
                doc.lang = line.split("=", 1)[1].strip()
            elif line.startswith("# script = "):
                doc.script = line.split("=", 1)[1].strip()
            continue

        fields = line.split("\t")
        if len(fields) != 10:
            continue

        tok_id = fields[0]
        if "-" in tok_id:
            try:
                start, end = tok_id.split("-", 1)
                _close_mwt()
                pending_mwt = Token(id=(int(start), int(end)), text=fields[1])
            except ValueError:
                continue
            continue
        if "." in tok_id:
            continue

        try:
            wid = int(tok_id)
        except ValueError:
            continue

        misc = fields[9] if fields[9] != "_" else None
        script = None
        if misc:
            # Script= goes to Word.script (written back by to_conllu_line).
            rest = []
            for part in misc.split("|"):
                if part.startswith("Script=") and script is None:
                    script = part.split("=", 1)[1]
                else:
                    rest.append(part)
            misc = "|".join(rest) or None

        head: Optional[int] = None
        if fields[6] != "_":
            try:
                head = int(fields[6])
            except ValueError:
                head = None

        word = Word(
            id=wid,
            text=fields[1],
            lemma=None if fields[2] == "_" else fields[2],
            upos=None if fields[3] == "_" else fields[3],
            xpos=None if fields[4] == "_" else fields[4],
            feats=None if fields[5] == "_" else fields[5],
            head=head,
            deprel=None if fields[7] == "_" else fields[7],
            deps=None if fields[8] == "_" else fields[8],
            misc=misc,
            script=script,
        )

        if pending_mwt and not (pending_mwt.id[0] <= wid <= pending_mwt.id[1]):
            _close_mwt()
        if pending_mwt and pending_mwt.id[0] <= wid <= pending_mwt.id[1]:
            pending_mwt.words.append(word)
            if wid == pending_mwt.id[1]:
                current_tokens.append(pending_mwt)
                pending_mwt = None
        else:
            token = Token(id=(wid,), text=fields[1], words=[word])
            current_tokens.append(token)
        current_words.append(word)

    _flush_sentence()
    doc.sentences = sentences
    if not doc.text:
        doc.text = "\n\n".join(s.text for s in sentences)
    return doc


def read_conllu(path: Union[str, Path]) -> list[Document]:
    """Read a CoNLL-U file and return a list of documents.

    Args:
        path: Path to a ``.conllu`` file.

    Returns:
        List of :class:`Document` objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    docs: list[Document] = []
    if "# newdoc" in data:
        parts = data.split("# newdoc")
        for part in parts:
            if part.strip():
                docs.append(parse_conllu(part))
    else:
        docs.append(parse_conllu(data))
    return docs


def write_conllu(doc: Document, path: Union[str, Path]) -> None:
    """Write a document to a CoNLL-U file.

    Args:
        doc: The document to export.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc.to_conllu())
