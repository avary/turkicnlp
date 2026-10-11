"""
Self-contained inference-only TurkicParser model for the turkicnlp toolkit.

This is a minimal copy of the model architecture from train-unified-models,
with all configuration values hard-coded for the Glot500 Phase 1 checkpoint.
No external config files are needed.

Architecture:
  Glot500 Backbone (frozen) -> Script Adapter -> Lang Embedding
  -> Shared BiLSTM -> POS Head + Biaffine Dep Parser
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ---------------------------------------------------------------------------
# Hard-coded constants (Glot500 Phase 1 checkpoint)
# ---------------------------------------------------------------------------

HIDDEN_DIM = 768
ADAPTER_BOTTLENECK = 128
LANG_EMBED_DIM = 64
BILSTM_HIDDEN = 400
BILSTM_LAYERS = 2
DROPOUT = 0.33
ARC_MLP_DIM = 500
LABEL_MLP_DIM = 100

BACKBONE_NAME = "cis-lmu/glot500-base"

# Maximum sequence length of the XLM-R style backbone (incl. special tokens)
MAX_SEQ_LEN = 512

logger = logging.getLogger(__name__)

# UD UPOS tagset (17 tags) — order must match training
UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
NUM_UPOS = len(UPOS_TAGS)

# UD universal dependency relations (37 base relations) — order must match training
DEPREL_TAGS = [
    "acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc",
    "ccomp", "clf", "compound", "conj", "cop", "csubj", "dep", "det",
    "discourse", "dislocated", "expl", "fixed", "flat", "goeswith",
    "iobj", "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl",
    "orphan", "parataxis", "punct", "reparandum", "root", "vocative", "xcomp",
]
NUM_DEPRELS = len(DEPREL_TAGS)

# Short code -> script mapping (used by the model internally)
LANG_TO_SCRIPT = {
    "tr": "Latn", "az": "Latn", "uz": "Latn", "tm": "Latn",
    "kk": "Cyrl", "ky": "Cyrl", "ba": "Cyrl", "tt": "Cyrl",
    "ug": "Arab", "ota": "Arab",
}

# Short code -> integer ID for language embedding lookup
LANG_ID_MAP = {
    "tr": 0, "az": 1, "uz": 2, "tm": 3, "kk": 4,
    "ky": 5, "ba": 6, "tt": 7, "ug": 8, "ota": 9,
}
NUM_LANGS = len(LANG_ID_MAP)

# ISO 639-3 (turkicnlp) -> short code (model)
ISO3_TO_SHORT = {
    "tur": "tr", "aze": "az", "uzb": "uz", "tuk": "tm", "kaz": "kk",
    "kir": "ky", "bak": "ba", "tat": "tt", "uig": "ug", "ota": "ota",
}

# Zero-shot language configuration: ISO 639-3 -> proxy info
ZEROSHOT_LANGS = {
    "kaa": {"name": "Karakalpak",     "script": "Latn", "proxy_lang": "uz"},
    "kum": {"name": "Kumyk",          "script": "Cyrl", "proxy_lang": "tt"},
    "sah": {"name": "Sakha",          "script": "Cyrl", "proxy_lang": "ba"},
    "krc": {"name": "Karachay-Balkar","script": "Cyrl", "proxy_lang": "tt"},
    "nog": {"name": "Nogai",          "script": "Cyrl", "proxy_lang": "tt"},
}

# All supported languages (trained + zero-shot)
SUPPORTED_LANGS = set(ISO3_TO_SHORT.keys()) | set(ZEROSHOT_LANGS.keys())


def resolve_lang(iso3: str) -> tuple[str, str]:
    """Resolve ISO 639-3 code to (short_code, script) for the model.

    For trained languages, returns the direct mapping.
    For zero-shot languages, returns the proxy language's short code and
    the zero-shot language's own script.

    Returns:
        (short_code, script) tuple for model routing.

    Raises:
        ValueError: If language is not supported.
    """
    if iso3 in ISO3_TO_SHORT:
        short = ISO3_TO_SHORT[iso3]
        return short, LANG_TO_SCRIPT[short]
    if iso3 in ZEROSHOT_LANGS:
        cfg = ZEROSHOT_LANGS[iso3]
        proxy_short = cfg["proxy_lang"]
        return proxy_short, cfg["script"]
    raise ValueError(
        f"Language '{iso3}' is not supported by the multilingual Glot500 model. "
        f"Supported: {sorted(SUPPORTED_LANGS)}"
    )



# ---------------------------------------------------------------------------
# Shared backbone loading
# ---------------------------------------------------------------------------

# One frozen Glot500 backbone (+ tokenizer) per (directory, device) is shared by
# every model that uses it (the POS/dep parser and the morph analyzer), so running
# both does not keep two 1 GB copies in memory.
_BACKBONE_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_BACKBONE_LOCK = threading.RLock()


def _is_local_backbone_dir(path: Optional[str]) -> bool:
    return bool(path) and (Path(path) / "config.json").exists()


def _load_backbone_uncached(hf_dir: Optional[str]) -> tuple[Any, Any]:
    """Load (tokenizer, backbone).

    ``hf_dir`` is the directory filled by ``snapshot_download(local_dir=...)``;
    it is loaded directly (no second download, works offline).  Only when the
    directory does not hold a model is the hub name used, with ``hf_dir`` as cache.
    """
    from transformers import AutoModel, AutoTokenizer
    import logging as _logging

    if _is_local_backbone_dir(hf_dir):
        source, kwargs = str(hf_dir), {}
    else:
        source, kwargs = BACKBONE_NAME, ({"cache_dir": hf_dir} if hf_dir else {})

    _loggers = [
        _logging.getLogger(name)
        for name in ("transformers.modeling_utils", "transformers.configuration_utils")
    ]
    _prev_levels = [lg.level for lg in _loggers]
    for lg in _loggers:
        lg.setLevel(_logging.ERROR)
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        # Suppress "not sharded" stderr spam from safetensors
        _devnull = os.open(os.devnull, os.O_WRONLY)
        _saved_stderr = os.dup(2)
        os.dup2(_devnull, 2)
        try:
            backbone = AutoModel.from_pretrained(source, **kwargs)
        finally:
            os.dup2(_saved_stderr, 2)
            os.close(_saved_stderr)
            os.close(_devnull)
    finally:
        for lg, lvl in zip(_loggers, _prev_levels):
            lg.setLevel(lvl)

    # Frozen backbone (inference only)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    return tokenizer, backbone


def load_backbone(
    hf_dir: Optional[str],
    device: Optional[torch.device | str] = None,
    shared: bool = True,
) -> tuple[Any, Any]:
    """Return ``(tokenizer, backbone)`` for the Glot500 backbone in *hf_dir*.

    With ``shared=True`` (default) the instance is cached per (resolved path,
    device) and reused by every model, under a lock so concurrent loads do not
    load it twice.  Use ``shared=False`` when the caller is going to overwrite
    backbone weights (a checkpoint with a fine-tuned backbone).
    """
    dev = torch.device(device) if device is not None else None
    if not shared:
        tok, bb = _load_backbone_uncached(hf_dir)
        return tok, (bb.to(dev) if dev is not None else bb)
    path_key = str(Path(hf_dir).resolve()) if hf_dir else BACKBONE_NAME
    key = (path_key, str(dev) if dev is not None else "cpu")
    with _BACKBONE_LOCK:
        if key not in _BACKBONE_CACHE:
            tok, bb = _load_backbone_uncached(hf_dir)
            if dev is not None:
                bb = bb.to(dev)
            _BACKBONE_CACHE[key] = (tok, bb)
        return _BACKBONE_CACHE[key]


def clear_backbone_cache() -> None:
    """Drop the shared backbone instances (models already built keep theirs)."""
    with _BACKBONE_LOCK:
        _BACKBONE_CACHE.clear()


def state_has_backbone(state_dict: dict) -> bool:
    """True if a checkpoint state dict carries (fine-tuned) backbone weights."""
    return any(k.startswith("backbone.") for k in state_dict)


def load_trained_state(model: nn.Module, state_dict: dict, name: str = "model") -> None:
    """``load_state_dict(strict=False)`` that reports what did not match.

    Backbone keys are expected to be missing (the frozen backbone is not saved
    in the checkpoint); every other missing or unexpected key is logged.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.startswith("backbone.")]
    if missing:
        logger.warning("%s checkpoint is missing %d weight(s): %s", name, len(missing), missing)
    if unexpected:
        logger.warning(
            "%s checkpoint has %d unexpected weight(s): %s", name, len(unexpected), unexpected
        )

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class ScriptAdapter(nn.Module):
    """Bottleneck adapter: down -> GELU -> up + residual + LayerNorm."""

    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        return self.norm(residual + x)


class LangEmbedding(nn.Module):
    """Per-language embedding concatenated with hidden states, projected back."""

    def __init__(self, num_langs: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_langs, embed_dim)
        self.proj = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, lang_ids: torch.Tensor) -> torch.Tensor:
        lang_vecs = self.embed(lang_ids)  # [B, embed_dim]
        lang_vecs = lang_vecs.unsqueeze(1).expand(-1, hidden.size(1), -1)
        concat = torch.cat([hidden, lang_vecs], dim=-1)
        return self.norm(self.proj(concat))


class SharedBiLSTM(nn.Module):
    """Multi-layer BiLSTM with dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.clamp(min=1, max=x.size(1))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=x.size(1))
        return self.dropout(output)


class POSHead(nn.Module):
    """Linear classifier for Universal POS tagging."""

    def __init__(self, input_dim: int, num_tags: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_tags)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class BiaffineScorer(nn.Module):
    """Bilinear scorer for arc and label prediction."""

    def __init__(self, in1_dim: int, in2_dim: int, out_dim: int = 1,
                 bias_x: bool = True, bias_y: bool = True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_dim = out_dim
        self.weight = nn.Parameter(
            torch.zeros(out_dim, in1_dim + int(bias_x), in2_dim + int(bias_y))
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.bias_x:
            x = torch.cat([x, x.new_ones(*x.shape[:-1], 1)], dim=-1)
        if self.bias_y:
            y = torch.cat([y, y.new_ones(*y.shape[:-1], 1)], dim=-1)
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        if self.out_dim == 1:
            return s.squeeze(1)
        else:
            return s.permute(0, 2, 3, 1)


class BiaffineParser(nn.Module):
    """Deep Biaffine Attention parser (Dozat & Manning, 2017)."""

    def __init__(self, input_dim: int, arc_mlp_dim: int, label_mlp_dim: int,
                 num_labels: int, dropout: float):
        super().__init__()
        self.arc_mlp_head = self._mlp(input_dim, arc_mlp_dim, dropout)
        self.arc_mlp_dep = self._mlp(input_dim, arc_mlp_dim, dropout)
        self.lbl_mlp_head = self._mlp(input_dim, label_mlp_dim, dropout)
        self.lbl_mlp_dep = self._mlp(input_dim, label_mlp_dim, dropout)
        self.arc_scorer = BiaffineScorer(arc_mlp_dim, arc_mlp_dim, out_dim=1)
        self.label_scorer = BiaffineScorer(label_mlp_dim, label_mlp_dim, out_dim=num_labels)

    def _mlp(self, in_dim: int, out_dim: int, dropout: float):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        arc_h = self.arc_mlp_head(x)
        arc_d = self.arc_mlp_dep(x)
        lbl_h = self.lbl_mlp_head(x)
        lbl_d = self.lbl_mlp_dep(x)
        arc_logits = self.arc_scorer(arc_d, arc_h)
        label_logits = self.label_scorer(lbl_d, lbl_h)
        return arc_logits, label_logits


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class TurkicParser(nn.Module):
    """Unified model for POS tagging and dependency parsing across Turkic languages.

    Loads the Glot500 backbone from HuggingFace and assembles adapters,
    language embeddings, BiLSTM, and task heads with hard-coded architecture.
    """

    def __init__(
        self,
        hf_cache_dir: str,
        device: Optional[torch.device | str] = None,
        share_backbone: bool = True,
    ):
        super().__init__()
        # The backbone is shared (by path + device) with other Glot500 models
        self.tokenizer, self.backbone = load_backbone(
            hf_cache_dir, device=device, shared=share_backbone,
        )

        self.adapters = nn.ModuleDict({
            "Latn": ScriptAdapter(HIDDEN_DIM, ADAPTER_BOTTLENECK),
            "Cyrl": ScriptAdapter(HIDDEN_DIM, ADAPTER_BOTTLENECK),
            "Arab": ScriptAdapter(HIDDEN_DIM, ADAPTER_BOTTLENECK),
        })

        self.lang_embed = LangEmbedding(NUM_LANGS, LANG_EMBED_DIM, HIDDEN_DIM)

        self.root_embedding = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))

        bilstm_out = BILSTM_HIDDEN * 2
        self.bilstm = SharedBiLSTM(HIDDEN_DIM, BILSTM_HIDDEN, BILSTM_LAYERS, DROPOUT)
        self.pos_head = POSHead(bilstm_out, NUM_UPOS)
        self.dep_head = BiaffineParser(
            bilstm_out, ARC_MLP_DIM, LABEL_MLP_DIM, NUM_DEPRELS, DROPOUT,
        )

    def _align_subwords_to_words(
        self, hidden: torch.Tensor, word_starts: torch.Tensor,
    ) -> torch.Tensor:
        """First-subword pooling (see :func:`align_subwords_to_words`)."""
        return align_subwords_to_words(hidden, word_starts)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: torch.Tensor,
        word_lengths: torch.Tensor,
        lang_ids: torch.Tensor,
        script: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for inference.

        Returns:
            pos_logits: [B, T_words, NUM_UPOS]
            arc_logits: [B, T_words, T_words+1]
            label_logits: [B, T_words, T_words+1, NUM_DEPRELS]
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        hidden = self._align_subwords_to_words(hidden, word_starts)
        hidden = self.adapters[script](hidden)
        hidden = self.lang_embed(hidden, lang_ids)

        B = hidden.size(0)
        root = self.root_embedding.expand(B, -1, -1)
        hidden_with_root = torch.cat([root, hidden], dim=1)
        lengths_with_root = word_lengths + 1

        lstm_out = self.bilstm(hidden_with_root, lengths_with_root)

        pos_logits = self.pos_head(lstm_out[:, 1:, :])
        arc_logits, label_logits = self.dep_head(lstm_out)
        arc_logits = arc_logits[:, 1:, :]
        label_logits = label_logits[:, 1:, :]

        return pos_logits, arc_logits, label_logits


def align_subwords_to_words(hidden: torch.Tensor, word_starts: torch.Tensor) -> torch.Tensor:
    """First-subword pooling.

    ``hidden`` is ``[N_windows, L, H]``; ``word_starts`` is ``[B, T]`` holding,
    for every word, the *flat* index ``window * L + position`` of its first
    subword (see :func:`tokenize_words`).  Returns ``[B, T, H]``.
    """
    flat = hidden.reshape(-1, hidden.size(-1))
    return flat[word_starts]


def _word_pieces(tokenizer, words: list[str]) -> list[list[int]]:
    """Subword ids of every word (without special tokens).

    Words that the tokenizer maps to no subword at all (e.g. stray control or
    format characters) get the unknown token, so every word owns at least one
    position and later words are never shifted.
    """
    pieces: Optional[list[list[int]]] = None
    if getattr(tokenizer, "is_fast", False):
        try:
            enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
            word_ids = enc.word_ids()
            pieces = [[] for _ in words]
            for tok_id, wid in zip(enc["input_ids"], word_ids):
                if wid is not None:
                    pieces[wid].append(tok_id)
        except (ValueError, AttributeError, TypeError):
            pieces = None
    if pieces is None:
        pieces = [
            list(tokenizer(w, add_special_tokens=False)["input_ids"]) if w else []
            for w in words
        ]
    unk = tokenizer.unk_token_id
    if unk is None:
        unk = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return [p if p else [unk] for p in pieces]


def _special_layout(tokenizer) -> tuple[list[int], list[int]]:
    """Special tokens placed before / after a single sequence (e.g. <s> ... </s>)."""
    sentinel = -12345
    try:
        built = tokenizer.build_inputs_with_special_tokens([sentinel])
        k = built.index(sentinel)
        return list(built[:k]), list(built[k + 1:])
    except Exception:  # noqa: BLE001
        pre = [tokenizer.cls_token_id] if tokenizer.cls_token_id is not None else []
        post = [tokenizer.sep_token_id] if tokenizer.sep_token_id is not None else []
        return pre, post


def chunk_word_pieces(pieces: list[list[int]], budget: int) -> list[list[int]]:
    """Split word indices into consecutive windows whose subwords fit *budget*.

    A single word longer than the budget gets a window of its own (and is cut
    to the budget; its first subword, the one used for prediction, is kept).
    """
    windows: list[list[int]] = []
    current: list[int] = []
    used = 0
    for wi, p in enumerate(pieces):
        n = min(len(p), budget)
        if current and used + n > budget:
            windows.append(current)
            current, used = [], 0
        current.append(wi)
        used += n
    if current:
        windows.append(current)
    return windows


def tokenize_words(
    tokenizer,
    words_list: list[list[str]],
    device: torch.device,
    max_length: int = MAX_SEQ_LEN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize pre-split word lists and compute subword-to-word alignment.

    Each word is explicitly mapped to the position of its first subword.
    Sentences longer than the backbone limit are split into several windows of
    whole words (no truncation); the backbone runs on every window and the
    word-level states are gathered back into one sequence per sentence.

    Args:
        tokenizer: HuggingFace tokenizer.
        words_list: List of word lists (one per sentence).
        device: Target device for tensors.
        max_length: Maximum window length including special tokens.

    Returns:
        ``(input_ids, attention_mask, word_starts, word_lengths)`` where
        ``input_ids``/``attention_mask`` are ``[N_windows, L]`` and
        ``word_starts`` is ``[B, T]`` with flat indices ``window * L + pos``
        (consumed by :func:`align_subwords_to_words`).
    """
    pre, post = _special_layout(tokenizer)
    limit = min(max_length, getattr(tokenizer, "model_max_length", max_length) or max_length)
    budget = max(1, limit - len(pre) - len(post))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    rows: list[list[int]] = []
    sent_positions: list[list[tuple[int, int]]] = []  # per word: (row, position)
    for words in words_list:
        pieces = _word_pieces(tokenizer, words) if words else []
        positions: list[tuple[int, int]] = []
        for window in chunk_word_pieces(pieces, budget) if pieces else []:
            row = list(pre)
            for wi in window:
                positions.append((len(rows), len(row)))
                row.extend(pieces[wi][:budget])
            row.extend(post)
            rows.append(row)
        sent_positions.append(positions)

    if not rows:  # only empty sentences: one dummy window
        rows.append(list(pre) + list(post) or [pad_id])

    L = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), L), dtype=torch.long)
    for r, row in enumerate(rows):
        input_ids[r, :len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[r, :len(row)] = 1

    max_word_len = max((len(ws) for ws in words_list), default=0)
    max_word_len = max(max_word_len, 1)
    word_starts = torch.zeros((len(words_list), max_word_len), dtype=torch.long)
    for i, positions in enumerate(sent_positions):
        for j, (r, pos) in enumerate(positions):
            word_starts[i, j] = r * L + pos

    word_lengths = torch.tensor(
        [len(ws) for ws in words_list], dtype=torch.long
    )
    return (
        input_ids.to(device),
        attention_mask.to(device),
        word_starts.to(device),
        word_lengths.to(device),
    )


# ---------------------------------------------------------------------------
# Tree decoding
# ---------------------------------------------------------------------------

_NEG = -1e9


def _find_cycle(heads: np.ndarray) -> Optional[list[int]]:
    """Return the nodes of one cycle in a head array (node 0 = root), or None."""
    n = len(heads)
    state = [0] * n  # 0 = unvisited, 1 = on current path, 2 = done
    state[0] = 2
    for start in range(1, n):
        if state[start]:
            continue
        path = []
        v = start
        while state[v] == 0:
            state[v] = 1
            path.append(v)
            v = int(heads[v])
        if state[v] == 1:  # closed a cycle
            return path[path.index(v):]
        for u in path:
            state[u] = 2
    return None


def _chu_liu_edmonds(scores: np.ndarray) -> np.ndarray:
    """Maximum spanning arborescence rooted at node 0.

    ``scores[d, h]`` is the score of head ``h`` for dependent ``d``.
    Returns the head of every node (``heads[0]`` is meaningless).
    """
    n = scores.shape[0]
    scores = scores.copy()
    np.fill_diagonal(scores, _NEG)
    scores[0, :] = _NEG
    heads = scores.argmax(axis=1)
    heads[0] = 0
    cycle = _find_cycle(heads)
    if cycle is None:
        return heads

    cyc = np.array(sorted(cycle))
    cyc_set = set(cycle)
    rest = np.array([i for i in range(n) if i not in cyc_set])  # rest[0] == 0
    m = len(rest)
    c = m  # index of the contracted node
    new = np.full((m + 1, m + 1), _NEG)
    new[:m, :m] = scores[np.ix_(rest, rest)]
    # head u outside -> cycle: gain of replacing the cycle arc of the entered node
    enter = scores[np.ix_(cyc, rest)] - scores[cyc, heads[cyc]][:, None]
    enter_best = enter.argmax(axis=0)
    new[c, :m] = enter.max(axis=0)
    # cycle node -> dependent d outside
    leave = scores[np.ix_(rest, cyc)]
    leave_best = leave.argmax(axis=1)
    new[:m, c] = leave.max(axis=1)
    new[0, :] = _NEG

    sub = _chu_liu_edmonds(new)
    result = heads.copy()
    for i in range(1, m):
        h = int(sub[i])
        result[rest[i]] = rest[h] if h != c else cyc[leave_best[i]]
    u = int(sub[c])
    result[cyc[enter_best[u]]] = rest[u]
    return result


def decode_tree(arc_scores: Any) -> list[int]:
    """Decode a well-formed dependency tree from arc scores.

    Args:
        arc_scores: ``[n, n + 1]`` scores (rows = words 1..n, columns = heads
            0..n with 0 = ROOT), preferably log-probabilities.

    Returns:
        ``n`` heads (0 = ROOT): the highest-scoring tree with no self-loops,
        no cycles and exactly one word attached to ROOT (Chu-Liu/Edmonds).
        The single-root constraint is enforced exactly by giving every ROOT
        arc the same large penalty, so the optimum uses as few ROOT arcs as
        possible (one) and is otherwise unchanged.
    """
    if isinstance(arc_scores, torch.Tensor):
        arc_scores = arc_scores.detach().double().cpu().numpy()
    arc = np.asarray(arc_scores, dtype=np.float64)[:, : arc_scores.shape[0] + 1]
    n = arc.shape[0]
    if n == 0:
        return []
    finite = np.isfinite(arc)
    arc = np.where(finite, np.clip(arc, -1e4, 1e4), _NEG)
    span = float(arc[finite].max() - arc[finite].min()) if finite.any() else 0.0
    penalty = (span + 1.0) * (n + 1)
    scores = np.full((n + 1, n + 1), _NEG)
    scores[1:, :] = arc
    scores[1:, 0] = np.where(finite[:, 0], scores[1:, 0] - penalty, _NEG)
    heads = _chu_liu_edmonds(scores)
    return [int(h) for h in heads[1:]]
