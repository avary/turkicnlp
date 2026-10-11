"""Tests for the Glot500 parser / morph analyzer fixes (A8, B21, B22, D30).

Everything runs offline on a tiny, randomly initialised BERT-style backbone
saved to a temporary directory (hidden size 768 to match the task heads).
"""

from __future__ import annotations

import itertools
import logging
import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from turkicnlp.processors import multilingual_model as mm  # noqa: E402
from turkicnlp.processors import multilingual_morph_model as mmm  # noqa: E402

WORDS = ["kitap", "ev", "okul", "gel", "git", "bu", "bir", "ve", "çok", "güzel"]


@pytest.fixture(scope="module")
def tiny_backbone_dir(tmp_path_factory):
    from transformers import BertConfig, BertModel, BertTokenizerFast

    d = tmp_path_factory.mktemp("glot500_tiny")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS + ["##lar", "##ler", "##da", "##de"]
    vocab += list("abcçdefgğhıijklmnoöprsştuüvyz")
    vocab += ["##" + c for c in "abcçdefgğhıijklmnoöprsştuüvyz"]
    (d / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(d / "vocab.txt"), do_lower_case=False)
    tok.save_pretrained(str(d))
    torch.manual_seed(0)
    cfg = BertConfig(
        vocab_size=len(vocab), hidden_size=mm.HIDDEN_DIM, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=512,
    )
    BertModel(cfg).save_pretrained(str(d))
    return d


@pytest.fixture(autouse=True)
def _fresh_backbone_cache():
    mm.clear_backbone_cache()
    yield
    mm.clear_backbone_cache()


# ---------------------------------------------------------------------------
# A8: backbone loading from the snapshot directory, shared instance (D30)
# ---------------------------------------------------------------------------


class TestBackboneLoading:
    def test_loads_from_local_dir(self, tiny_backbone_dir, monkeypatch):
        calls = []
        from transformers import AutoModel

        orig = AutoModel.from_pretrained

        def spy(src, *a, **kw):
            calls.append((str(src), kw))
            return orig(src, *a, **kw)

        monkeypatch.setattr(AutoModel, "from_pretrained", spy)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        tok, bb = mm.load_backbone(str(tiny_backbone_dir))
        assert calls == [(str(tiny_backbone_dir), {})]
        assert not any(p.requires_grad for p in bb.parameters())

    def test_shared_between_parser_and_morph(self, tiny_backbone_dir):
        parser = mm.TurkicParser(hf_cache_dir=str(tiny_backbone_dir))
        morph = mmm.TurkicMorphAnalyzer(hf_cache_dir=str(tiny_backbone_dir), num_edit_scripts=4)
        assert parser.backbone is morph.backbone
        assert parser.tokenizer is morph.tokenizer

    def test_unshared_when_requested(self, tiny_backbone_dir):
        a = mm.TurkicParser(hf_cache_dir=str(tiny_backbone_dir))
        b = mm.TurkicParser(hf_cache_dir=str(tiny_backbone_dir), share_backbone=False)
        assert a.backbone is not b.backbone

    def test_concurrent_load_once(self, tiny_backbone_dir, monkeypatch):
        n_loads = []
        orig = mm._load_backbone_uncached

        def counting(path):
            n_loads.append(path)
            return orig(path)

        monkeypatch.setattr(mm, "_load_backbone_uncached", counting)
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(mm.load_backbone(str(tiny_backbone_dir))))
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(n_loads) == 1
        assert all(r[1] is results[0][1] for r in results)

    def test_ensure_backbone_skips_complete_dir(self, tmp_path, monkeypatch):
        from turkicnlp.processors import multilingual_backend as mb
        from turkicnlp.processors import multilingual_morph_backend as mmb

        d = tmp_path / "huggingface" / mb.BACKBONE_HF_NAME.replace("/", "--")
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"x")
        import huggingface_hub

        def boom(**kw):
            raise AssertionError("must not download")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
        assert mb._ensure_backbone(tmp_path) == d
        assert mmb._ensure_backbone(tmp_path) == d

    def test_ensure_backbone_redownloads_incomplete_dir(self, tmp_path, monkeypatch):
        from turkicnlp.processors import multilingual_backend as mb

        d = tmp_path / "huggingface" / mb.BACKBONE_HF_NAME.replace("/", "--")
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")  # weights missing (interrupted download)
        import huggingface_hub

        calls = []
        monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw))
        mb._ensure_backbone(tmp_path)
        assert calls and calls[0]["local_dir"] == str(d)


# ---------------------------------------------------------------------------
# A8: long sentences (windowing) and words without subwords
# ---------------------------------------------------------------------------


class TestTokenizeWords:
    def _check_alignment(self, tok, words, input_ids, word_starts):
        flat = input_ids.reshape(-1)
        for j, w in enumerate(words):
            ids = tok(w, add_special_tokens=False)["input_ids"] or [tok.unk_token_id]
            assert flat[word_starts[0, j]].item() == ids[0], (j, w)

    def test_long_sentence_is_windowed_not_truncated(self, tiny_backbone_dir):
        tok, _ = mm.load_backbone(str(tiny_backbone_dir))
        words = [WORDS[i % len(WORDS)] + ("lar" if i % 3 == 0 else "") for i in range(700)]
        input_ids, mask, starts, lengths = mm.tokenize_words(tok, [words], torch.device("cpu"))
        assert input_ids.size(0) > 1  # several windows
        assert input_ids.size(1) <= 512
        assert lengths.tolist() == [700]
        assert starts.shape == (1, 700)
        self._check_alignment(tok, words, input_ids, starts)
        # every window is a well-formed sequence
        for r in range(input_ids.size(0)):
            n = int(mask[r].sum())
            assert input_ids[r, 0].item() == tok.cls_token_id
            assert input_ids[r, n - 1].item() == tok.sep_token_id

    def test_small_windows(self, tiny_backbone_dir):
        tok, _ = mm.load_backbone(str(tiny_backbone_dir))
        words = ["kitaplar", "evde", "okul", "güzel", "xyzq"] * 5
        input_ids, mask, starts, _ = mm.tokenize_words(tok, [words], "cpu", max_length=8)
        assert input_ids.size(1) <= 8
        self._check_alignment(tok, words, input_ids, starts)

    def test_word_without_subwords_does_not_shift(self, tiny_backbone_dir):
        tok, _ = mm.load_backbone(str(tiny_backbone_dir))
        empty = "\u0007"  # control char: the tokenizer produces no subword
        assert tok(empty, add_special_tokens=False)["input_ids"] == []
        words = ["kitap", empty, "ev", "okul"]
        input_ids, _, starts, _ = mm.tokenize_words(tok, [words], "cpu")
        flat = input_ids.reshape(-1)
        assert flat[starts[0, 1]].item() == tok.unk_token_id
        assert flat[starts[0, 2]].item() == tok.convert_tokens_to_ids("ev")
        assert flat[starts[0, 3]].item() == tok.convert_tokens_to_ids("okul")

    def test_batch_padding(self, tiny_backbone_dir):
        tok, _ = mm.load_backbone(str(tiny_backbone_dir))
        batch = [["kitap", "ev"], ["okul"], []]
        input_ids, _, starts, lengths = mm.tokenize_words(tok, batch, "cpu")
        assert lengths.tolist() == [2, 1, 0]
        assert starts.shape == (3, 2)

    def test_forward_on_long_sentence(self, tiny_backbone_dir):
        model = mm.TurkicParser(hf_cache_dir=str(tiny_backbone_dir)).eval()
        words = [WORDS[i % len(WORDS)] for i in range(600)]
        inputs = mm.tokenize_words(model.tokenizer, [words], torch.device("cpu"))
        with torch.no_grad():
            pos, arc, lbl = model(*inputs, lang_ids=torch.tensor([0]), script="Latn")
        assert pos.shape[:2] == (1, 600)
        assert arc.shape[:2] == (1, 600)


# ---------------------------------------------------------------------------
# B21: tree decoding and weight-loading report
# ---------------------------------------------------------------------------


def _is_tree(heads):
    n = len(heads)
    if sum(1 for h in heads if h == 0) != 1:
        return False
    for d, h in enumerate(heads, start=1):
        if h == d or not 0 <= h <= n:
            return False
    for d in range(1, n + 1):  # every node reaches the root
        seen, v = set(), d
        while v != 0:
            if v in seen:
                return False
            seen.add(v)
            v = heads[v - 1]
    return True


def _brute_force_best(scores):
    n = scores.shape[0]
    best, best_s = None, None
    for heads in itertools.product(range(n + 1), repeat=n):
        if not _is_tree(list(heads)):
            continue
        s = sum(scores[d, h] for d, h in enumerate(heads))
        if best_s is None or s > best_s:
            best, best_s = list(heads), s
    return best, best_s


class TestDecodeTree:
    def test_self_loops_and_multiple_roots_fixed(self):
        # greedy argmax would give: 1->1 (self), 2->0, 3->0 (two roots)
        scores = np.full((3, 4), -5.0)
        scores[0, 1] = 10.0
        scores[1, 0] = 9.0
        scores[2, 0] = 9.0
        heads = mm.decode_tree(scores)
        assert _is_tree(heads)

    def test_cycle_broken(self):
        scores = np.full((3, 4), -5.0)
        scores[0, 2] = 5.0  # 1 <- 2
        scores[1, 1] = 5.0  # 2 <- 1  (cycle 1 <-> 2)
        scores[2, 0] = 5.0
        assert _is_tree(mm.decode_tree(scores))

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_matches_brute_force(self, n):
        rng = np.random.default_rng(n)
        for _ in range(30):
            scores = rng.normal(size=(n, n + 1))
            heads = mm.decode_tree(scores)
            assert _is_tree(heads)
            _, best_s = _brute_force_best(scores)
            s = sum(scores[d, h] for d, h in enumerate(heads))
            assert s == pytest.approx(best_s)

    def test_random_long_sentences_are_trees(self):
        rng = np.random.default_rng(0)
        for n in (10, 40, 120):
            heads = mm.decode_tree(torch.tensor(rng.normal(size=(n, n + 1)) * 5))
            assert _is_tree(heads)

    def test_decode_parse_labels_consistent(self):
        from turkicnlp.processors.multilingual_backend import decode_parse

        torch.manual_seed(0)
        n = 6
        arc = torch.randn(n, n + 1)
        lbl = torch.randn(n, n + 1, mm.NUM_DEPRELS)
        lbl[..., mm.DEPREL_TAGS.index("root")] = 100.0  # model loves "root"
        heads, deprels = decode_parse(arc, lbl)
        assert _is_tree(heads)
        for h, r in zip(heads, deprels):
            assert (r == "root") == (h == 0)

    def test_missing_and_unexpected_keys_logged(self, tiny_backbone_dir, caplog):
        model = mm.TurkicParser(hf_cache_dir=str(tiny_backbone_dir))
        sd = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
        sd.pop("pos_head.classifier.weight")
        sd["old_head.weight"] = torch.zeros(1)
        with caplog.at_level(logging.WARNING):
            mm.load_trained_state(model, sd, name="Glot500 parser")
        text = caplog.text
        assert "pos_head.classifier.weight" in text
        assert "old_head.weight" in text
        assert "backbone." not in text

    def test_state_has_backbone(self):
        assert mm.state_has_backbone({"backbone.x": 1, "pos_head.w": 2})
        assert not mm.state_has_backbone({"pos_head.w": 2})


# ---------------------------------------------------------------------------
# Managers end to end: one backbone for parser + morph, long sentence, tree
# ---------------------------------------------------------------------------


class TestManagersEndToEnd:
    @pytest.fixture
    def managers(self, tiny_backbone_dir, tmp_path, monkeypatch):
        from turkicnlp.processors import multilingual_backend as mb
        from turkicnlp.processors import multilingual_morph_backend as mmb

        torch.manual_seed(0)
        parser = mm.TurkicParser(hf_cache_dir=str(tiny_backbone_dir))
        p_ckpt = tmp_path / "parser.pt"
        torch.save({"trained_state_dict": {
            k: v for k, v in parser.state_dict().items() if not k.startswith("backbone.")
        }}, p_ckpt)
        morph = mmm.TurkicMorphAnalyzer(hf_cache_dir=str(tiny_backbone_dir), num_edit_scripts=4)
        m_ckpt = tmp_path / "morph.pt"
        torch.save({
            "trained_state_dict": {
                k: v for k, v in morph.state_dict().items() if not k.startswith("backbone.")
            },
            "edit_vocab": {"script_to_id": {"<unk>": 0, "0": 1, "-3+": 2, "-2+": 3}},
        }, m_ckpt)
        del parser, morph
        mm.clear_backbone_cache()
        monkeypatch.setattr(mb._MultilingualParserManager, "ensure_downloaded",
                            classmethod(lambda cls: (p_ckpt, tiny_backbone_dir)))
        monkeypatch.setattr(mmb._MorphAnalyzerManager, "ensure_downloaded",
                            classmethod(lambda cls: (m_ckpt, tiny_backbone_dir)))
        mb._MultilingualParserManager.clear()
        mmb._MorphAnalyzerManager.clear()
        yield mb._MultilingualParserManager, mmb._MorphAnalyzerManager
        mb._MultilingualParserManager.clear()
        mmb._MorphAnalyzerManager.clear()

    def _doc(self, words, lang="tur"):
        from turkicnlp.models.document import Document, Sentence, Word

        sent = Sentence(text=" ".join(words),
                        words=[Word(id=i + 1, text=w) for i, w in enumerate(words)])
        return Document(text=sent.text, sentences=[sent], lang=lang)

    def test_one_backbone_in_memory(self, managers):
        pm, mgr = managers
        parser, _ = pm.get_model()
        morph, _, _ = mgr.get_model()
        assert parser.backbone is morph.backbone

    def test_long_sentence_parse_is_tree(self, managers):
        pm, _ = managers
        words = [WORDS[i % len(WORDS)] for i in range(550)] + ["\u0007", "ev"]
        result = pm.run(self._doc(words))
        assert len(result["heads"][0]) == len(words)
        assert len(result["pos_tags"][0]) == len(words)
        assert _is_tree(result["heads"][0])

    def test_morph_long_sentence(self, managers):
        _, mgr = managers
        words = [WORDS[i % len(WORDS)] for i in range(530)]
        result = mgr.run(self._doc(words))
        assert len(result["lemmas"][0]) == 530
        assert len(result["feats"][0]) == 530


# ---------------------------------------------------------------------------
# B22: lemma casing
# ---------------------------------------------------------------------------


class TestLemmaCasing:
    def test_turkic_lower(self):
        assert mmm.turkic_lower("İstanbul", "tur") == "istanbul"
        assert mmm.turkic_lower("IŞIK", "tur") == "ışık"
        assert mmm.turkic_lower("IŞIK", "aze") == "ışık"
        assert mmm.turkic_lower("ISHLAR", "uzb") == "ishlar"
        assert mmm.turkic_lower("İ", "kaz") == "i"
        assert mmm.turkic_lower("İ", "uzb") == "i"  # never i + combining dot
        assert mmm.turkic_lower("Бала", "kaz") == "бала"

    def test_turkic_upper(self):
        assert mmm.turkic_upper("istanbul", "tur") == "İSTANBUL"
        assert mmm.turkic_upper("ışık", "tur") == "IŞIK"
        assert mmm.turkic_upper("ishlar", "uzb") == "ISHLAR"

    def test_apply_edit_script_turkic(self):
        assert mmm.apply_edit_script("IŞIK", "0", "tur") == "ışık"
        assert mmm.apply_edit_script("İstanbul", "0", "tur") == "istanbul"
        assert mmm.apply_edit_script("geldi", "-9+x", "tur") == "geldi"  # over-long removal
        assert mmm.apply_edit_script("geldi", "bogus", "tur") == "geldi"

    def test_predict_lemma(self):
        from turkicnlp.processors.multilingual_morph_backend import predict_lemma

        assert predict_lemma("İstanbul", "0", "tur", "PROPN") == "İstanbul"
        assert predict_lemma("İstanbul'da", "-3+", "tur", "PROPN") == "İstanbul"
        assert predict_lemma("IŞIK", "0", "tur", "NOUN") == "ışık"
        assert predict_lemma("IŞIK", "0", "tur", "PROPN") == "ışık"  # all caps -> lowercase
        assert predict_lemma("Kitaplar", "-3+", "tur", "NOUN") == "kitap"
        assert predict_lemma("Ankara", "<unk>", "tur", "PROPN") == "Ankara"
        assert predict_lemma("Ismoil", "0", "uzb", "PROPN") == "Ismoil"
        assert predict_lemma("Алматыда", "-2+", "kaz", "PROPN") == "Алматы"
        assert predict_lemma("iyi", "0", "tur", "ADJ") == "iyi"

    def test_restore_case_capitalises_when_not_prefix(self):
        assert mmm.restore_lemma_case("İzmir'e", "izmir", "tur", "PROPN") == "İzmir"
        assert mmm.restore_lemma_case("Ağrı'ya", "ağrı", "tur", "PROPN") == "Ağrı"
        assert mmm.restore_lemma_case("Gitti", "gitmek", "tur", "PROPN") == "Gitmek"
