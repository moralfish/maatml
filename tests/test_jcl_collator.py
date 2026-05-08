from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from flow_ml.training.jcl_validator import JclCollator, NUM_CATEGORIES  # noqa: E402


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, *, max_length, truncation, padding, return_offsets_mapping, return_tensors):
        ids: list[list[int]] = []
        masks: list[list[int]] = []
        offsets: list[list[tuple[int, int]]] = []
        for text in texts:
            row_ids = [1]
            row_off = [(0, 0)]
            for i, ch in enumerate(text[:max_length]):
                row_ids.append(1000 + ord(ch))
                row_off.append((i, i + 1))
            row_ids.append(2)
            row_off.append((0, 0))
            ids.append(row_ids)
            offsets.append(row_off)
            masks.append([1] * len(row_ids))
        max_len = max(len(r) for r in ids)
        for i in range(len(ids)):
            pad_n = max_len - len(ids[i])
            ids[i] += [0] * pad_n
            masks[i] += [0] * pad_n
            offsets[i] += [(0, 0)] * pad_n

        class _Enc(dict):
            def pop(self, k):
                v = self[k]
                del self[k]
                return v

        return _Enc(
            input_ids=torch.tensor(ids, dtype=torch.long),
            attention_mask=torch.tensor(masks, dtype=torch.long),
            offset_mapping=torch.tensor(offsets, dtype=torch.long),
        )


def test_line_labels_align_with_newlines() -> None:
    tok = _FakeTokenizer()
    coll = JclCollator(tok, max_length=64)
    text = "line1\nline2\nline3\n"
    batch = [
        {"sanitized_jcl": text, "is_valid": False, "error_category": "missing_dd", "error_line": 2},
    ]
    out = coll(batch)
    line_labels = out["line_labels"][0].tolist()
    input_ids = out["input_ids"][0].tolist()
    # Specials should be -100; tokens for "line2" should be 1; everything else 0.
    assert line_labels[0] == -100
    sep_pos = next(t for t, tid in enumerate(input_ids[1:], start=1) if tid == 2)
    assert line_labels[sep_pos] == -100
    body = line_labels[1:sep_pos]
    # Each char is one token. Trailing '\n' is grouped with the line it terminates.
    # Line 1 = "line1\n" (6 toks), line 2 = "line2\n" (6 toks), line 3 = "line3\n" (6 toks).
    assert body[0:6] == [0, 0, 0, 0, 0, 0]
    assert body[6:12] == [1, 1, 1, 1, 1, 1]
    assert body[12:18] == [0, 0, 0, 0, 0, 0]
    # Padding (label -100) only applied where attention_mask == 0; this batch has no padding.


def test_valid_sample_line_labels_all_zero_or_ignored() -> None:
    tok = _FakeTokenizer()
    coll = JclCollator(tok, max_length=64)
    batch = [
        {"sanitized_jcl": "abc\ndef\n", "is_valid": True, "error_category": None, "error_line": None},
    ]
    out = coll(batch)
    labels = out["line_labels"][0].tolist()
    assert labels[0] == -100
    body_real = [v for v, m in zip(labels, out["attention_mask"][0].tolist()) if m == 1]
    body_real = [v for v in body_real if v != -100]
    assert all(v == 0 for v in body_real)


def test_seq_and_cat_labels_indexing() -> None:
    tok = _FakeTokenizer()
    coll = JclCollator(tok, max_length=32)
    batch = [
        {"sanitized_jcl": "a", "is_valid": True, "error_category": None, "error_line": None},
        {"sanitized_jcl": "b", "is_valid": False, "error_category": "missing_dd", "error_line": 1},
        {"sanitized_jcl": "c", "is_valid": False, "error_category": "other", "error_line": 1},
    ]
    out = coll(batch)
    assert out["seq_label"].tolist() == [0, 1, 1]
    cat = out["cat_label"].tolist()
    assert cat[0] != cat[1]
    assert cat[1] != cat[2]
    assert max(cat) < NUM_CATEGORIES
