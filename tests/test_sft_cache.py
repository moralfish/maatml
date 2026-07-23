"""S3: tokenized cache loads with weights_only=True (plain int-list payload)."""
from __future__ import annotations

import pytest


def test_tokenized_cache_weights_only_roundtrip(tmp_path) -> None:
    pytest.importorskip("torch")
    from maatml.training.sft_base import _load_or_build_tokenized_cache

    rows = [{"a": 1}, {"a": 2}]

    def build(_row):
        return {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "length": 3}

    cache = tmp_path / "c.pt"
    built = _load_or_build_tokenized_cache(rows, cache, build)
    assert len(built) == 2
    assert cache.is_file()

    # Second call must load from cache under weights_only=True, not rebuild. If
    # the payload were not weights_only-loadable, the code would rebuild and hit
    # this poisoned build_fn.
    def _boom(_row):
        raise AssertionError("should have loaded from cache, not rebuilt")

    loaded = _load_or_build_tokenized_cache(rows, cache, _boom)
    assert loaded == built
