from __future__ import annotations

import pytest

from flow_ml.data.schemas import ErrorCategory, JclSample
from flow_ml.data.synthetic.jcl_generator import generate_corpus


CATEGORIES = [c for c in ErrorCategory if c is not ErrorCategory.none]


@pytest.mark.parametrize("category", CATEGORIES)
def test_generator_produces_valid_sample_per_category(category: ErrorCategory) -> None:
    samples = list(
        generate_corpus(
            seed=42,
            n_per_class={category: 5},
            n_valid=0,
        )
    )
    assert len(samples) == 5
    for s in samples:
        assert isinstance(s, JclSample)
        assert s.is_valid is False
        assert s.error_category is category
        assert s.error_line is not None and s.error_line >= 1
        line_count = s.sanitized_jcl.count("\n")
        assert s.error_line <= line_count + 1


def test_generator_produces_valid_samples() -> None:
    samples = list(generate_corpus(seed=42, n_per_class={}, n_valid=10))
    assert len(samples) == 10
    for s in samples:
        assert s.is_valid is True
        assert s.error_category is ErrorCategory.none
        assert s.error_line is None
        assert s.error_column is None


def test_generator_is_deterministic_with_fixed_seed() -> None:
    a = list(
        generate_corpus(
            seed=1234,
            n_per_class={ErrorCategory.missing_dd: 10},
            n_valid=5,
        )
    )
    b = list(
        generate_corpus(
            seed=1234,
            n_per_class={ErrorCategory.missing_dd: 10},
            n_valid=5,
        )
    )
    assert [s.sample_id for s in a] == [s.sample_id for s in b]
    assert [s.sanitized_jcl for s in a] == [s.sanitized_jcl for s in b]


def test_generator_seeds_diverge() -> None:
    a = list(generate_corpus(seed=1, n_per_class={ErrorCategory.missing_dd: 10}, n_valid=0))
    b = list(generate_corpus(seed=2, n_per_class={ErrorCategory.missing_dd: 10}, n_valid=0))
    assert [s.sample_id for s in a] != [s.sample_id for s in b]
