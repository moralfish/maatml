"""from_pretrained_kwargs revision passthrough."""
from __future__ import annotations

from maatml.device import get_profile
from maatml.training.load import from_pretrained_kwargs


def test_from_pretrained_kwargs_includes_revision() -> None:
    profile = get_profile("cpu")
    kwargs = from_pretrained_kwargs(profile, precision="fp32", revision="abc123")
    assert kwargs.get("revision") == "abc123"


def test_from_pretrained_kwargs_omits_revision_when_none() -> None:
    profile = get_profile("cpu")
    kwargs = from_pretrained_kwargs(profile, precision="fp32", revision=None)
    assert "revision" not in kwargs
