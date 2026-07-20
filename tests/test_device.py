"""Device profile selection tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from flow_ml.device import get_profile, resolve_device


def test_mps_profile() -> None:
    p = get_profile("mps")
    assert p.name == "mps"
    assert p.allow_mid_train_eval is False
    assert p.dataloader_workers == 0
    assert p.allow_grad_checkpointing is False
    assert p.weights_dtype_policy == "fp32_master"


def test_cuda_profile() -> None:
    p = get_profile("cuda")
    assert p.name == "cuda"
    assert p.allow_mid_train_eval is True
    assert p.dataloader_workers == 2
    assert p.allow_grad_checkpointing is True
    assert p.weights_dtype_policy == "native"


def test_cpu_profile() -> None:
    p = get_profile("cpu")
    assert p.name == "cpu"
    assert p.allow_mid_train_eval is True
    assert p.dataloader_workers == 0
    assert p.allow_grad_checkpointing is False
    assert p.weights_dtype_policy == "fp32_master"


def test_get_profile_from_device_like_object() -> None:
    """Profiles accept any object with a ``.type`` attribute (no torch needed)."""
    p = get_profile(SimpleNamespace(type="mps"))
    assert p.name == "mps"


def test_resolve_device_cpu() -> None:
    pytest.importorskip("torch")
    assert resolve_device("cpu").type == "cpu"


def test_get_profile_from_torch_device() -> None:
    torch = pytest.importorskip("torch")
    p = get_profile(torch.device("cpu"))
    assert p.name == "cpu"
