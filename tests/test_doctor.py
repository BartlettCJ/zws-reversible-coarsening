from __future__ import annotations

from zws_nullity_lab.doctor import recommend_config


def test_windows_rtx3060_12gb_profile_selected() -> None:
    devices = [{"name": "NVIDIA GeForce RTX 3060", "memory_gib": 12.0}]
    assert (
        recommend_config(devices, memory_gib=64.0, platform_name="win32")
        == "configs/rtx3060_12gb_windows.yaml"
    )


def test_linux_3060_keeps_generic_vram_policy() -> None:
    devices = [{"name": "NVIDIA GeForce RTX 3060", "memory_gib": 12.0}]
    assert recommend_config(devices, platform_name="linux") == "configs/workstation.yaml"


def test_large_vram_uses_deep_profile() -> None:
    devices = [{"name": "NVIDIA GeForce RTX 5060 Ti", "memory_gib": 15.8}]
    assert recommend_config(devices, platform_name="win32") == "configs/deep.yaml"


def test_no_cuda_uses_smoke_profile() -> None:
    assert recommend_config([], platform_name="win32") == "configs/smoke.yaml"
