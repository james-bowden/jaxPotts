"""Tests for the resource sampler and timing helpers."""

from __future__ import annotations

import time

from jaxpotts import profiling


def test_sampler_collects_samples():
    with profiling.ResourceSampler(interval=0.02) as s:
        time.sleep(0.2)
    # ~10 samples expected in 0.2s at 20ms; allow slack.
    assert len(s.result.t) >= 3
    assert s.result.duration() > 0.1
    # Summaries do not raise even if NVML/psutil are unavailable.
    _ = s.result.peak_gpu_mem_mb()
    _ = s.result.mean_gpu_util()
    _ = s.result.mean_cpu_util()


def test_time_median():
    calls = {"n": 0}

    def work():
        calls["n"] += 1
        time.sleep(0.01)

    r = profiling.time_median(work, n=3, warmup=1)
    assert calls["n"] == 4  # 1 warmup + 3 timed
    assert r["median"] >= 0.005
    assert "first_with_compile" in r
    assert len(r["times"]) == 3


def test_sampler_is_reusable_without_gpu(monkeypatch):
    # Force the NVML path to be unavailable; sampler should still work.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pynvml":
            raise ImportError("no nvml")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with profiling.ResourceSampler(interval=0.02) as s:
        time.sleep(0.1)
    assert len(s.result.t) >= 2
