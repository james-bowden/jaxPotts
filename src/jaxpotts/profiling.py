"""Timing and GPU/CPU utilization sampling.

The :class:`ResourceSampler` context manager samples NVML GPU utilization/memory
(via ``pynvml``) and process CPU utilization (via ``psutil``) in a background
thread at a fixed interval, for use around any workload -- jaxPotts, CCMpred, or
CCMpredPy. It is deliberately tool-agnostic so the same instrument measures all
three in the notebook.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from statistics import median
from typing import Callable

import numpy as np


@dataclass
class SamplerResult:
    """Time series and summaries collected by :class:`ResourceSampler`."""

    t: list[float] = field(default_factory=list)             # seconds since start
    gpu_util: list[float] = field(default_factory=list)      # percent
    gpu_mem_mb: list[float] = field(default_factory=list)    # MiB used
    cpu_util: list[float] = field(default_factory=list)      # percent (process)
    sys_cpu_util: list[float] = field(default_factory=list)  # percent (whole machine)

    def peak_gpu_mem_mb(self) -> float:
        return max(self.gpu_mem_mb) if self.gpu_mem_mb else float("nan")

    def mean_gpu_util(self) -> float:
        return float(np.mean(self.gpu_util)) if self.gpu_util else float("nan")

    def mean_cpu_util(self) -> float:
        return float(np.mean(self.cpu_util)) if self.cpu_util else float("nan")

    def duration(self) -> float:
        return self.t[-1] if self.t else 0.0


class ResourceSampler:
    """Background NVML + psutil sampler usable as a context manager.

    Parameters
    ----------
    interval : float
        Sampling period in seconds (default 0.1 = ~100 ms).
    gpu_index : int
        NVML device index to sample (default 0).
    pid : int or None
        Process to sample CPU for (default: current process).

    Example
    -------
    >>> with ResourceSampler() as s:      # doctest: +SKIP
    ...     do_work()
    >>> s.result.peak_gpu_mem_mb()        # doctest: +SKIP
    """

    def __init__(self, interval: float = 0.1, gpu_index: int = 0, pid: int | None = None):
        self.interval = interval
        self.gpu_index = gpu_index
        self.pid = pid
        self.result = SamplerResult()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._proc = None

    def _setup(self):
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pynvml  # deprecated shim for nvidia-ml-py; the API we use is stable

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception:
            self._pynvml = None
        try:
            import psutil

            self._psutil = psutil
            self._proc = psutil.Process(self.pid)
            self._proc.cpu_percent(None)  # prime the measurement
            psutil.cpu_percent(None)      # prime system-wide measurement
        except Exception:
            self._psutil = None
            self._proc = None

    def _loop(self, t0: float):
        while not self._stop.is_set():
            now = time.perf_counter() - t0
            if self._pynvml is not None:
                try:
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                    mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used / (1024 ** 2)
                except Exception:
                    util, mem = float("nan"), float("nan")
            else:
                util, mem = float("nan"), float("nan")
            cpu = self._proc.cpu_percent(None) if self._proc is not None else float("nan")
            sys_cpu = self._psutil.cpu_percent(None) if self._psutil is not None else float("nan")
            self.result.t.append(now)
            self.result.gpu_util.append(util)
            self.result.gpu_mem_mb.append(mem)
            self.result.cpu_util.append(cpu)
            self.result.sys_cpu_util.append(sys_cpu)
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceSampler":
        self._setup()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(time.perf_counter(),), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        return False


def time_median(fn: Callable, n: int = 3, warmup: int = 1) -> dict:
    """Median wall-clock time (seconds) of ``fn`` over ``n`` runs after ``warmup`` runs.

    ``fn`` should block until its work is complete (e.g. call ``block_until_ready``).
    The warmup runs absorb JAX compilation; the returned dict also reports the first
    (compile-inclusive) time separately so it can be shown, not hidden.
    """
    t0 = time.perf_counter()
    fn()
    compile_plus_first = time.perf_counter() - t0
    for _ in range(warmup - 1):
        fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return {
        "median": median(times),
        "min": min(times),
        "max": max(times),
        "times": times,
        "first_with_compile": compile_plus_first,
    }


def jax_peak_memory_mb(device=None) -> float:
    """Peak bytes allocated on a JAX device, in MiB (``nan`` if unavailable)."""
    import jax

    dev = device or jax.local_devices()[0]
    try:
        stats = dev.memory_stats()
        peak = stats.get("peak_bytes_in_use", stats.get("bytes_in_use", float("nan")))
        return peak / (1024 ** 2)
    except Exception:
        return float("nan")
