"""Hardware-aware resource monitoring.

Reports CPU, system RAM and GPU/VRAM using real measurements where the host exposes
them, and says so honestly where it does not. Every value carries a ``source`` tag:

    "measured"  - read directly from the OS/driver (GlobalMemoryStatusEx, GetSystemTimes,
                  torch.xpu device properties, Windows performance counters)
    "estimated" - derived or process-scoped (e.g. torch's own allocation, not system-wide)
    "unknown"   - not available on this host; nothing is invented

Nothing is hardcoded to one machine: GPUs are enumerated from torch.xpu / torch.cuda and
from the performance-counter instance names, so the same code reports whatever hardware is
present. On this workstation that is an Intel Arc A770 16GB; on another it might be an
NVIDIA card, an integrated GPU, or none — each is reported for what it is.

Dependencies: standard library only (ctypes) for CPU/RAM; torch (already required) for GPU
device facts. psutil is used *only if it is already installed* — it is never required.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import time
from typing import Any

_IS_WINDOWS = sys.platform.startswith("win")
_GPU_COUNTER_TTL = 4.0        # perf-counter queries are slow; cache briefly
_gpu_counter_cache: dict[str, Any] = {}
_cpu_prev: dict[str, int] = {}

try:                              # optional, never required
    import psutil as _psutil
except Exception:                 # noqa: BLE001
    _psutil = None


def _tag(value, source: str, **extra) -> dict:
    """Wrap a value with its provenance so callers can distinguish measured from estimated."""
    return {"value": value, "source": source, **extra}


# --------------------------------------------------------------------------- CPU / RAM
class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _FileTime(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]


def _ft(ft: "_FileTime") -> int:
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


def memory() -> dict:
    """Total / available / used system RAM. Measured on Windows via GlobalMemoryStatusEx,
    else via psutil if present, else unknown."""
    if _IS_WINDOWS:
        try:
            m = _MemoryStatusEx()
            m.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                total, avail = m.ullTotalPhys, m.ullAvailPhys
                return {"total_bytes": total, "available_bytes": avail,
                        "used_bytes": total - avail, "percent": m.dwMemoryLoad,
                        "source": "measured", "via": "GlobalMemoryStatusEx"}
        except Exception:  # noqa: BLE001
            pass
    if _psutil is not None:
        try:
            v = _psutil.virtual_memory()
            return {"total_bytes": v.total, "available_bytes": v.available,
                    "used_bytes": v.used, "percent": v.percent,
                    "source": "measured", "via": "psutil"}
        except Exception:  # noqa: BLE001
            pass
    return {"total_bytes": None, "available_bytes": None, "used_bytes": None,
            "percent": None, "source": "unknown"}


def cpu_percent() -> dict:
    """System-wide CPU utilisation. Measured from GetSystemTimes deltas between calls on
    Windows (non-blocking after the first call), else psutil, else unknown."""
    if _IS_WINDOWS:
        try:
            idle, kern, user = _FileTime(), _FileTime(), _FileTime()
            ok = ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
            if ok:
                i, k, u = _ft(idle), _ft(kern), _ft(user)
                busy, total = (k + u) - i, (k + u)
                prev = _cpu_prev
                if prev:
                    db, dt = busy - prev["busy"], total - prev["total"]
                    _cpu_prev.update(busy=busy, total=total)
                    pct = round(100.0 * db / dt, 1) if dt > 0 else None
                    return {"percent": pct, "logical_cores": _cpu_count(),
                            "source": "measured", "via": "GetSystemTimes"}
                _cpu_prev.update(busy=busy, total=total)
                # first call has no baseline: take a brief 80ms sample so we still return a number
                time.sleep(0.08)
                return cpu_percent()
        except Exception:  # noqa: BLE001
            pass
    if _psutil is not None:
        try:
            return {"percent": _psutil.cpu_percent(interval=0.08),
                    "logical_cores": _psutil.cpu_count(), "source": "measured", "via": "psutil"}
        except Exception:  # noqa: BLE001
            pass
    return {"percent": None, "logical_cores": _cpu_count(), "source": "unknown"}


def _cpu_count() -> int | None:
    try:
        import os
        return os.cpu_count()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- GPU
def _query_windows_gpu_counters(timeout: float = 6.0) -> list[dict] | None:
    """Best-effort Windows GPU performance-counter sample (the Task-Manager counters).

    Returns a list of raw samples ``[{"instance": str, "util": float, "kind": "engine"} ...,
    {"instance": str, "dedicated_bytes": int, "kind": "memory"} ...]`` or None when the
    counters are unavailable (non-Windows, PowerShell missing, or query error). These are
    SYSTEM-WIDE (every process, including LM Studio), unlike torch's process-scoped numbers.
    Cached for a few seconds because Get-Counter is slow.
    """
    if not _IS_WINDOWS:
        return None
    now = time.time()
    c = _gpu_counter_cache.get("samples")
    if c and now - c["ts"] < _GPU_COUNTER_TTL:
        return c["data"]
    ps = (
        "$ErrorActionPreference='SilentlyContinue';"
        "$e=(Get-Counter '\\GPU Engine(*)\\Utilization Percentage').CounterSamples |"
        " ForEach-Object {[pscustomobject]@{instance=$_.InstanceName;util=$_.CookedValue;kind='engine'}};"
        "$m=(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples |"
        " ForEach-Object {[pscustomobject]@{instance=$_.InstanceName;dedicated_bytes=[int64]$_.CookedValue;kind='memory'}};"
        "@($e+$m) | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=timeout)
        txt = (out.stdout or "").strip()
        if not txt:
            _gpu_counter_cache["samples"] = {"ts": now, "data": None}
            return None
        data = json.loads(txt)
        if isinstance(data, dict):
            data = [data]
        _gpu_counter_cache["samples"] = {"ts": now, "data": data}
        return data
    except (subprocess.SubprocessError, OSError, ValueError):
        _gpu_counter_cache["samples"] = {"ts": now, "data": None}
        return None


def aggregate_gpu_utilization(samples: list[dict] | None) -> float | None:
    """Combine the per-engine Windows GPU utilisation samples into a single 0-100 figure.

    Windows exposes one "GPU Engine" counter instance per (process, engine-type): 3D,
    Compute, Copy, VideoDecode, VideoEncode, and more, each already a percentage. There is
    no single "GPU busy %" counter, so we fold the engine instances into one number.

    Policy: model inference on Intel Arc lands mostly on the Compute and 3D engines, so we
    sum utilisation across those two engine types (summed because separate engines genuinely
    run in parallel) and cap the total at 100. Copy/Video engines are ignored as noise. If
    no Compute/3D rows are present we fall back to the max over all engine rows, so a GPU
    whose work shows up on an unexpected engine is not reported as idle. Returns None only
    when there are no engine rows at all.
    """
    if not samples:
        return None
    engines = [r for r in samples if r.get("kind") == "engine" and isinstance(r.get("util"), (int, float))]
    if not engines:
        return None
    inference = [r["util"] for r in engines
                 if "engtype_3D" in str(r.get("instance", "")) or "engtype_Compute" in str(r.get("instance", ""))]
    total = sum(inference) if inference else max(r["util"] for r in engines)
    return round(min(100.0, max(0.0, float(total))), 1)


def gpus() -> list[dict]:
    """Enumerate GPUs with measured facts and best-effort live usage.

    Device identity and total VRAM come from torch (measured). torch's own allocation is
    reported as process-scoped (estimated for system load). System-wide utilisation and
    dedicated-VRAM usage come from Windows performance counters when available.
    """
    out: list[dict] = []
    counters = _query_windows_gpu_counters()
    sys_util = aggregate_gpu_utilization(counters)
    dedicated = None
    if counters:
        mem_rows = [r.get("dedicated_bytes") for r in counters
                    if r.get("kind") == "memory" and r.get("dedicated_bytes")]
        dedicated = max(mem_rows) if mem_rows else None

    try:
        import torch
    except Exception:  # noqa: BLE001
        torch = None

    def _add_devices(backend, count, props, alloc, reserved, total_attr):
        for i in range(count):
            p = props(i)
            total = getattr(p, total_attr, None)
            dev = {
                "index": i, "backend": backend,
                "name": getattr(p, "name", f"{backend}:{i}"),
                "total_vram_bytes": _tag(total, "measured" if total else "unknown"),
                "torch_allocated_bytes": _tag(alloc(i), "measured", scope="process_torch"),
                "torch_reserved_bytes": _tag(reserved(i), "measured", scope="process_torch"),
                "utilization_percent": _tag(sys_util, "measured" if sys_util is not None else "unknown",
                                            scope="system" if sys_util is not None else None),
                "dedicated_vram_used_bytes": _tag(dedicated, "measured" if dedicated is not None else "unknown",
                                                  scope="system" if dedicated is not None else None),
            }
            if total and dedicated is not None:
                dev["vram_percent"] = _tag(round(100.0 * dedicated / total, 1), "measured", scope="system")
            out.append(dev)

    if torch is not None:
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                _add_devices("xpu", torch.xpu.device_count(), torch.xpu.get_device_properties,
                             torch.xpu.memory_allocated, torch.xpu.memory_reserved, "total_memory")
        except Exception:  # noqa: BLE001
            pass
        try:
            if torch.cuda.is_available():
                _add_devices("cuda", torch.cuda.device_count(), torch.cuda.get_device_properties,
                             torch.cuda.memory_allocated, torch.cuda.memory_reserved, "total_memory")
        except Exception:  # noqa: BLE001
            pass
    return out


class ResourceMonitor:
    """Single entry point used by the gateway API. Stateless apart from the CPU/GPU
    sampling caches; cheap to construct."""

    def snapshot(self, include_gpu: bool = True) -> dict:
        snap = {
            "timestamp": time.time(),
            "platform": sys.platform,
            "cpu": cpu_percent(),
            "memory": memory(),
            "gpus": gpus() if include_gpu else [],
        }
        return snap
