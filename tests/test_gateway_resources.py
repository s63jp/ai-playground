"""Stage A: hardware-aware resource monitor.

The GPU-utilisation aggregation is a pure function and is tested exhaustively. The OS
probes (RAM/CPU/GPU) are asserted on structure and provenance tags rather than exact
values, and the measured path is checked where the host supports it (Windows / torch.xpu).
"""
from __future__ import annotations

import sys

from fastapi.testclient import TestClient

from playground.config import load_config
from playground.dashboard import create_app
from playground.gateway.resources import ResourceMonitor, aggregate_gpu_utilization, cpu_percent, gpus, memory

_SOURCES = {"measured", "estimated", "unknown"}


# --------------------------------------------------------------------------- aggregation (pure)
def _eng(instance, util):
    return {"kind": "engine", "instance": instance, "util": util}


def test_aggregate_sums_compute_and_3d():
    samples = [_eng("pid_1_..._engtype_3D", 40.0), _eng("pid_2_..._engtype_Compute", 30.0),
               _eng("pid_3_..._engtype_Copy", 90.0),
               {"kind": "memory", "instance": "luid", "dedicated_bytes": 1000}]
    assert aggregate_gpu_utilization(samples) == 70.0      # Copy ignored


def test_aggregate_caps_at_100():
    samples = [_eng("a_engtype_Compute", 80.0), _eng("b_engtype_3D", 60.0)]
    assert aggregate_gpu_utilization(samples) == 100.0


def test_aggregate_falls_back_to_max_when_no_inference_engines():
    samples = [_eng("a_engtype_Copy", 12.0), _eng("b_engtype_VideoDecode", 55.0)]
    assert aggregate_gpu_utilization(samples) == 55.0


def test_aggregate_none_when_no_engine_rows():
    assert aggregate_gpu_utilization(None) is None
    assert aggregate_gpu_utilization([]) is None
    assert aggregate_gpu_utilization([{"kind": "memory", "dedicated_bytes": 1}]) is None


# --------------------------------------------------------------------------- RAM / CPU
def test_memory_shape_and_source():
    m = memory()
    assert m["source"] in _SOURCES
    if m["source"] == "measured":
        assert isinstance(m["total_bytes"], int) and m["total_bytes"] > 0
        assert 0 <= m["percent"] <= 100
        assert m["used_bytes"] == m["total_bytes"] - m["available_bytes"]


def test_cpu_shape_and_source():
    c = cpu_percent()
    assert c["source"] in _SOURCES
    if c["percent"] is not None:
        assert 0 <= c["percent"] <= 100


# --------------------------------------------------------------------------- GPU
def test_gpus_structure_and_tags():
    for g in gpus():
        assert {"total_vram_bytes", "utilization_percent", "torch_allocated_bytes"} <= set(g)
        for field in ("total_vram_bytes", "utilization_percent", "torch_allocated_bytes"):
            assert g[field]["source"] in _SOURCES


def test_gpus_report_xpu_when_available():
    try:
        import torch
        has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:  # noqa: BLE001
        has_xpu = False
    if not has_xpu:
        return
    devs = gpus()
    xpu = [g for g in devs if g["backend"] == "xpu"]
    assert xpu, "torch reports an XPU device but the monitor listed none"
    assert xpu[0]["total_vram_bytes"]["source"] == "measured"
    assert xpu[0]["total_vram_bytes"]["value"] > 0


# --------------------------------------------------------------------------- snapshot + endpoint
def test_snapshot_shape():
    snap = ResourceMonitor().snapshot(include_gpu=True)
    assert {"timestamp", "platform", "cpu", "memory", "gpus"} <= set(snap)
    assert isinstance(snap["gpus"], list)


def test_resources_endpoint(cfg, db):
    app = create_app(cfg, db)
    client = TestClient(app)
    r = client.get("/api/gateway/resources", params={"gpu": "true"})
    assert r.status_code == 200
    body = r.json()
    assert body["cpu"]["source"] in _SOURCES
    assert body["memory"]["source"] in _SOURCES
