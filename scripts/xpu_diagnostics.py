"""XPU diagnostic harness (Intel Arc / torch XPU / Level Zero).

Each test runs in a FRESH Python process (a Level Zero DEVICE_LOST poisons the rest of a process), is
repeated several times, and calls torch.xpu.synchronize() after EVERY stage. XPU kernels run
asynchronously, so without a sync after each stage an error raised by one kernel is reported by
whichever later operation happens to synchronise next (for example a ReLU). The per-stage sync is
what attributes a failure to the operation that actually caused it.

Usage:  python scripts/xpu_diagnostics.py [--repeats 3] [--json out.json] [--only name1,name2]
Exit code: 0 if every test passed on every repeat, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

CHILD = r'''
import json, sys, traceback
import torch
dev = "xpu"
stages = []
def stage(name, fn):
    out = fn()
    torch.xpu.synchronize()          # attribute any asynchronous failure to THIS stage
    stages.append(name)
    return out
try:
    torch.manual_seed(0)
{body}
    print("RESULT " + json.dumps({{"ok": True, "stages": stages}}))
except BaseException as e:
    print("RESULT " + json.dumps({{"ok": False, "stages_passed": stages, "error_type": type(e).__name__,
                                   "error": str(e)[:400], "traceback": traceback.format_exc()[-2500:]}}))
'''

NET = ("    net = stage('build+move net', lambda: torch.nn.Sequential(torch.nn.Linear(12, 64), torch.nn.ReLU(), "
       "torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(dev))\n")
DATA = ("    X = stage('create X on xpu', lambda: torch.randn(980, 12, device=dev))\n"
        "    Y = stage('create Y on xpu', lambda: torch.randn(980, 1, device=dev))\n")

TESTS = {
    # ---------------------------------------------------------------- primitives
    "tensor_creation": "    a = stage('randn on xpu', lambda: torch.randn(256, 256, device=dev))\n"
                       "    b = stage('zeros on xpu', lambda: torch.zeros(256, device=dev))\n"
                       "    c = stage('h2d copy', lambda: torch.randn(256, 256).to(dev))\n"
                       "    d = stage('d2h copy', lambda: c.cpu())\n",
    "matmul": "    a = stage('a', lambda: torch.randn(512, 512, device=dev))\n"
              "    c = stage('matmul 512', lambda: a @ a)\n"
              "    s = stage('reduce', lambda: float(c.sum()))\n",
    "relu": "    a = stage('a', lambda: torch.randn(980, 64, device=dev))\n"
            "    r = stage('relu', lambda: torch.relu(a))\n"
            "    r2 = stage('nn.ReLU module', lambda: torch.nn.ReLU()(a))\n",
    "forward": NET + DATA + "    o = stage('forward 980', lambda: net(X))\n"
                            "    o2 = stage('forward slice 64', lambda: net(X[0:64]))\n",
    "backward": NET + DATA + "    loss = stage('forward+mse', lambda: torch.nn.functional.mse_loss(net(X[0:64]), Y[0:64]))\n"
                             "    stage('backward', lambda: loss.backward())\n"
                             "    stage('grads finite', lambda: all(bool(torch.isfinite(p.grad).all()) for p in net.parameters()))\n",
    "optimizer_step": NET + DATA +
        "    opt = torch.optim.Adam(net.parameters(), lr=3e-3)\n"
        "    w0 = net[0].weight.detach().clone()\n"
        "    def one():\n"
        "        opt.zero_grad(); l = torch.nn.functional.mse_loss(net(X[0:64]), Y[0:64]); l.backward(); opt.step(); return l\n"
        "    stage('adam step 1', one)\n"
        "    stage('adam step 2', one)\n"
        "    stage('weights changed', lambda: bool((net[0].weight.detach() - w0).abs().sum() > 0))\n",
    # ---------------------------------------------------------------- shuffling / indexing
    "randperm_on_xpu": "    p = stage('torch.randperm(device=xpu)', lambda: torch.randperm(980, device=dev))\n",
    "randperm_cpu_then_move": "    p = stage('randperm cpu', lambda: torch.randperm(980))\n"
                              "    q = stage('move perm to xpu', lambda: p.to(dev))\n",
    "index_int64_on_xpu": DATA + "    i = stage('idx int64 on xpu', lambda: torch.tensor([1, 5, 9, 33], device=dev))\n"
                                 "    g = stage('X[idx] (advanced indexing)', lambda: X[i])\n",
    "index_int32_on_xpu": DATA + "    i = stage('idx int32 on xpu', lambda: torch.tensor([1, 5, 9, 33], dtype=torch.int32, device=dev))\n"
                                 "    g = stage('X[idx int32]', lambda: X[i])\n",
    "index_select_int64": DATA + "    i = stage('idx int64 on xpu', lambda: torch.tensor([1, 5, 9, 33], device=dev))\n"
                                 "    g = stage('index_select', lambda: torch.index_select(X, 0, i))\n",
    "index_bool_mask": DATA + "    m = stage('mask', lambda: torch.arange(980, device=dev) % 7 == 0)\n"
                              "    g = stage('X[mask]', lambda: X[m])\n",
    "gather_rows_via_cpu": "    Xc = torch.randn(980, 12)\n"
                           "    p = torch.randperm(980)\n"
                           "    g = stage('index on cpu, then move', lambda: Xc[p[:64]].to(dev))\n",
    "slice_contiguous": DATA + "    g = stage('X[0:64]', lambda: X[0:64])\n"
                              "    h = stage('X[64:128].contiguous()', lambda: X[64:128].contiguous())\n",
    # ---------------------------------------------------------------- attribution check
    "index_then_relu_no_sync": DATA +
        "    i = torch.tensor([1, 5, 9, 33], device=dev)\n"
        "    g = X[i]                      # no sync here on purpose\n"
        "    r = stage('relu after unsynced index', lambda: torch.relu(g))\n",
    # ---------------------------------------------------------------- the real training loop
    "train_epoch_current_code": "    import random as _r\n"
        "    sys.path.insert(0, '.')\n"
        "    from playground.ml.nets import train\n"
        "    rows = [{'force': _r.uniform(20, 200), 'elevation_deg': _r.uniform(5, 88), 'agent_vz': _r.choice([0.0, 3.0]),\n"
        "             'object_mass': 5.0} for _ in range(400)]\n"
        "    for r in rows: r['max_height'] = 1.2 + r['force'] * 0.02 * (r['elevation_deg'] / 90) ** 2 + 0.45 * r['agent_vz']\n"
        "    spec = {'name': 'diag', 'inputs': ['force', 'elevation_deg', 'agent_vz', 'object_mass'], 'target': 'max_height'}\n"
        "    res = stage('train() 3 epochs on xpu', lambda: train(spec, rows, {'epochs': 3, 'device': 'xpu', 'patience': 99}))\n"
        "    stage('loss finite', lambda: all(h['train_loss'] == h['train_loss'] for h in res['history']))\n",
}


def env_report() -> dict:
    code = ("import torch, json; p = torch.xpu.get_device_properties(0) if torch.xpu.is_available() else None; "
            "print(json.dumps({'torch': torch.__version__, 'xpu_available': torch.xpu.is_available(), "
            "'device': p.name if p else None, 'driver_version': getattr(p, 'driver_version', None) if p else None, "
            "'total_memory_gb': round(p.total_memory / 2**30, 2) if p else None}))")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    return json.loads(line[-1]) if line else {"error": p.stderr[-400:]}


def run_one(body: str, timeout: int) -> dict:
    t = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-c", CHILD.format(body=body)], capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error_type": "Timeout", "error": f"no result within {timeout} s", "seconds": timeout}
    res = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    out = json.loads(res[-1][7:]) if res else {"ok": False, "error_type": "ProcessDied",
                                                "error": f"exit {p.returncode}: {p.stderr[-600:]}"}
    out["seconds"] = round(time.perf_counter() - t, 2)
    out["exit_code"] = p.returncode
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--timeout", type=int, default=240)
    a = ap.parse_args()
    env = env_report()
    print("environment:", json.dumps(env))
    if not env.get("xpu_available"):
        print("No XPU available: nothing to diagnose.")
        return 1
    names = a.only.split(",") if a.only else list(TESTS)
    report = {"environment": env, "repeats": a.repeats, "tests": {}}
    all_ok = True
    for name in names:
        runs = [run_one(TESTS[name], a.timeout) for _ in range(a.repeats)]
        passed = sum(1 for r in runs if r["ok"])
        all_ok &= passed == len(runs)
        report["tests"][name] = {"passed": passed, "runs": runs}
        first_fail = next((r for r in runs if not r["ok"]), None)
        detail = ""
        if first_fail:
            detail = (f"  failed at stage after {first_fail.get('stages_passed')} -> "
                      f"{first_fail.get('error_type')}: {first_fail.get('error', '')[:110]}")
        print(f"{'PASS' if passed == len(runs) else 'FAIL'} {name:28s} {passed}/{len(runs)}{detail}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print("full report (with tracebacks):", a.json)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
