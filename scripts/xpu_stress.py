"""XPU stress test: run the training-step patterns many times in fresh processes and count failures.

Patterns (all with a per-epoch torch.xpu.synchronize() so failures are attributed to the epoch):
  original      data resident on XPU, CPU randperm moved to XPU, X[idx] advanced indexing per batch,
                float(loss) per batch  (the loop as it was when the crashes were seen)
  advanced_idx  data resident on XPU, CPU randperm -> per-batch index tensor on XPU, X[idx]; loss summed
                on device, one .item() per epoch  (variation of original)
  index_select  data resident on XPU, CPU randperm -> per-batch index tensor on XPU, torch.index_select;
                loss summed on device, one .item() per epoch  (the FIX)
  slice         data shuffled on CPU, moved once per epoch, contiguous slices (interim workaround)

Usage:
  python scripts/xpu_stress.py [--procs 4] [--epochs 60] [--patterns original,advanced_idx,index_select,slice]
  python scripts/xpu_stress.py --under-load  # starts LM Studio load generator, runs test, stops it
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

CHILD = r'''
import json, time, traceback, torch
dev, pattern, epochs = "xpu", "{pattern}", {epochs}
torch.manual_seed({seed})
N, D, B = 1000, 12, 64
Xc, Yc = torch.randn(N, D), torch.randn(N, 1)
net = torch.nn.Sequential(torch.nn.Linear(D, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                          torch.nn.Linear(64, 1)).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
lf = torch.nn.MSELoss()
g = torch.Generator().manual_seed({seed})
X, Y = Xc.to(dev), Yc.to(dev)
done, t0 = 0, time.perf_counter()
try:
    for ep in range(epochs):
        perm = torch.randperm(N, generator=g)
        if pattern == "original":
            permd = perm.to(dev)
            tot = 0.0
            for i in range(0, N, B):
                b = permd[i:i + B]
                opt.zero_grad(); loss = lf(net(X[b]), Y[b]); loss.backward(); opt.step()
                tot += float(loss) * len(b)
        elif pattern == "advanced_idx":
            # Data resident on the device; CPU permutation; only each batch's indices are moved
            # Uses X[b] advanced indexing (PROBLEMATIC under contention)
            tot = torch.zeros((), device=dev)
            for i in range(0, N, B):
                b = perm[i:i + B].to(dev, non_blocking=True)
                xb, yb = X[b], Y[b]  # Advanced indexing
                opt.zero_grad(); loss = lf(net(xb), yb); loss.backward(); opt.step()
                tot += loss.detach() * b.numel()
            tot = tot.item()
        elif pattern == "index_select":
            # Data resident on the device; CPU permutation; only each batch's indices are moved
            # Uses torch.index_select (THE FIX)
            tot = torch.zeros((), device=dev)
            for i in range(0, N, B):
                b = perm[i:i + B].to(dev, non_blocking=True)
                xb, yb = torch.index_select(X, 0, b), torch.index_select(Y, 0, b)
                opt.zero_grad(); loss = lf(net(xb), yb); loss.backward(); opt.step()
                tot += loss.detach() * b.numel()
            tot = tot.item()
        else:  # slice
            Xe, Ye = Xc[perm].to(dev), Yc[perm].to(dev)
            tot = 0.0
            for i in range(0, N, B):
                opt.zero_grad(); loss = lf(net(Xe[i:i + B]), Ye[i:i + B]); loss.backward(); opt.step()
                tot += float(loss.detach()) * len(Xe[i:i + B])
        torch.xpu.synchronize()
        assert tot == tot, "loss is NaN"
        done += 1
    print("RESULT " + json.dumps({{"ok": True, "epochs": done, "s": round(time.perf_counter() - t0, 2)}}))
except BaseException as e:
    print("RESULT " + json.dumps({{"ok": False, "epochs": done, "error": f"{{type(e).__name__}}: {{e}}"[:300],
                                   "tb": traceback.format_exc()[-1500:]}}))
'''


def check_lmstudio_status():
    """Check if LM Studio has a model loaded and its status."""
    import os
    import shutil
    lms = shutil.which("lms") or os.path.expanduser("~/.lmstudio/bin/lms")
    try:
        p = subprocess.run([lms, "ps"], capture_output=True, text=True, timeout=10)
        output = p.stdout.strip()
        if "No models" in output:
            return {"loaded": False}
        if "GENERATING" in output:
            return {"loaded": True, "status": "GENERATING"}
        if "IDLE" in output:
            return {"loaded": True, "status": "IDLE"}
        return {"loaded": True, "status": "UNKNOWN"}
    except Exception as e:
        return {"error": str(e)}


def start_lmstudio_load(seconds: int = 120):
    """Start the GPU load generator in background."""
    import os
    import shutil
    lms = shutil.which("lms") or os.path.expanduser("~/.lmstudio/bin/lms")
    
    # First ensure model is loaded
    status = check_lmstudio_status()
    if not status.get("loaded"):
        print("Loading LM Studio model...")
        subprocess.run([lms, "load", "unslothqen 4b", "-y"], capture_output=True, timeout=120)
        time.sleep(2)
    
    # Start the load generator
    print(f"Starting GPU load generator for {seconds}s...")
    load_proc = subprocess.Popen(
        [sys.executable, "scripts/gpu_load_lmstudio.py", str(seconds)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    # Wait for GENERATING status
    for _ in range(30):
        time.sleep(1)
        status = check_lmstudio_status()
        if status.get("status") == "GENERATING":
            print("LM Studio is GENERATING")
            return load_proc
    
    print(f"Warning: LM Studio status is {status.get('status', 'unknown')}")
    return load_proc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patterns", default="original,advanced_idx,index_select,slice")
    ap.add_argument("--json", default=None)
    ap.add_argument("--under-load", action="store_true", help="Start LM Studio load generator before running")
    a = ap.parse_args()
    
    load_proc = None
    if a.under_load:
        estimated_time = a.procs * len(a.patterns.split(",")) * 15  # ~15s per pattern/proc
        load_proc = start_lmstudio_load(max(120, estimated_time + 30))
    
    out = {}
    fails = 0
    try:
        for pattern in a.patterns.split(","):
            runs = []
            for k in range(a.procs):
                p = subprocess.run([sys.executable, "-c", CHILD.format(pattern=pattern, epochs=a.epochs, seed=k)],
                                   capture_output=True, text=True, timeout=900)
                res = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
                r = json.loads(res[-1][7:]) if res else {"ok": False, "error": f"process died: {p.stderr[-300:]}"}
                r["at"] = time.strftime("%H:%M:%S")
                runs.append(r)
            ok = sum(1 for r in runs if r["ok"])
            fails += len(runs) - ok
            steps = sum(r.get("epochs", 0) for r in runs) * ((1000 + 63) // 64)
            secs = [r.get("s") for r in runs if r.get("s")]
            print(f"{pattern:14s} {ok}/{len(runs)} processes clean | {steps} optimiser steps | "
                  f"s/proc {secs}" + ("" if ok == len(runs) else
                                      f" | first failure: {next(r for r in runs if not r['ok'])['error'][:140]}"))
            out[pattern] = runs
    finally:
        if load_proc:
            load_proc.terminate()
            print("\nGPU load generator stopped.")
    
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
