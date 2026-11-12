"""Bisect, part 2: each case in a fresh process (a DEVICE_LOST poisons the rest of a process)."""
import subprocess
import sys

CASES = {
    "full batch forward": "net(X)",
    "tensor index": "net(X[torch.tensor([1, 5, 9, 33], device=dev)])",
    "slice batch": "net(X[0:64])",
    "contiguous slice": "net(X[0:64].contiguous())",
    "separate tensor of 64 rows": "net(torch.randn(64, 9, device=dev))",
    "batch of 980 again (2 calls)": "(net(X), net(X))[1]",
    "batch of 64 twice": "(net(torch.randn(64, 9, device=dev)), net(torch.randn(64, 9, device=dev)))[1]",
    "batch 980 then 64": "(net(X), net(torch.randn(64, 9, device=dev)))[1]",
    "batch 64 then 980": "(net(torch.randn(64, 9, device=dev)), net(X))[1]",
    "batch 980 then 980 different": "(net(X), net(torch.randn(980, 9, device=dev)))[1]",
    "batch 980 then 979": "(net(X), net(torch.randn(979, 9, device=dev)))[1]",
}
PRELUDE = """
import torch
dev = 'xpu'
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(9, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                          torch.nn.Linear(64, 1)).to(dev)
X = torch.randn(980, 9, device=dev)
out = {expr}
torch.xpu.synchronize()
print(tuple(out.shape))
"""
for label, expr in CASES.items():
    p = subprocess.run([sys.executable, "-c", PRELUDE.format(expr=expr)], capture_output=True, text=True, timeout=120)
    lines = [ln for ln in (p.stdout + p.stderr).splitlines() if ln.strip() and "Warning" not in ln]
    tail = lines[-1][:100] if lines else ""
    print(f"{'OK  ' if p.returncode == 0 else 'FAIL'} {label:32s} {tail}")
