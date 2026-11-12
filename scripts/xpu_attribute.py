"""Attribute the under-load failure: index -> forward -> backward -> step, synchronising after each stage."""
import json
import subprocess
import sys

CHILD = r'''
import json, traceback, torch
dev = "xpu"
torch.manual_seed(0)
X, Y = torch.randn(1000, 12, device=dev), torch.randn(1000, 1, device=dev)
net = torch.nn.Sequential(torch.nn.Linear(12, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                          torch.nn.Linear(64, 1)).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
lf = torch.nn.MSELoss()
stages = []
def s(name, fn):
    o = fn(); torch.xpu.synchronize(); stages.append(name); return o
try:
    for it in range({iters}):
        b = s(f"{{it}}:index to xpu", lambda: torch.randperm(1000)[:64].to(dev))
        xb = s(f"{{it}}:{gather}", lambda: {gather_expr})
        yb = s(f"{{it}}:gather Y", lambda: {gather_y})
        opt.zero_grad()
        out = s(f"{{it}}:forward", lambda: net(xb))
        loss = s(f"{{it}}:loss", lambda: lf(out, yb))
        s(f"{{it}}:backward", lambda: loss.backward())
        s(f"{{it}}:step", lambda: opt.step())
    print("RESULT " + json.dumps({{"ok": True, "n": len(stages)}}))
except BaseException as e:
    print("RESULT " + json.dumps({{"ok": False, "failed_after": stages[-1] if stages else None,
                                   "error": f"{{type(e).__name__}}: {{e}}"[:200]}}))
'''
VARIANTS = {
    "advanced index X[b]": ("X[b]", "Y[b]"),
    "index_select": ("torch.index_select(X, 0, b)", "torch.index_select(Y, 0, b)"),
    "index then .contiguous().clone()": ("X[b].contiguous().clone()", "Y[b].contiguous().clone()"),
    "slice (control)": ("X[0:64]", "Y[0:64]"),
}
for name, (gx, gy) in VARIANTS.items():
    res = []
    for k in range(3):
        p = subprocess.run([sys.executable, "-c", CHILD.format(iters=30, gather=name, gather_expr=gx, gather_y=gy)],
                           capture_output=True, text=True, timeout=300)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
        res.append(json.loads(line[-1][7:]) if line else {"ok": False, "error": p.stderr[-200:]})
    ok = sum(r["ok"] for r in res)
    fail = next((r for r in res if not r["ok"]), None)
    print(f"{name:34s} {ok}/3 clean" + (f" | last stage that completed: {fail['failed_after']} | {fail['error'][:90]}"
                                         if fail else ""))
