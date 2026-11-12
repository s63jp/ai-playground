"""Under-load attribution, part 2: which kernel inside the forward pass fails, and does on-device randperm fail?"""
import json
import subprocess
import sys

LAYERWISE = r'''
import json, torch
dev = "xpu"; torch.manual_seed(0)
X = torch.randn(1000, 12, device=dev)
l1, l2 = torch.nn.Linear(12, 64).to(dev), torch.nn.Linear(64, 1).to(dev)
st = []
def s(n, f):
    o = f(); torch.xpu.synchronize(); st.append(n); return o
try:
    b = s("idx", lambda: torch.randperm(1000)[:64].to(dev))
    xb = s("X[b]", lambda: X[b])
    s("read xb back to cpu", lambda: xb.cpu())
    s("elementwise xb*2", lambda: xb * 2)
    s("matmul xb @ W.T (no bias)", lambda: xb @ l1.weight.T)
    h = s("linear1 (addmm)", lambda: l1(xb))
    r = s("relu", lambda: torch.relu(h))
    s("linear2", lambda: l2(r))
    print("RESULT " + json.dumps({"ok": True, "stages": st}))
except BaseException as e:
    print("RESULT " + json.dumps({"ok": False, "after": st[-1] if st else None, "error": f"{type(e).__name__}: {e}"[:160]}))
'''
RANDPERM = r'''
import json, torch
try:
    p = torch.randperm(1000, device="xpu"); torch.xpu.synchronize()
    q = (p.float() * 2).sum(); torch.xpu.synchronize()
    print("RESULT " + json.dumps({"ok": True}))
except BaseException as e:
    print("RESULT " + json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:160]}))
'''
for name, code in (("forward layer-by-layer after X[b]", LAYERWISE), ("randperm(device=xpu) then use", RANDPERM)):
    out = []
    for _ in range(3):
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
        out.append(json.loads(line[-1][7:]) if line else {"ok": False, "error": p.stderr[-160:]})
    for r in out:
        print(f"{name:36s} {'OK  ' if r['ok'] else 'FAIL'} " +
              (f"stages={r.get('stages')}" if r["ok"] else f"last completed={r.get('after')} | {r['error'][:100]}"))
