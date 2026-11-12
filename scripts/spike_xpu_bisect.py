"""Bisect the XPU (Arc A770) failure seen when training through playground.ml.nets."""
import torch

dev = "xpu"
print("torch", torch.__version__, torch.xpu.get_device_name(0))


def attempt(label, fn):
    try:
        out = fn()
        torch.xpu.synchronize()
        print(f"OK   {label}: {tuple(out.shape) if hasattr(out, 'shape') else out}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {label}: {str(e)[:110]}")


net = torch.nn.Sequential(torch.nn.Linear(9, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                          torch.nn.Linear(64, 1)).to(dev)
X = torch.randn(980, 9, device=dev)
attempt("full batch forward", lambda: net(X))
attempt("tensor index (cpu-made idx moved)", lambda: net(X[torch.tensor([1, 5, 9, 33], device=dev)]))
attempt("slice batch", lambda: net(X[0:64]))
attempt("index_select", lambda: net(torch.index_select(X, 0, torch.tensor(list(range(64)), device=dev))))
Xc = torch.randn(980, 9)
attempt("cpu-index then move", lambda: net(Xc[torch.tensor([1, 5, 9, 33])].to(dev)))
attempt("tensor from python list of lists", lambda: net(torch.tensor([[0.1] * 9] * 64, dtype=torch.float32, device=dev)))
attempt("built on cpu then .to(xpu)", lambda: net(torch.tensor([[0.1] * 9] * 64, dtype=torch.float32).to(dev)))
