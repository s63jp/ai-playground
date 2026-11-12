"""Spike: does a small MLP train on the Arc A770 (XPU), and how fast vs CPU?"""
import time

import torch

for dev in (["xpu"] if torch.xpu.is_available() else []) + ["cpu"]:
    torch.manual_seed(0)
    x = torch.rand(20000, 6, device=dev)
    y = (x[:, :3].sum(1, keepdim=True) ** 2) + torch.sin(x[:, 3:4] * 6)
    net = torch.nn.Sequential(torch.nn.Linear(6, 128), torch.nn.SiLU(), torch.nn.Linear(128, 128), torch.nn.SiLU(),
                              torch.nn.Linear(128, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    t = time.perf_counter()
    for epoch in range(30):
        for i in range(0, 20000, 512):
            loss = torch.nn.functional.mse_loss(net(x[i:i + 512]), y[i:i + 512])
            opt.zero_grad()
            loss.backward()
            opt.step()
    if dev == "xpu":
        torch.xpu.synchronize()
    print(f"{dev}: 30 epochs x 20k samples in {time.perf_counter() - t:.2f}s, final loss {loss.item():.5f}")
