from __future__ import annotations

import torch

from nmt_deep4.model import Deep4Net


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Deep4Net(n_chans=19, n_times=600).to(device)
    x = torch.randn(8, 19, 600, device=device)
    y = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(y, torch.randint(0, 2, (8,), device=device).float())
    loss.backward()
    print("Device:", device)
    print("Input:", tuple(x.shape))
    print("Output:", tuple(y.shape))
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    print("Forward/backward smoke test: PASS")


if __name__ == "__main__":
    main()
