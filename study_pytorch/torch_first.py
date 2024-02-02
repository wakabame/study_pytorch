"""
損失関数に対して自動微分による勾配の計算と、手計算による勾配の計算の比較

z = w * x + b に対して、損失関数を
$$
L(w, b) = (y - x)^2
$$
により定義するには、w, b を requires_grad=True として勾配の計算対象にする
"""

import torch

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
x = torch.tensor([1.4])
y = torch.tensor([2.1])
z = w * x + b
loss = (y - z).pow(2).sum()
loss.backward()  # type: ignore[no-untyped-call]

print(f"dL/dw by torch: {w.grad:.6f}")
print(f"dL/db by torch: {b.grad:.6f}")

print(f"dL/dw by hand: {(2 * x * (w*x+b-y)).item():.6f}")
print(f"dL/dw by hand: {(2 * (w*x+b-y)).item():.6f}")
