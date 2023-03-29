import torch

x1 = torch.tensor(10.0, requires_grad=True)
x2 = torch.tensor(2.0, requires_grad=True)
x3 = torch.tensor(30.0, requires_grad=True)
x4 = x1 + x2 * x3
x4.backward()

print(x1.data, x2.data, x3.data, x4.data)
print(x1.grad.item(), x2.grad.item(), x3.grad.item())
