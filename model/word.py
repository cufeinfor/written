import torch

x=torch.rand([128,20,30])
print(x)
print(x.size())
y=x.view([-1,20,128])
print(y.size())