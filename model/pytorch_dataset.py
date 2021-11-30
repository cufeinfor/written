from numpy.core.fromnumeric import size
from torch.utils import data
from torchvision.datasets import MNIST, mnist
from torchvision import transforms
import numpy as np

mnist=MNIST(root="C:/Users/qiaoy/Desktop/data",train=True,download=False)

print(mnist[0])
ret=transforms.ToTensor()(mnist[0][0])
print(ret)
print(ret.size())

