import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

m = nn.BatchNorm2d(100)
# Without Learnable Parameters
m = nn.BatchNorm2d(100, affine=False)
#input = torch.randint.random(20,100,35,45) 
input = torch.randn(20,100,35,45)
#randn(20, 100, 35, 45)
output = m(input)
print(input)
print(output)