import torch
import numpy as np
from describe import describe

# Torch randoms
describe(torch.rand(2,3))  # uniform random
describe(torch.randn(2,3)) # random normal

# Torch creating tensors
describe(torch.zeros(2,3))
x = torch.ones(2,3)
describe(x)
x.fill_(5)                 # All _ methods refer to in-place operations
describe(x)

x = torch.Tensor([[1,2,3],
                [4,5,6]])
describe(x)

npy = np.random.rand(2,3)
describe(torch.from_numpy(npy))

# Torch tensor operations
describe(torch.add(x,x))
describe(x + x)

x = torch.arange(6)
describe(x)

x = x.view(2,3)
describe(x)
describe(torch.sum(x, dim=0)) # dimension 0
describe(torch.sum(x, dim=1)) # dimension 1
describe(torch.transpose(x, 0, 1))

# Torch indexing slicing joining
x = torch.arange(6).view(2,3)
describe(x[:1, :2])
describe(x[0, 1])

indices = torch.LongTensor([0,2])
describe(torch.index_select(x, dim=1, index=indices))

indices = torch.LongTensor([0,0])
describe(torch.index_select(x, dim=0, index=indices))

row_indices = torch.arange(2).long()
col_indices = torch.LongTensor([0,1])
describe(x[row_indices, col_indices])

# Concatenating tensors
x = torch.arange(6).view(2,3)
describe(x)
describe(torch.cat([x, x], dim=0))
describe(torch.cat([x, x], dim=1))
describe(torch.stack([x, x]))

# Linear Algebra
x1 = torch.arange(6).view(2,3).float() # default is long, must have 2 of the same type (Long, float, double, etc.)
describe(x1)
x2 = torch.ones(3,2)
x2[:, 1] += 1
describe(x2)
describe(torch.mm(x1, x2))

# Gradients
x = torch.ones(2,2, requires_grad=True)
describe(x)
print(x.grad is None)
y = (x + 2) * (x + 5) + 3
describe(y)
print(x.grad is None)
z = y.mean()
describe(z)
z.backward()
print(x.grad is None)
print(x.grad)

# CUDA
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x = torch.rand(3,3).to(device)
describe(x)
# Don't mix devices!
y = torch.rand(3, 3)
# x + y  # Will error out because x and y are on gpu and cpu respectively

cpu_device = torch.device("cpu")
y = y.to(cpu_device)
x = x.to(cpu_device)