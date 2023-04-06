import torch
import numpy as np
import torch.autograd
from torch.autograd import Variable

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

## Simple example https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2  # Our function 3a^3 - b^2

## We want to compute dQ/da and dQ/db
# We use backgward method but it requires a vector as input which represents dQ/dU assuming we use chain dQ/da = dQ/dU * dU/da
dqdu = torch.tensor([1.,1.])
Q.backward(gradient=dqdu)
##
print(f"(a={a.detach().numpy()}) dQ/da (9a^2) = {a.grad}")
print(f"(b={b.detach().numpy()}) dQ/db (-2b) = {b.grad}")
##

