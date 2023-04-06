#%% 
import os
import matplotlib.pyplot as plt

import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
## Simple tensor
x = torch.tensor([1,3,6,4,2,3,3])
A = torch.Tensor(
    [
     [[1, 2], [3, 4]], 
     [[5, 6], [7, 8]], 
     [[9, 0], [1, 2]]
    ]
)

plt.plot()
os.listdir()

## Properties
print("----- Basics-----------")
print(F"Shape: {A.shape}")
print(F"Size = {A.size()}")
print(F"Length (Rank) = {len(A.shape)}")
print(F"Number of elements = {A.numel()}")
print(F"Sum of tensor: {A.sum()}")

# -- Examples with matrix --
print(F"A: {A}")
print(F"Flatten of A: {A.flatten()}")

# -- Examples with vector --
print(F"Vector: {x}")
print(F"Vector Tranpose: {(torch.t(x))}")
print(F"To numpy: {x.numpy()}")

## GPU
print("----- Testingt GPU-----------")
print(F"Cuda is available: {torch.cuda.is_available()}")
print(F"Number of GPUs: {torch.cuda.device_count()} {[torch.cuda.get_device_name(x) for x in range(torch.cuda.device_count())]}")
print(F"Current Device: {A.device}")
# print(F"Set device (when more than a single GPU): {torch.cuda.set_device(1)}")
gpu_tensor = A.to(device)
print(F"Current Device: {gpu_tensor.device}")

## Indexing
print("----- Indexing -----------")
print(A)
print(F"A[1]: {A[1]}")
print(F"A[1, 1, 0]: {A[1, 1, 0]}")
print(F"A[1, 1, 0].item(): {A[1, 1, 0].item()}")  # Scalar value
print(F"A[:, 0, 0]: {A[:, 0, 0]}")
print(f"Unique vec: {torch.unique(x)}")
print(f"Reshape as column: {A.reshape(1,-1).shape}")  # Reshape as col vector
print(f"Reshape as row vector: {A.reshape(-1,1).shape}")  # Reshape as row
print(f"View as row vector: {A.view(-1,1).shape}")  # Reshape as row

## Initialization
print("------- Initialization -----------")
print(F"Init ones: {torch.ones_like(x)}")
print(F"Init zeros: {torch.zeros_like(x)}")
print(F"Init vector filled from other tensor: {x.fill_(323)}")
print(F"Init random: {torch.randn_like(x, dtype=torch.float32)}")  # Rand doesn't work for Long so we need to define type
print(F"Init random gpu: {torch.randn(2, 2, device='cuda')}")  # Alternatively 'cuda' or 'cpu'

## ------- Basic Functions -------
(A - 5) * 2
print("Sum:", A.mean())
print("Mean:", A.mean())
print("Stdev:", A.std())
A.mean(0)
# Equivalently, you could also write:
# A.mean(dim=0)
# A.mean(axis=0)
# torch.mean(A, 0)
# torch.mean(A, dim=0)
# torch.mean(A, axis=0)

##

# %%
