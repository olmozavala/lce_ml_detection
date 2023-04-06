# This notebook contains a **hello world** example of neural networks with PyTorch. Basically a linear regression approximation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time as t

device = "cuda" if torch.cuda.is_available() else "cpu"

## Linear regression Example

# 1. Generate some synthetic data
x = torch.linspace(0,10,100)
m = 2
y = x*m + torch.rand(x.shape)*2

plt.scatter(x, y, s=10, label='Data')
plt.title("Training noisy data")
plt.legend()
plt.show()

## 2. Create a model of a single neuron. By default the parameters of that model are initialized randomly.
# Models are created by classes that inherit from Module
class SingleNeuronModel(nn.Module):
    # On the init function we define our model
    def __init__(self):
        super().__init__() # Constructor of parent class
        self.single_neuron =  nn.Linear(1, 1)
    
    # On the foreward model we indicate how to make one 'pass' of the model
    def forward(self, x):
        return self.single_neuron(x)

ex_model = SingleNeuronModel().to(device)  # Send to GPU
X = torch.reshape(x, (x.shape[0],1)).to(device)  # Reshape with proper shape (with 1 band)
Y = torch.reshape(y, (y.shape[0],1)).to(device)
print("Total number of parameters: ", sum(p.numel() for p in ex_model.parameters() if p.requires_grad))
print(list(ex_model.named_parameters()))

##-------------- Just for plotting --------------
fig, ax = plt.subplots(1,1)
def plotCurrentModel(x, y, model, ax):
    # Torch receives inputs with shape [Examples, input_size]
    model_y = model(X).cpu().detach().numpy()

    ax.scatter(x, y, s=10, label='True')
    p = ax.scatter(x, model_y, s=10, label='Random model')
    ax.set_title('Default model')
    ax.legend()
    return p

plotCurrentModel(x, y, ex_model, ax)
plt.show()

## 3.Optimize the parameters of the model using backpropagation
loss_mse = nn.MSELoss() # Define loss function
optimizer = torch.optim.SGD(ex_model.parameters(), lr=2e-3) # Define optimization algorithm

# Optimize the parameters several times
ex_model.train()
for i in range(200):
    # A (basic) training step in PyTorch consists of four basic parts:
    # 1.   Set all the gradients to zero using `opt.zero_grad()`
    # 2.   Calculate the loss, `loss`
    # 3.   Calculate the gradients with respect to the loss using `loss.backward()`
    # 4.   Update the parameters being optimized using `opt.step()`
    optimizer.zero_grad()
    pred = ex_model(X)
    loss = loss_mse(pred, Y)
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    # ---------- Just for plotting ---------
    if i % 10 == 0:
        fig, ax = plt.subplots(1,1)
        title = f"Iteration number {i} loss: {loss:0.3f}"
        print(title)
        plotCurrentModel(x, y, ex_model, ax)
        ax.set_title(title)
        plt.show()

print("Done!")
##

