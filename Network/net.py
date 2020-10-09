import torch as tc
import torch.nn as nn
import torch.optim as optim



class Network(nn.Module):
    def __init__(self, net_plane: list,activate: str = "Sigmoid", reduction: str = "mean", optimizer_lr: float = 0.01):
        super(Network,self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        if activate == "ReLU":
            self.activate = nn.ReLU()
        elif activate == "Sigmoid":
            self.activate = nn.Sigmoid()
        elif activate == "Tanh":
            self.activate = nn.Tanh()
        elif activate == "LeakyReLU":
            self.activate = nn.LeakyReLU()
        elif activate == "RReLU":
            self.activate = self.activate = nn.RReLU()
        elif activate == "Hardsigmoid":
            self.activate = self.activate = nn.Hardsigmoid()
        elif activate == "Hardtanh":
            self.activate = self.activate = nn.Hardtanh()
        elif activate == "Hardswish":
            self.activate = self.activate = nn.Hardswish()
        elif activate == "Softplus":
            self.activate = self.activate = nn.Softplus()
        elif activate == "Softshrink":
            self.activate = self.activate = nn.Softplus()
        elif activate == "Tanhshrink":
            self.activate = self.activate = nn.Tanhshrink()
        elif activate == "GELU":
            self.activate = self.activate = nn.GELU()
        elif activate == "CELU":
            self.activate = self.activate = nn.CELU()
        elif activate == "SELU":
            self.activate = self.activate = nn.SELU()
        elif activate == "ReLU6":
            self.activate = self.activate = nn.ReLU6()
        elif activate == "PReLU":
            self.activate = self.activate = nn.PReLU()
        elif activate == "LogSigmoid":
            self.activate = self.activate = nn.LogSigmoid()
        self.layers = nn.Sequential()
        plane_index = 0
        layer_index = 0
        for layer_ in range(len(net_plane)-1):
            self.layers.add_module(name=str(layer_index),module=nn.Linear(in_features=net_plane[plane_index],out_features=net_plane[plane_index+1]))
            plane_index += 1
            layer_index += 1
            self.layers.add_module(name=str(layer_index), module=self.activate)
            layer_index += 1


        """Values from training"""
        self.loss = None
        self.output_from_net = None
        self.optimizer = optim.SGD(params=self.parameters(),lr=optimizer_lr)

    def forward(self, inputs):
        return self.layers(inputs)

    def training_net(self,inputs, must_outputs):
        self.optimizer.zero_grad()
        self.output_from_net = self.forward(inputs)
        self.loss = self.criterion(self.output_from_net, must_outputs)
        self.loss.backward()
        self.optimizer.step()
        return self.output_from_net
    def save(self, path):
        tc.save(self.layers,path)
    def load(self, path: str):
        self.layers = tc.load(path)


"""from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch import tensor
from torch import optim

import matplotlib.pyplot as plt

torch.manual_seed(0)
device = 'cpu'

# XOR gate inputs and outputs.
X = xor_input = tensor([[0,0], [0,1], [1,0], [1,1]]).float().to(device)
Y = xor_output = tensor([[0],[1],[1],[0]]).float().to(device)


# Use tensor.shape to get the shape of the matrix/tensor.
num_data, input_dim = X.shape
print('Inputs Dim:', input_dim) # i.e. n=2 

num_data, output_dim = Y.shape
print('Output Dim:', output_dim) 
print('No. of Data:', num_data) # i.e. n=4

# Step 1: Initialization. 

# Initialize the model.
# Set the hidden dimension size.
hidden_dim = 5
# Use Sequential to define a simple feed-forward network.
model = nn.Sequential(
            # Use nn.Linear to get our simple perceptron.
            nn.Linear(input_dim, hidden_dim),
            # Use nn.Sigmoid to get our sigmoid non-linearity.
            nn.Sigmoid(),
            # Second layer neurons.
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
model

# Initialize the optimizer
learning_rate = 0.3
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Initialize the loss function.
criterion = nn.MSELoss()

# Initialize the stopping criteria
# For simplicity, just stop training after certain no. of epochs.
num_epochs = 5000 

losses = [] # Keeps track of the loses.

# Step 2-4 of training routine.

for _e in tqdm(range(num_epochs)):
    # Reset the gradient after every epoch. 
    optimizer.zero_grad() 
    # Step 2: Foward Propagation
    predictions = model(X)

    # Step 3: Back Propagation 
    # Calculate the cost between the predictions and the truth.
    loss = criterion(predictions, Y)
    # Remember to back propagate the loss you've computed above.
    loss.backward()

    # Step 4: Optimizer take a step and update the weights.
    optimizer.step()

    # Log the loss value as we proceed through the epochs.
    losses.append(loss.data.item())


plt.plot(losses)
plt.show()"""