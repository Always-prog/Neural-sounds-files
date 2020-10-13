"""In this file is network by torch"""
print("import network library")
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
        elif activate == "Softmax":
            self.activate = self.activate = nn.Softmax()
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
        self.errors = []
        self.optimizer = optim.SGD(params=self.parameters(),lr=optimizer_lr)

    def forward(self, inputs):
        return self.layers(inputs)
    def reset_errors(self):
        self.errors = []

    def training_net(self,inputs, must_outputs):
        self.optimizer.zero_grad()
        self.output_from_net = self.forward(inputs)
        self.loss = self.criterion(self.output_from_net, must_outputs)
        self.loss.backward()
        self.errors.append(self.loss)
        self.optimizer.step()
        return self.output_from_net
    def save(self, path):
        tc.save(self.layers,path)
    def load(self, path: str):
        self.layers = tc.load(path)
