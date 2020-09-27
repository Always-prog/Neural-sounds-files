import torch as tc
import torch.nn as nn
import torch.optim as optim



class Network(nn.Module):
    def __init__(self, net_plane: list,activate: str = "ReLU", reduction: str = "mean", optimizer_momentum: float = 0.9,optimizer_lr: float = 0.01):
        super(Network,self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        if activate == "ReLU":
            self.activate = nn.ReLU()
        elif activate == "Sigmoid":
            self.activate = nn.Sigmoid()
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
        self.optimizer = optim.SGD(params=self.parameters(),lr=optimizer_lr, momentum=optimizer_momentum)

    def forward(self, inputs):
        return self.layers(inputs)

    def training_net(self,inputs, must_outputs):
        self.optimizer.zero_grad()
        self.output_from_net = self.forward(inputs)
        self.loss = self.criterion(self.output_from_net, must_outputs)
        self.loss.backward()
        self.optimizer.step()
        return self.loss





Net = Network(net_plane=[1,10,5,1])
print(Net)
print(Net.forward(inputs=tc.rand(1,1)))