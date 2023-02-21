import torch
import torch.nn as nn


class MyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyNetwork, self).__init__()
        ###
        '''The constructor for the neural network: the "architecture" of the network is FC (fully connected) with 3 
        layers: input: hidden layers: output layer: the activation in the hidden layers is the Rectified linear Unit, 
        the output is in the shape (2,) per example, ie 2 floats, ' one is the predicted stock price based on the 
        previous 5 days closing price and the 2nd value is the risk of prediction. '''


        ###

        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
    
    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    