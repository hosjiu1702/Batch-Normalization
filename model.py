"""
Defines k (input) hidden layers feedforward network
"""
from typing import List, Text
import torch
import torch.nn as nn

from batchnorm import BatchNorm


""" A simple Feedforward Neural Network """
class FFN(nn.Module):

    def __init__(
            self,
            num_hidden: List[int],
            input_size: int,
            output_size: int,
            use_bn: bool = False
        ):
        super(FFN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bn = use_bn

        self.fc_layers = nn.ModuleList() # List of fully-connected layers
        self.bn_layers = nn.ModuleList() # List of batchnorm layers

        self.fc_layers.append(nn.Linear(input_size, num_hidden[0]))

        self._k = len(num_hidden) # Number of hidden layers
        for i in range(self._k - 1):
            self.fc_layers.append(nn.Linear(num_hidden[i], num_hidden[i+1]))
            if self.use_bn:
                self.bn_layers.append(BatchNorm(num_hidden[i+1]))

        self.fc_layers.append(nn.Linear(num_hidden[-1], output_size))

    def forward(self, x):
        for i in range(self._k - 1):
            x = self.fc_layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = torch.sigmoid(x)
        x = self.fc_layers[-1](x)

        return x

