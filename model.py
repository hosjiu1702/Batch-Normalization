"""
Defines k (input) hidden layers feedforward network
"""
from typing import List, Text
import torch
import torch.nn as nn


""" A simple Feedforward Neural Network """
class FFN(nn.Module):

    def __init__(
            self,
            num_hidden: List[int],
            input_size: int,
            output_size: int,
        ):
        super(FFN, self).__init__()
        self._k = len(num_hidden)
        self.input_size = input_size
        self.output_size = output_size
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(input_size, num_hidden[0]))
        for i in range(self._k - 1):
            self.fc_layers.append(nn.Linear(num_hidden[i], num_hidden[i+1]))
        self.fc_layers.append(nn.Linear(num_hidden[-1], output_size))

    def forward(self, x):
        for i in range(self._k - 1):
            x = self.fc_layers[i](x)
            x = torch.sigmoid(x)
        x = self.fc_layers[-1](x)

        return x
