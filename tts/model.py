import torch
import numpy as np
import pytorch_lightning as pl
from .config import Config

class Encoder(torch.nn.Module):

    def __init__(self,config):
        """
        Args:
            config: an instance of the Config class
        """
        super().__init__()
        self.config = config
        self.n_features = config.n_features
        self.n_basis = config.n_basis
        self.hidden_sizes =  config.encoder.hidden_sizes

        assert len(self.hidden_sizes) > 0

        self.layers = []
        activation = torch.nn.ReLU()

        self.layers.append(torch.nn.Linear(self.n_features,self.hidden_sizes[0]))
        self.layers.append(activation)

        for i in range(len(self.hidden_sizes) - 1):
            self.layers.append(torch.nn.Linear(self.hidden_sizes[i],self.hidden_sizes[i+1]))
            self.layers.append(activation)
        
        self.layers.append(torch.nn.Linear(self.hidden_sizes[-1],self.n_basis))
        self.nn = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.nn(x)

class TTS(torch.nn.Module):

    def __init__(self,config):
        """
        Args:
            config: an instance of the Config class
        """
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
    
    def forward(self, X, Phis):
        """
        Args:
            x: a tensor of shape (M,) where M is the number of static features
            Phi: a tensor of shape (N,B) where N is the number of time steps and B is the number of basis functions
        """
        if self.config.dataloader_type == "iterative":
            h = self.encoder(X)
            return [torch.matmul(Phi,h[d,:]) for d, Phi in enumerate(Phis)]
        elif self.config.dataloader_type == "tensor":
            h = self.encoder(X)
            return torch.matmul(Phis,torch.unsqueeze(h,-1)).squeeze(-1)