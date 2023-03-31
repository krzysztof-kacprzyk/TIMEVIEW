import torch
import numpy as np
import pytorch_lightning as pl

from tts.basis import BSplineBasis
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
        torch.manual_seed(config.seed)
    
    def forward(self, X, Phis):
        """
        Args:
            X: a tensor of shape (D,M) where D is the number of sample and M is the number of static features
            Phi:
                if dataloader_type = 'tensor': a tensor of shape (D,N_max,B) where D is the number of sample, N_max is the maximum number of time steps and B is the number of basis functions
                if dataloader_type = 'iterative': a list of D tensors of shape (N_d,B) where N_d is the number of time steps and B is the number of basis functions
        """
        if self.config.dataloader_type == "iterative":
            h = self.encoder(X)
            return [torch.matmul(Phi,h[d,:]) for d, Phi in enumerate(Phis)]
        elif self.config.dataloader_type == "tensor":
            h = self.encoder(X)
            return torch.matmul(Phis,torch.unsqueeze(h,-1)).squeeze(-1)
        

    def forecast_trajectory(self,x,t):
        """
        Args:
            x: a numpy array of shape (M,) where M is the number of static features
            t: a numpy array of shape (N,) where N is the number of time steps
        """
        x = torch.unsqueeze(torch.from_numpy(x),0).float()
        bspline = BSplineBasis(self.config.n_basis, (0,self.config.T))
        Phi = torch.from_numpy(bspline.get_matrix(t)).float()
        h = self.encoder(x)
        return torch.matmul(Phi,h[0,:])

    def forecast_trajectories(self,X,t):
        """
        Args:
            X: a numpy array of shape (D,M) where D is the number of sample and M is the number of static features
            t: a numpy array of shape (N,) where N is the number of time steps
        """
        X = torch.from_numpy(X).float()
        bspline = BSplineBasis(self.config.n_basis, (0,self.config.T))
        Phi = torch.from_numpy(bspline.get_matrix(t)).float() # shape (N,B)
        h = self.encoder(X) # shape (D,B)
        return torch.matmul(h,Phi.T) # shape (D,N)