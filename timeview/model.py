import torch
import numpy as np
import pytorch_lightning as pl

from timeview.basis import BSplineBasis
from .config import Config

def is_dynamic_bias_enabled(config):
    if hasattr(config, 'dynamic_bias'):
        return config.dynamic_bias
    else:
        return False

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
        self.dropout_p = config.encoder.dropout_p

        assert len(self.hidden_sizes) > 0

        self.layers = []
        activation = torch.nn.ReLU()

        self.layers.append(torch.nn.Linear(self.n_features,self.hidden_sizes[0]))
        self.layers.append(torch.nn.BatchNorm1d(self.hidden_sizes[0]))
        self.layers.append(activation)
        self.layers.append(torch.nn.Dropout(self.dropout_p))

        for i in range(len(self.hidden_sizes) - 1):
            self.layers.append(torch.nn.Linear(self.hidden_sizes[i],self.hidden_sizes[i+1]))
            self.layers.append(torch.nn.BatchNorm1d(self.hidden_sizes[i+1]))
            self.layers.append(activation)
            self.layers.append(torch.nn.Dropout(self.dropout_p))
        
        latent_size = self.n_basis

        if is_dynamic_bias_enabled(config):
            latent_size += 1
        
        self.layers.append(torch.nn.Linear(self.hidden_sizes[-1],latent_size))

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
        torch.manual_seed(config.seed)
        self.config = config
        self.encoder = Encoder(self.config)
        if not is_dynamic_bias_enabled(self.config):
            self.bias = torch.nn.Parameter(torch.zeros(1))
        
    
    def forward(self, X, Phis):
        """
        Args:
            X: a tensor of shape (D,M) where D is the number of sample and M is the number of static features
            Phi:
                if dataloader_type = 'tensor': a tensor of shape (D,N_max,B) where D is the number of sample, N_max is the maximum number of time steps and B is the number of basis functions
                if dataloader_type = 'iterative': a list of D tensors of shape (N_d,B) where N_d is the number of time steps and B is the number of basis functions
        """
        h = self.encoder(X)
        if is_dynamic_bias_enabled(self.config):
            self.bias = h[:,-1]
            h = h[:,:-1]
        
        if self.config.dataloader_type == "iterative":
            if is_dynamic_bias_enabled(self.config):
                return [torch.matmul(Phi,h[d,:]) + self.bias[d] for d, Phi in enumerate(Phis)]
            else:
                return [torch.matmul(Phi,h[d,:]) + self.bias for d, Phi in enumerate(Phis)]
        elif self.config.dataloader_type == "tensor":
            if is_dynamic_bias_enabled(self.config):
                return torch.matmul(Phis,torch.unsqueeze(h,-1)).squeeze(-1) + torch.unsqueeze(self.bias,-1)
            else:
                return torch.matmul(Phis,torch.unsqueeze(h,-1)).squeeze(-1) + self.bias
        
    def predict_latent_variables(self,X):
        """
        Args:
            X: a numpy array of shape (D,M) where D is the number of sample and M is the number of static features
        Returns:
            a numpy array of shape (D,B) where D is the number of sample and B is the number of basis functions
        """
        device = self.encoder.layers[0].bias.device
        X = torch.from_numpy(X).float().to(device)
        self.encoder.eval()
        if is_dynamic_bias_enabled(self.config):
            with torch.no_grad():
                return self.encoder(X)[:,:-1].cpu().numpy()
        else:
            with torch.no_grad():
                return self.encoder(X).cpu().numpy()        

    def forecast_trajectory(self,x,t):
        """
        Args:
            x: a numpy array of shape (M,) where M is the number of static features
            t: a numpy array of shape (N,) where N is the number of time steps
        Returns:
            a numpy array of shape (N,) where N is the number of time steps
        """
        device = self.encoder.layers[0].bias.device
        x = torch.unsqueeze(torch.from_numpy(x),0).float().to(device)
        bspline = BSplineBasis(self.config.n_basis, (0,self.config.T), internal_knots=self.config.internal_knots)
        Phi = torch.from_numpy(bspline.get_matrix(t)).float().to(device)
        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(x)
            if is_dynamic_bias_enabled(self.config):
                self.bias = h[0,-1]
                h = h[:,:-1]
            return (torch.matmul(Phi,h[0,:]) + self.bias).cpu().numpy()

    def forecast_trajectories(self,X,t):
        """
        Args:
            X: a numpy array of shape (D,M) where D is the number of sample and M is the number of static features
            t: a numpy array of shape (N,) where N is the number of time steps
        Returns:
            a numpy array of shape (D,N) where D is the number of sample and N is the number of time steps
        """
        device = self.encoder.layers[0].bias.device
        X = torch.from_numpy(X).float().to(device)
        bspline = BSplineBasis(self.config.n_basis, (0,self.config.T), internal_knots=self.config.internal_knots)
        Phi = torch.from_numpy(bspline.get_matrix(t)).float().to(device) # shape (N,B)
        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(X) # shape (D,B)
            if is_dynamic_bias_enabled(self.config):
                self.bias = h[:,-1]
                h = h[:,:-1]
            return (torch.matmul(h,Phi.T)+self.bias).cpu().numpy() # shape (D,N), broadcasting will take care of the bias