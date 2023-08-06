import pytorch_lightning as pl
import torch
from .config import NeuralLaplaceConfig, OPTIMIZERS
from .neural_laplace import LaplaceEncoder, LaplaceRepresentationFunc
from torchlaplace import laplace_reconstruct

class LitNeuralLaplace(pl.LightningModule):

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.encoder = LaplaceEncoder(config)
        self.representation_func = LaplaceRepresentationFunc(s_dim=33, output_dim=1, latent_dim=self.config.encoder.latent_dim, hidden_units=64)
        self.loss = torch.nn.MSELoss()

    def predict_step(self, batch, batch_idx):
        batch_X, batch_ts, batch_ys = batch
        for i in range(len(batch_ts)):
            X = batch_X[i,:].unsqueeze(0) # shape (1,n_covariates)
            t = batch_ts[i].unsqueeze(0) # shape (1,n_timepoints)
            y = batch_ys[i] # shape (n_timepoints)
            h = self.encoder.forward(X) # shape (1,latent_dim)
            pred = laplace_reconstruct(self.representation_func, h, t, recon_dim=1, ilt_algorithm='fourier', use_sphere_projection=True, ilt_reconstruction_terms=33, options=None, compute_deriv=False, x0=None)
            y_pred = torch.nan_to_num(pred[0,:,0])
          
        return y_pred

    def training_step(self, batch, batch_idx):
        batch_X, batch_ts, batch_ys = batch
        losses = []
        for i in range(len(batch_ts)):
            X = batch_X[i,:].unsqueeze(0) # shape (1,n_covariates)
            t = batch_ts[i].unsqueeze(0) # shape (1,n_timepoints)
            y = batch_ys[i] # shape (n_timepoints)
            h = self.encoder.forward(X) # shape (1,latent_dim)
            pred = laplace_reconstruct(self.representation_func, h, t, recon_dim=1, ilt_algorithm='fourier', use_sphere_projection=True, ilt_reconstruction_terms=33, options=None, compute_deriv=False, x0=None)
            y_pred = torch.nan_to_num(pred[0,:,0])
            loss = self.loss(y_pred, y)
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        self.log('train_loss', loss)
        print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_X, batch_ts, batch_ys = batch
        losses = []
        for i in range(len(batch_ts)):
            X = batch_X[i,:].unsqueeze(0) # shape (1,n_covariates)
            t = batch_ts[i].unsqueeze(0) # shape (1,n_timepoints)
            y = batch_ys[i] # shape (n_timepoints)
            h = self.encoder.forward(X) # shape (1,latent_dim)
            pred = laplace_reconstruct(self.representation_func, h, t, recon_dim=1, ilt_algorithm='fourier', use_sphere_projection=True, ilt_reconstruction_terms=33, options=None, compute_deriv=False, x0=None)
            y_pred = torch.nan_to_num(pred[0,:,0])
            loss = self.loss(y_pred, y)
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        batch_X, batch_ts, batch_ys = batch
        losses = []
        for i in range(len(batch_ts)):
            X = batch_X[i,:].unsqueeze(0) # shape (1,n_covariates)
            t = batch_ts[i].unsqueeze(0) # shape (1,n_timepoints)
            y = batch_ys[i] # shape (n_timepoints)
            h = self.encoder.forward(X) # shape (1,latent_dim)
            pred = laplace_reconstruct(self.representation_func, h, t, recon_dim=1, ilt_algorithm='fourier', use_sphere_projection=True, ilt_reconstruction_terms=33, options=None, compute_deriv=False, x0=None)
            y_pred = torch.nan_to_num(pred[0,:,0])
            loss = self.loss(y_pred, y)
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        params = list(self.representation_func.parameters()) + list(self.encoder.parameters())
        optimizer = OPTIMIZERS[self.config.training.optimizer](params, lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        return optimizer

    