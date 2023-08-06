import pytorch_lightning as pl
import torch
from .config import NeuralODEConfig, OPTIMIZERS
from .neural_ode import NeuralGradient, NeuralGradient2
from torchdiffeq import odeint as odeint
from torchdiffeq import odeint_adjoint as odeint_adjoint

class LitNeuralODE(pl.LightningModule):

    def __init__(self,config: NeuralODEConfig):
        super().__init__()
        self.config = config
        self.model = NeuralGradient2(config)
        self.loss = torch.nn.MSELoss()

    def predict_step(self, batch, batch_idx):
        batch_X, batch_ts, batch_ys = batch
        for i in range(len(batch_ts)):
            X = batch_X[i,:] # shape (n_covariates)
            t = batch_ts[i] # shape (n_timepoints)
            y = batch_ys[i] # shape (n_timepoints)
            y0 = y[[0]] # shape (1)
            # pred = odeint(lambda t, y: self.model.forward(t,y,X), y0, t)
            y_cov_0 = torch.cat([y0, X], dim=-1) # shape (n_covariates + 1)
            pred = odeint_adjoint(self.model, y_cov_0, t)
            y_pred = pred[:,0]
        return y_pred

    def training_step(self, batch, batch_idx):
        batch_X, batch_ts, batch_ys = batch
        losses = []
        for i in range(len(batch_ts)):
            X = batch_X[i,:] # shape (n_covariates)
            t = batch_ts[i] # shape (n_timepoints)
            y = batch_ys[i] # shape (n_timepoints)
            y0 = y[[0]] # shape (1)
            # pred = odeint(lambda t, y: self.model.forward(t,y,X), y0, t)
            y_cov_0 = torch.cat([y0, X], dim=-1) # shape (n_covariates + 1)
            pred = odeint_adjoint(self.model, y_cov_0, t, method='rk4', rtol=1e-3, atol=1e-3, options={'max_num_steps': 20}, adjoint_options={'max_num_steps': 20})
            y_pred = pred[:,0]
            loss = self.loss(y_pred, y)
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_X, batch_ts, batch_ys = batch
        losses = []
        for i in range(len(batch_ts)):
            X = batch_X[i,:] # shape (n_covariates)
            t = batch_ts[i] # shape (n_timepoints)
            y = batch_ys[i] # shape (n_timepoints)
            y0 = y[[0]] # shape (1)
            # pred = odeint(lambda t, y: self.model.forward(t,y,X), y0, t)
            y_cov_0 = torch.cat([y0, X], dim=-1) # shape (n_covariates + 1)
            pred = odeint_adjoint(self.model, y_cov_0, t)
            y_pred = pred[:,0]
            loss = self.loss(y_pred, y)
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        batch_X, batch_ts, batch_ys = batch
        losses = []
        for i in range(len(batch_ts)):
            X = batch_X[i,:] # shape (n_covariates)
            t = batch_ts[i] # shape (n_timepoints)
            y = batch_ys[i] # shape (n_timepoints)
            y0 = y[[0]] # shape (1)
            # pred = odeint(lambda t, y: self.model.forward(t,y,X), y0, t)
            y_cov_0 = torch.cat([y0, X], dim=-1) # shape (n_covariates + 1)
            pred = odeint_adjoint(self.model, y_cov_0, t)
            y_pred = pred[:,0]
            loss = self.loss(y_pred, y)
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.config.training.optimizer](self.model.parameters(
        ), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        return optimizer

    