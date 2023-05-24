import pytorch_lightning as pl
import torch
from .config import NeuralODEConfig, OPTIMIZERS
from .neural_ode import NeuralGradient
from torchdiffeq import odeint as odeint



class LitNeuralODE(pl.LightningModule):

    def __init__(self,config: NeuralODEConfig):
        super().__init__()
        self.config = config
        self.model = NeuralGradient(config)
        self.loss = torch.nn.MSELoss()

    def predict_step(self, batch, batch_idx):
        batch_X, batch_ts, batch_ys = batch
        for i in range(len(batch_ts)):
            X = batch_X[i,:]
            t = batch_ts[i]
            y = batch_ys[i]
            y0 = torch.cat((y[[0]],X))
            pred = odeint(self.model, y0, t)
            y_pred = pred[:,0]
        return y_pred

    def training_step(self, batch, batch_idx):
        batch_X, batch_ts, batch_ys = batch
        losses = []
        for i in range(len(batch_ts)):
            X = batch_X[[i],:]
            t = batch_ts[i]
            y = batch_ys[i]
            y0 = y[[0]].unsqueeze(0)
            pred = odeint(lambda t, y: self.model.forward(t,y,X), y0, t)
            y_pred = pred[:,0,0]
            loss = self.loss(y_pred, y)
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_X, batch_ts, batch_ys = batch
        losses = []
        for i in range(len(batch_ts)):
            X = batch_X[[i],:]
            t = batch_ts[i]
            y = batch_ys[i]
            y0 = y[[0]].unsqueeze(0)
            pred = odeint(lambda t, y: self.model.forward(t,y,X), y0, t)
            y_pred = pred[:,0,0]
            loss = self.loss(y_pred, y)
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        batch_X, batch_ts, batch_ys = batch
        losses = []
        for i in range(len(batch_ts)):
            X = batch_X[[i],:]
            t = batch_ts[i]
            y = batch_ys[i]
            y0 = y[[0]].unsqueeze(0)
            pred = odeint(lambda t, y: self.model.forward(t,y,X), y0, t)
            y_pred = pred[:,0,0]
            loss = self.loss(y_pred, y)
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.config.training.optimizer](self.model.parameters(
        ), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        return optimizer

    