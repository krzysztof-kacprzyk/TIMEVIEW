import pytorch_lightning as pl
import torch
from .config import RNNConfig, OPTIMIZERS
from .rnn import RNN, DeltaTRNN


class LitRNN(pl.LightningModule):

    def __init__(self,config: RNNConfig):
        super().__init__()
        self.config = config
        self.model = RNN(config)
        self.loss = torch.nn.MSELoss()

    def predict_step(self, batch, batch_idx):
        X, y = batch
        pred = self.model(X)
        return pred

    def training_step(self, batch, batch_idx):
        X, y = batch
        pred = self.model(X)
        loss = self.loss(pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred = self.model(X)
        loss = self.loss(pred, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        pred = self.model(X)
        loss = self.loss(pred, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.config.training.optimizer](self.model.parameters(
        ), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        return optimizer

class LitDeltaTRNN(pl.LightningModule):

    def __init__(self,config: RNNConfig):
        super().__init__()
        self.config = config
        self.model = DeltaTRNN(config)
        self.loss = torch.nn.MSELoss()

    def predict_step(self, batch, batch_idx):
        """
        batch is a tuple of (X, y, dt, mask)
        X.shape (batch_size, n_features)
        y.shape (batch_size, T_max_len)
        dt.shape (batch_size, T_max_len)
        mask.shape (batch_size, T_max_len)

        y and dt are padded with zeros
        mask is 1 if the value is not padded, 0 otherwise
        """
        X, dt, y, mask = batch
        pred = self.model(X, dt)
        pred = pred * mask
        return pred

    def training_step(self, batch, batch_idx):
        X, dt, y, mask = batch
        pred = self.model(X, dt)
        pred = pred * mask
        n = torch.sum(mask, dim=1)
        loss = torch.sum(torch.sum(((pred - y) ** 2),
                             dim=1) / n) / X.shape[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, dt, y, mask = batch
        pred = self.model(X, dt)
        pred = pred * mask
        n = torch.sum(mask, dim=1)
        loss = torch.sum(torch.sum(((pred - y) ** 2),
                             dim=1) / n) / X.shape[0]
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        X, dt, y, mask = batch
        pred = self.model(X, dt)
        pred = pred * mask
        n = torch.sum(mask, dim=1)
        loss = torch.sum(torch.sum(((pred - y) ** 2),
                             dim=1) / n) / X.shape[0]
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.config.training.optimizer](self.model.parameters(
        ), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        return optimizer
    