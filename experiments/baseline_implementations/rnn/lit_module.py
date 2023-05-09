import pytorch_lightning as pl
import torch
from .config import RNNConfig, OPTIMIZERS
from .rnn import RNN


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

    