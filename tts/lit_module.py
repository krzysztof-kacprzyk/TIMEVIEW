import pytorch_lightning as pl
import torch
from .config import Config, OPTIMIZERS
from .model import TTS

class LitTTS(pl.LightningModule):

    def __init__(self,config: Config, model: TTS):
        super().__init__()
        self.config = config
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.lr = self.config.training.lr

    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis) # list of tensors
            return preds

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            return pred # 2D tensor

    def training_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch   
            preds = self.model(batch_X, batch_Phis)
            losses = [self.loss_fn(pred, y) for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2), dim=1) / batch_N) / batch_X.shape[0]

        self.log('train_loss', loss)
      
        return loss

    def validation_step(self, batch, batch_idx):
         
        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch  
            preds = self.model(batch_X, batch_Phis)
            losses = [self.loss_fn(pred, y) for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2), dim=1) / batch_N) / batch_X.shape[0]

        self.log('val_loss', loss)
      
        return loss
        
    def test_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)
            losses = [self.loss_fn(pred, y) for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2), dim=1) / batch_N) / batch_X.shape[0]

        self.log('test_loss', loss)
      
        return loss

    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.config.training.optimizer](self.model.parameters(), lr=self.config.training.lr)
        return optimizer
