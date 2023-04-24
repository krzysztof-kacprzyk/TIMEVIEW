import pytorch_lightning as pl
import torch
from .config import Config, OPTIMIZERS
from .model import TTS
import glob
import os
import pickle


def _get_seed_number(path):
    seeds = [os.path.basename(path).split("_")[1] for path in glob.glob(os.path.join(path, '*'))]
    seed = seeds[0]
    return seed

def _get_logs_seed_path(benchmarks_folder, timestamp, final=True, seed=None):

    # Create path
    if final:
        path = os.path.join(benchmarks_folder, timestamp, 'TTS', 'final', 'logs')
    else:
        path = os.path.join(benchmarks_folder, timestamp, 'TTS', 'tuning', 'logs')
    
    if seed is None:
        seed = _get_seed_number(path)

    logs_path = os.path.join(path, f'seed_{seed}')
    return logs_path

def _get_checkpoint_path_from_logs_seed_path(path):
    checkpoint_path = os.path.join(path, 'lightning_logs', 'version_0', 'checkpoints', 'best_val.ckpt')
    return checkpoint_path

def _load_config_from_logs_seed_path(path):
    config_path = os.path.join(path, 'config.pkl')
    # load config from a pickle
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    return config


def load_config(benchmarks_folder, timestamp, final=True, seed=None):

    logs_seed_path = _get_logs_seed_path(benchmarks_folder, timestamp, final=final, seed=seed)
    return _load_config_from_logs_seed_path(logs_seed_path)


def load_model(timestamp, benchmarks_folder='benchmarks', final=True, seed=None):

    logs_seed_path = _get_logs_seed_path(benchmarks_folder, timestamp, final=final, seed=seed)
    config = _load_config_from_logs_seed_path(logs_seed_path)
    checkpoint_path = _get_checkpoint_path_from_logs_seed_path(logs_seed_path)
    model = LitTTS.load_from_checkpoint(checkpoint_path, config=config)
    return model


class LitTTS(pl.LightningModule):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = TTS(config)
        self.loss_fn = torch.nn.MSELoss()
        self.lr = self.config.training.lr

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # def forward(self, batch, batch_idx, dataloader_idx=0):
        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)  # list of tensors
            return preds

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            return pred  # 2D tensor

    def training_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]

        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]

        self.log('test_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.config.training.optimizer](self.model.parameters(
        ), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        return optimizer
