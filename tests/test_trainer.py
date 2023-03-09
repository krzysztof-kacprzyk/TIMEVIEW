import pytest
from tts.trainer import LitTTS
from tts.config import Config
from tts.model import TTS
from tts.data import create_dataloaders, TTSDataset
import numpy as np
import torch
import pytorch_lightning as pl

@pytest.fixture
def data_fixture():
    rng = np.random.default_rng(0)
    Ns = rng.integers(1, 20, 10)
    X = rng.random((10,3))
    ts = [np.linspace(0, 1, N) for N in Ns]

    def trajectory(coeffs, t):
        return coeffs[0] + coeffs[1] * t + coeffs[2] * t ** 2
    
    ys = [trajectory(X[i,:], ts[i]) for i in range(10)]

    return X, ts, ys


def test_trainer_iterative(data_fixture):
    data = data_fixture
    config = Config(n_features=3,n_basis=7,dataloader_type='iterative')
    dataset = TTSDataset(config, data, 1)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(config, dataset)
    tts = TTS(config)
    litmodel = LitTTS(config, tts)

    trainer = pl.Trainer(deterministic=True,devices=1,check_val_every_n_epoch=1,auto_lr_find=True,enable_model_summary = False,enable_progress_bar=False,auto_scale_batch_size=False,accelerator='cpu',max_epochs=10)
            
    trainer.tune(litmodel,train_dataloaders=train_dataloader)
            
    trainer.fit(model=litmodel,train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

def test_trainer_tensor(data_fixture):
    data = data_fixture
    config = Config(n_features=3,n_basis=7,dataloader_type='tensor')
    dataset = TTSDataset(config, data, 1)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(config, dataset)
    tts = TTS(config)
    litmodel = LitTTS(config, tts)

    trainer = pl.Trainer(deterministic=True,devices=1,check_val_every_n_epoch=1,auto_lr_find=True,enable_model_summary = False,enable_progress_bar=False,auto_scale_batch_size=False,accelerator='cpu',max_epochs=10)
            
    trainer.tune(litmodel,train_dataloaders=train_dataloader)
            
    trainer.fit(model=litmodel,train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)