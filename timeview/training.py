from pytorch_lightning.utilities.seed import seed_everything
from timeview.lit_module import LitTTS
from timeview.model import TTS
from timeview.data import create_train_val_test_dataloaders
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
import timeview
import optuna
import sys
sys.path.append('../')


def training(
    seed: int,
    config: timeview.config.Config,
    dataset: timeview.data.TTSDataset,
    trial: optuna.trial.Trial = None
):
    '''
    Train evaluation script
    Arguments:
        seed: int - seed for reproducibility
        config: tts.config.Config - config object
        dataset: tts.data.TTSDataset - dataset object
        trial: optuna.trial.Trial - optuna trial object, only pass in during hyperparameter tuning else None
    Returns:
        if performing hyperparameter tuning, returns the validation loss
        else returns the best model found during training
    '''

    seed_everything(seed=seed, workers=True)
    train_dataloader, val_dataloader, _ = create_train_val_test_dataloaders(
        config, dataset)
    tts = TTS(config)
    litmodel = LitTTS(config, tts)

    # updated logger to log to logs/{training, tuning}/run_seed so
    # different seeded runs are stored in different folders
    if trial:
        log_dir = f'logs/tuning/run_{seed}'
    else:
        log_dir = f'logs/training/run_{seed}'
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir, name='tts')

    # create callbacks
    best_val_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename='best_val'
    )
    # added early stopping callback, if validation loss does not improve over 10 epochs -> terminate training.
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=False,
        mode='min'
    )
    callback_ls = [best_val_checkpoint, early_stop_callback]

    # add additional callback for optuna hyperopt
    if trial:
        callback_ls.append(PyTorchLightningPruningCallback(
            trial, monitor='val_loss'))

    trainer_dict = {
        'deterministic': True,
        'devices': 1,
        'auto_lr_find': True,
        'enable_model_summary': False,
        'enable_progress_bar': False,
        'accelerator': 'cpu',
        'max_epochs': 200,      # i would recommend increasing this, it appears your model hits max_epochs every time, suggesting it could be trained more
        'logger': tb_logger,
        'check_val_every_n_epoch': 10,
        'log_every_n_steps': 1,
        'callbacks': callback_ls
    }

    trainer = pl.Trainer(**trainer_dict)

    trainer.fit(
        model=litmodel,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    best_val_loss = early_stop_callback.best_score

    # if doing hyperparameter tuning, return loss on validation set
    if trial:
        return best_val_loss
    # else return best model for further inferencing
    else:
        litmodel = LitTTS.load_from_checkpoint(
            best_val_checkpoint.best_model_path,
            config=config,
            model=tts
        )
        return litmodel
