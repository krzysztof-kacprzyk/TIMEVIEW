from datetime import datetime
import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod, abstractproperty
import json
import pickle
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from interpret.glassbox import ExplainableBoostingRegressor

from train_eval.training import training
from train_eval.tuning import tuning

from tts.data import TTSDataset, create_dataloader
from tts.config import TuningConfig, Config
from tts.model import TTS
from tts.lit_module import LitTTS
import pytorch_lightning as pl
import torch

class XTYDataset():
    def __init__(self, X, ts, ys):
        self.X = X
        self.ts = ts
        self.ys = ys

    def __len__(self):
        return len(self.X)
    
    def get_single_matrix(self, indices):
        samples = []
        for i in indices:
            n_rows = len(self.ts[i])
            X_i_tiled = np.tile(self.X[i], (n_rows, 1))
            samples.append(np.concatenate((X_i_tiled, np.expand_dims(self.ts[i],1), np.expand_dims(self.ys[i],1)), axis=1))
        return np.concatenate(samples, axis=0)
            

class BaseBenchmark(ABC):
    """Base class for benchmarks."""

    def __init__(self):
        self.name = self.get_name()


    def tune(self, n_trials, seed, benchmarks_dir):
        """Tune the benchmark."""


        def objective(trial):
            model = self.get_model_for_tuning(trial, seed)
            return self.train(model, tuning=True)['val_loss']
        
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler,direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_trial = study.best_trial
        best_hyperparameters = best_trial.params

        print('[Best hyperparameter configuration]:')
        print(best_hyperparameters)

        tuning_dir = os.path.join(benchmarks_dir, self.name, f'seed_{seed}', 'tuning')
        os.makedirs(tuning_dir, exist_ok=True)

        # Save best hyperparameters
        hyperparam_save_path = os.path.join(tuning_dir, f'hyperparameters.json')
        with open(hyperparam_save_path, 'w') as f:
            json.dump(best_hyperparameters, f)
        
        # Save optuna study
        study_save_path = os.path.join(tuning_dir, f'study_{seed}.pkl')
        with open(study_save_path, 'wb') as f:
            pickle.dump(study, f)

         # save optuna visualizations
        # fig = optuna.visualization.plot_intermediate_values(study)
        # fig.write_image(os.path.join(tuning_dir, 'intermediate_values.png'))

        # fig = optuna.visualization.plot_optimization_history(study)
        # fig.write_image(os.path.join(tuning_dir, 'optimization_history.png'))

        # fig = optuna.visualization.plot_param_importances(study)
        # fig.write_image(os.path.join(tuning_dir, 'param_importance.png'))

        print(f'[Tuning complete], saved tuning results to {tuning_dir}')

        return best_hyperparameters

    def run(self, dataset: XTYDataset, train_indices, val_indices, test_indices, n_trials, n_tune, seed, benchmarks_dir, **kwargs):
        """Run the benchmark."""
        self.benchmarks_dir = benchmarks_dir

        # Create a numpy random generator
        rng = np.random.default_rng(seed)

        # Seed for tuning
        tuning_seed = seed

        # Generate seeds for training
        training_seeds = rng.integers(0, 2**31 - 1, size=n_trials)
        training_seeds = [s.item() for s in training_seeds]

        # Prepare the data
        self.prepare_data(dataset, train_indices, val_indices, test_indices)

        # Tune the model
        best_hyperparameters = self.tune(n_trials=n_tune, seed=tuning_seed, benchmarks_dir=benchmarks_dir)

        print(f"[Training for {n_trials} trials with best hyperparameters]")

        # Train the model n_trials times
        test_losses = []
        for i in range(n_trials):
            print(f"[Training trial {i+1}/{n_trials}] seed: {training_seeds[i]}")
            model = self.get_final_model(best_hyperparameters, training_seeds[i])
            test_loss = self.train(model)['test_loss']
            test_losses.append(test_loss)

        # Save the losses
        df = pd.DataFrame({'seed':training_seeds,'test_loss': test_losses})
        results_folder = os.path.join(benchmarks_dir, self.name, f'seed_{seed}', 'results')
        os.makedirs(results_folder, exist_ok=True)
        test_losses_save_path = os.path.join(results_folder, f'test_losses.csv')
        df.to_csv(test_losses_save_path, index=False)
        
        return test_losses

    @abstractmethod
    def prepare_data(self, dataset: XTYDataset, train_indices, val_indices, test_indices):
        """Prepare the data for the benchmark."""
        pass

    @abstractmethod
    def train(self, model, tuning=False):
        """
        Train the benchmark. Returns a dictionary with train, validation, and test loss
        Returns:
            dict: Dictionary with train, validation, and test loss {'train_loss': float, 'val_loss': float, 'test_loss': float}
        """
        pass
       
    @abstractmethod
    def get_model_for_tuning(self, trial, seed):
        """Get the model."""
        pass

    @abstractmethod
    def get_final_model(self, hyperparameters, seed):
        """Get the model."""
        pass
    
    @abstractmethod
    def get_name(self):
        """Get the name of the benchmark."""
        pass
       

def run_all_benchmarks(dataset: XTYDataset, benchmarks: list, dataset_split = [0.7,0.15,0.15], n_trials=10, n_tune=100, seed=0, benchmarks_dir='results/benchmarks'):

    # Validate benchmarks is a list of objects inheriting from BaseBenchmark
    for benchmark in benchmarks:
        if not isinstance(benchmark, BaseBenchmark):
            raise TypeError(f'benchmarks must be a list of objects inheriting from BaseBenchmark, but {benchmark} is not.')


    # Generate train, validation, and test indices
    gen = np.random.default_rng(seed)
    n = len(dataset)
    train_indices = gen.choice(n, int(n*dataset_split[0]), replace=False)
    train_indices = [i.item() for i in train_indices]
    print(train_indices)
    val_indices = gen.choice(list(set(range(n)) - set(train_indices)), int(n*dataset_split[1]), replace=False)
    val_indices = [i.item() for i in val_indices]
    test_indices = list(set(range(n)) - set(train_indices) - set(val_indices))
    results = {'name':[],'mean':[],'std':[]}


    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    benchmarks_dir = os.path.join(benchmarks_dir, timestamp)

    for benchmark in benchmarks:
        losses = benchmark.run(dataset, train_indices, val_indices, test_indices, n_trials=n_trials, n_tune=n_tune, seed=seed, benchmarks_dir=benchmarks_dir, timestamp=timestamp)
        results['name'].append(benchmark.name)
        results['mean'].append(np.mean(losses))
        results['std'].append(np.std(losses))

    # Create a dataframe with the results
    df = pd.DataFrame(results)

    # Save the results
    df.to_csv(os.path.join(benchmarks_dir, 'results.csv'), index=False)

    return df
    

class GAM(BaseBenchmark):
    """GAM benchmark."""
    
    def get_name(self):
        return 'GAM'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        parameters = {
            'max_bins': trial.suggest_int('max_bins', 128, 512),
            'validation_size': trial.suggest_float('validation_size', 0.1, 0.3),
            'outer_bags': trial.suggest_int('outer_bags', 4, 16),
            'inner_bags': trial.suggest_int('inner_bags', 0, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'max_leaves': trial.suggest_int('max_leaves', 1, 6),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
        }
        return ExplainableBoostingRegressor(**parameters,interactions=0,random_state=seed)

    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        return ExplainableBoostingRegressor(**parameters,interactions=0,random_state=seed)
    
    def prepare_data(self, dataset: XTYDataset, train_indices, val_indices, test_indices):
        
        data_train = np.asfortranarray(dataset.get_single_matrix(train_indices))
        self.X_train = data_train[:,:-1]    
        self.y_train = data_train[:,-1]
        
        data_val = np.asfortranarray(dataset.get_single_matrix(val_indices))
        self.X_val = data_val[:,:-1]
        self.y_val = data_val[:,-1]

        # data_test = np.asfortranarray(dataset.get_single_matrix(test_indices))
        # self.X_test = data_test[:,:-1]
        # self.y_test = data_test[:,-1]

        self.test_samples = []
        for i in test_indices:
            X = dataset.get_single_matrix([i])[:,:-1]
            y = dataset.get_single_matrix([i])[:,-1]
            self.test_samples.append((X,y))
    
    def train(self, model, tuning=False):
        """Train model."""
        
        if not tuning:
            X_train = np.concatenate([self.X_train, self.X_val], axis=0)
            y_train = np.concatenate([self.y_train, self.y_val], axis=0)
        else:
            X_train = self.X_train
            y_train = self.y_train


        model.fit(X_train, y_train)

        train_loss = mean_squared_error(y_train, model.predict(X_train))
        val_loss = mean_squared_error(self.y_val, model.predict(self.X_val))
        
        test_loss = 0
        if not tuning:
            for sample in self.test_samples:
                X, y = sample
                y_pred = model.predict(X)
                err = mean_squared_error(y, y_pred)
                test_loss += err
            test_loss /= len(self.test_samples)

        return {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}

class XGBBenchmark(BaseBenchmark):
    """XGBoost benchmark."""
    
    def get_name(self):
        return 'XGB'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        parameters = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'gamma': trial.suggest_float('gamma', 1e-6, 1.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0)
        }
        return XGBRegressor(**parameters, random_state=seed)

    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        return XGBRegressor(**parameters, random_state=seed)
    
    
    def train(self, model, dataset, train_indices, val_indices, test_indices):
        """Train model."""
        data = dataset.get_single_matrix()
        X = data[:,:-1]
        y = data[:,-1]

        model.fit(X[train_indices,:], y[train_indices])
        

        train_loss = mean_squared_error(y[train_indices], model.predict(X[train_indices,:]))
        val_loss = mean_squared_error(y[val_indices], model.predict(X[val_indices,:]))
        test_loss = r2_score(y[test_indices], model.predict(X[test_indices,:]))


        return {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}    

class TTSBenchmark(BaseBenchmark):
    """TTS benchmark."""

    def __init__(self, config):
        self.config = config
        super().__init__()

    def get_name(self):
        return 'TTS'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        config = TuningConfig(trial,n_features=self.config.n_features, n_basis=self.config.n_basis, T=self.config.T, seed=self.config.seed)
        model = TTS(config)
        litmodel = LitTTS(config, model)
        tuning_callback = PyTorchLightningPruningCallback(trial, monitor='val_loss')
        return (litmodel, tuning_callback)
       
    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        encoder = {
            'hidden_sizes': [parameters[f'hidden_size_{i}'] for i in range(3)],
            'activation': parameters['activation'],
            'dropout_p': parameters['dropout_p']
        }
        training = {
            'batch_size': parameters['batch_size'],
            'lr': parameters['lr'],
            'weight_decay': parameters['weight_decay'],
            'optimizer': 'adam'
        }
        config = Config(n_features=self.config.n_features,
                        n_basis=self.config.n_basis,
                        T=self.config.T,
                        seed=seed,
                        encoder=encoder,
                        training=training,
                        dataloader_type=self.config.dataloader_type)
        model = TTS(config)
        litmodel = LitTTS(config, model)
        return litmodel

    def prepare_data(self, dataset, train_indices, val_indices, test_indices):
        X = dataset.X
        ts = dataset.ts
        ys = dataset.ys
        self.train_dataset = TTSDataset(self.config, (X[train_indices,:], [ts[i] for i in train_indices], [ys[i] for i in train_indices]))
        self.val_dataset = TTSDataset(self.config, (X[val_indices,:], [ts[i] for i in val_indices], [ys[i] for i in val_indices]))
        self.test_dataset = TTSDataset(self.config, (X[test_indices,:], [ts[i] for i in test_indices], [ys[i] for i in test_indices]))
        

    def train(self, model, tuning=False):
        """Train model."""
        if tuning:
            tuning_callback = model[1]
            model = model[0]
            log_dir = os.path.join(self.benchmarks_dir, self.name, f'seed_{model.config.seed}', 'tuning', 'logs')
        else:
            log_dir =  os.path.join(self.benchmarks_dir, self.name, f'seed_{model.config.seed}', 'training', 'logs')
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name='tts')
        
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
        # if tuning:
        #     callback_ls.append(tuning_callback)
        
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

        train_dataloader = create_dataloader(model.config, self.train_dataset, None, shuffle=True)
        val_dataloader = create_dataloader(model.config, self.val_dataset, None, shuffle=False)
        test_dataloader = create_dataloader(model.config, self.test_dataset, None, shuffle=False)
    

        trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader)

        train_loss = trainer.logged_metrics['train_loss']
        print(f"train_loss: {train_loss}")
        val_loss = early_stop_callback.best_score
        test_loss = 0

        if not tuning:
            results = trainer.test(model=model, dataloaders=test_dataloader)
            test_loss = results[0]['test_loss']

        return {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}