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
from xgboost import XGBRegressor

# from tts.training import training
# from tts.tuning import tuning

from tts.data import TTSDataset, create_dataloader, BaseDataset
from tts.config import TuningConfig, Config
from tts.model import TTS
from tts.lit_module import LitTTS
from tts.knot_selection import calculate_knot_placement
import pytorch_lightning as pl
import torch

def get_baseline(name, parameter_dict=None):
    class_name = name + 'Benchmark'
    return globals()[class_name](**parameter_dict)

            

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

        tuning_dir = os.path.join(benchmarks_dir, self.name, 'tuning')
        os.makedirs(tuning_dir, exist_ok=True)

        # Save best hyperparameters
        hyperparam_save_path = os.path.join(tuning_dir, f'hyperparameters.json')
        with open(hyperparam_save_path, 'w') as f:
            json.dump(best_hyperparameters, f)
        
        # Save optuna study
        study_save_path = os.path.join(tuning_dir, f'study_{seed}.pkl')
        with open(study_save_path, 'wb') as f:
            pickle.dump(study, f)

        # Save trials dataframe
        df = study.trials_dataframe()
        df.set_index('number', inplace=True)
        df_save_path = os.path.join(tuning_dir, f'trials_dataframe.csv')
        df.to_csv(df_save_path)

        # save optuna visualizations
        # fig = optuna.visualization.plot_intermediate_values(study)
        # fig.write_image(os.path.join(tuning_dir, 'intermediate_values.png'))

        # fig = optuna.visualization.plot_optimization_history(study)
        # fig.write_image(os.path.join(tuning_dir, 'optimization_history.png'))

        # fig = optuna.visualization.plot_param_importances(study)
        # fig.write_image(os.path.join(tuning_dir, 'param_importance.png'))

        print(f'[Tuning complete], saved tuning results to {tuning_dir}')

        return best_hyperparameters

    def run(self, dataset: BaseDataset, train_indices, val_indices, test_indices, n_trials, n_tune, seed, benchmarks_dir, **kwargs):
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
        if n_tune > 0:
            print(f"[Tuning for {n_tune} trials]")
            best_hyperparameters = self.tune(n_trials=n_tune, seed=tuning_seed, benchmarks_dir=benchmarks_dir)
        else:
            print(f"[No tuning, using default hyperparameters]")
            best_hyperparameters = None

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
        results_folder = os.path.join(benchmarks_dir, self.name, 'final')
        os.makedirs(results_folder, exist_ok=True)
        test_losses_save_path = os.path.join(results_folder, f'results.csv')
        df.to_csv(test_losses_save_path, index=False)
        
        return test_losses

    @abstractmethod
    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):
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


class MeanBenchmark(BaseBenchmark):
    """Mean benchmark."""
    
    def get_name(self):
        return 'Mean'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        return None
    
    def get_final_model(self, hyperparameters, seed):
        """Get the model."""
        return None

    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):
        """Prepare the data for the benchmark."""
        data_train = dataset.get_single_matrix(train_indices)
        self.X_train = data_train[:,:-1]    
        self.y_train = data_train[:,-1]
        
        data_val = dataset.get_single_matrix(val_indices)
        self.X_val = data_val[:,:-1]
        self.y_val = data_val[:,-1]

        self.test_samples = []
        for i in test_indices:
            X = dataset.get_single_matrix([i])[:,:-1]
            y = dataset.get_single_matrix([i])[:,-1]
            self.test_samples.append((X,y))
    
    def train(self, model, tuning=False):
        """
        Train the benchmark. Returns a dictionary with train, validation, and test loss
        Returns:
            dict: Dictionary with train, validation, and test loss {'train_loss': float, 'val_loss': float, 'test_loss': float}
        """
        if not tuning:
            X_train = np.concatenate([self.X_train, self.X_val], axis=0)
            y_train = np.concatenate([self.y_train, self.y_val], axis=0)
        else:
            X_train = self.X_train
            y_train = self.y_train

        y_mean = np.mean(y_train)

        train_loss = mean_squared_error(y_train, y_mean * np.ones_like(y_train))
        val_loss = mean_squared_error(self.y_val, y_mean * np.ones_like(self.y_val))
        
        test_loss = 0
        if not tuning:
            for sample in self.test_samples:
                X, y = sample
                y_pred = y_mean * np.ones_like(y)
                err = mean_squared_error(y, y_pred)
                test_loss += err
            test_loss /= len(self.test_samples)

        return {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}

class GAMBenchmark(BaseBenchmark):
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
        if parameters is None:
            return ExplainableBoostingRegressor(interactions=0,random_state=seed)
        else:
            return ExplainableBoostingRegressor(**parameters,interactions=0,random_state=seed)
    
    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):
        
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

        # Save the model to a pickle file
        if tuning:
            log_dir = os.path.join(self.benchmarks_dir, self.name, 'tuning', 'logs', f'seed_{model.random_state}')
        else:
            log_dir =  os.path.join(self.benchmarks_dir, self.name, 'final', 'logs', f'seed_{model.random_state}')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)

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
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'eta': trial.suggest_float('eta', 0.001, 0.1, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        }
        return XGBRegressor(**parameters, random_state=seed)

    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        if parameters is None:
            return XGBRegressor(random_state=seed)
        else:
            return XGBRegressor(**parameters, random_state=seed)
    
    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):
        
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

        # Save the model to a pickle file
        if tuning:
            log_dir = os.path.join(self.benchmarks_dir, self.name, 'tuning', 'logs', f'seed_{model.random_state}')
        else:
            log_dir =  os.path.join(self.benchmarks_dir, self.name, 'final', 'logs', f'seed_{model.random_state}')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)


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

class TTSBenchmark(BaseBenchmark):
    """TTS benchmark."""

    def __init__(self, config):
        self.config = config
        super().__init__()

    def get_name(self):
        return 'TTS'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        config = TuningConfig(
                            trial,
                            n_features=self.config.n_features,
                            n_basis=self.config.n_basis,
                            T=self.config.T,
                            seed=self.config.seed,
                            device=self.config.device,
                            num_epochs=self.config.num_epochs,
                            dataloader_type=self.config.dataloader_type,
                            internal_knots=self.config.internal_knots,
                            n_basis_tunable=self.config.n_basis_tunable,
                            dynamic_bias=self.config.dynamic_bias)
        litmodel = LitTTS(config)
        tuning_callback = PyTorchLightningPruningCallback(trial, monitor='val_loss')
        return (litmodel, tuning_callback)
       
    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        if parameters is not None:
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
            if self.config.n_basis_tunable:
                n_basis = parameters['n_basis']
            else:
                n_basis = self.config.n_basis

            config = Config(n_features=self.config.n_features,
                            n_basis=n_basis,
                            T=self.config.T,
                            seed=seed,
                            encoder=encoder,
                            training=training,
                            dataloader_type=self.config.dataloader_type,
                            device=self.config.device,
                            num_epochs=self.config.num_epochs,
                            internal_knots=self.config.internal_knots,
                            n_basis_tunable=self.config.n_basis_tunable,
                            dynamic_bias=self.config.dynamic_bias
                            )
        else:
            config = self.config

        litmodel = LitTTS(config)
        return litmodel

    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):
        X, ts, ys = dataset.get_X_ts_ys()

        if self.config.n_basis_tunable:
            # That means that we cannot precompute the internal knots and instantiate the datasets
            # Instead we will have to do it in the train function
            self.X_train = X[train_indices,:]
            self.ts_train = [ts[i] for i in train_indices]
            self.ys_train = [ys[i] for i in train_indices]
            self.X_val = X[val_indices,:]
            self.ts_val = [ts[i] for i in val_indices]
            self.ys_val = [ys[i] for i in val_indices]
            self.X_test = X[test_indices,:]
            self.ts_test = [ts[i] for i in test_indices]
            self.ys_test = [ys[i] for i in test_indices]
            return
        else: 
            if self.config.internal_knots is None:
                # We need to find the internal knots
                ts_train = [ts[i] for i in train_indices]
                ys_train = [ys[i] for i in train_indices]

                n_internal_knots = self.config.n_basis - 2

                internal_knots = calculate_knot_placement(ts_train, ys_train, n_internal_knots, T=self.config.T, seed=self.config.seed)
                print(f'Found internal knots: {internal_knots}')

                self.config.internal_knots = internal_knots

            self.train_dataset = TTSDataset(self.config, (X[train_indices,:], [ts[i] for i in train_indices], [ys[i] for i in train_indices]))
            self.val_dataset = TTSDataset(self.config, (X[val_indices,:], [ts[i] for i in val_indices], [ys[i] for i in val_indices]))
            self.test_dataset = TTSDataset(self.config, (X[test_indices,:], [ts[i] for i in test_indices], [ys[i] for i in test_indices]))
        

    def train(self, model, tuning=False):
        """Train model."""
        if tuning:
            tuning_callback = model[1]
            model = model[0]
            log_dir = os.path.join(self.benchmarks_dir, self.name, 'tuning', 'logs', f'seed_{model.config.seed}')
        else:
            log_dir =  os.path.join(self.benchmarks_dir, self.name, 'final', 'logs', f'seed_{model.config.seed}')

        if self.config.n_basis_tunable:
            # We need to create the datasets here becuase we could not do it in prepare_data
            
            # We need to find the internal knots
            n_internal_knots = model.config.n_basis - 2
            internal_knots = calculate_knot_placement(self.ts_train, self.ys_train, n_internal_knots, T=model.config.T, seed=self.config.seed)
            print(f'Found internal knots: {internal_knots}')

            model.config.internal_knots = internal_knots

            train_dataset = TTSDataset(model.config, (self.X_train, self.ts_train, self.ys_train))
            val_dataset = TTSDataset(model.config, (self.X_val, self.ts_val, self.ys_val))
            test_dataset = TTSDataset(model.config, (self.X_test, self.ts_test, self.ys_test))
        else:
            train_dataset = self.train_dataset
            val_dataset = self.val_dataset
            test_dataset = self.test_dataset

        
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir)

        # Create folder if does not exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Save config as a pickle file
        config = model.config
        with open(os.path.join(log_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        

        
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
            'enable_model_summary': False,
            'enable_progress_bar': False,
            'accelerator': config.device,
            'max_epochs': config.num_epochs,
            'logger': tb_logger,
            'check_val_every_n_epoch': 10,
            'log_every_n_steps': 1,
            'callbacks': callback_ls
        }

        trainer = pl.Trainer(**trainer_dict)

        train_dataloader = create_dataloader(model.config, train_dataset, None, shuffle=True)
        val_dataloader = create_dataloader(model.config, val_dataset, None, shuffle=False)
        test_dataloader = create_dataloader(model.config, test_dataset, None, shuffle=False)
    

        trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader)

        print(f"Finished after {trainer.current_epoch} epochs.")

        train_loss = trainer.logged_metrics['train_loss']
        print(f"train_loss: {train_loss}")
        val_loss = early_stop_callback.best_score
        test_loss = 0

        if not tuning:
            results = trainer.test(model=model, dataloaders=test_dataloader)
            test_loss = results[0]['test_loss']

        return {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}