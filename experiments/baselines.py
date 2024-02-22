import sys
sys.path.append('../')
from datetime import datetime
import os
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod, abstractproperty
import json
import pickle
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
import sympy
from xgboost import XGBRegressor
import pysindy as ps

# from tts.training import training
# from tts.tuning import tuning

from tts.data import TTSDataset, create_dataloader, BaseDataset
from tts.config import TuningConfig, Config
from tts.model import TTS
from tts.lit_module import LitTTS
from tts.knot_selection import calculate_knot_placement
import pytorch_lightning as pl
import torch

from experiments.baseline_implementations.rnn.config import RNNConfig, RNNTuningConfig
from experiments.baseline_implementations.rnn.lit_module import LitRNN, LitDeltaTRNN

from lightgbm import LGBMRegressor
import lightgbm as lgb


from pysr import PySRRegressor

def get_baseline(name, parameter_dict=None):
    class_name = name + 'Benchmark'
    return globals()[class_name](**parameter_dict)

class YNormalizer:
    """Normalize y values."""

    def __init__(self):
        self.fitted = False

    def fit(self, ys):
        """Fit normalization parameters."""
        if self.fitted:
            raise RuntimeError('Already fitted.')
        if isinstance(ys, list):
            Y = np.concatenate(ys, axis=0)
        else:
            Y = ys
        self.y_mean = np.mean(Y)
        self.y_std = np.std(Y)
        self.fitted = True

    def transform(self, ys):
        """Normalize y values."""
        if not self.fitted:
            raise RuntimeError('Call fit before transform.')
        if isinstance(ys, list):
            return [(y - self.y_mean) / self.y_std for y in ys]
        else:
            return (ys - self.y_mean) / self.y_std
    
    def inverse_transform(self, ys):
        """Denormalize y values."""
        if not self.fitted:
            raise RuntimeError('Call fit before transform.')
        if isinstance(ys, list):
            return [y * self.y_std + self.y_mean for y in ys]
        else:
            return ys * self.y_std + self.y_mean
    
    def save(self, path):
        """Save normalization parameters using json"""
        y_normalization = {'y_mean': self.y_mean, 'y_std': self.y_std}
        full_path = os.path.join(path, "y_normalizer.json")
        with open(full_path, 'w') as f:
            json.dump(y_normalization, f)
    
    def load(path):
        """Load normalization parameters using json"""
        with open(path, 'r') as f:
            y_normalization = json.load(f)
        ynormalizer = YNormalizer()
        ynormalizer.set_params(y_normalization['y_mean'], y_normalization['y_std'])
        return ynormalizer

    def load_from_benchmark(timestamp, name, benchmark_dir='benchmarks'):
        """Load normalization parameters from a benchmark."""
        path = os.path.join(benchmark_dir, timestamp, name, 'y_normalizer.json')
        return YNormalizer.load(path)

    def set_params(self, y_mean, y_std):
        """Set normalization parameters."""
        self.y_mean = y_mean
        self.y_std = y_std
        self.fitted = True

    def fit_transform(self, ys):
        """Fit normalization parameters and normalize y values."""
        self.fit(ys)
        return self.transform(ys)
    

def _pad_to_shape(a, shape):
    """
    This function pads a 1D, 2D or 3D numpy array with zeros to a specified shape
    Args:
        a: a numpy array
        shape: a tuple of integers
    Returns:
        a numpy array of shape shape
    """
    if a.shape == shape:
        return a
    if len(a.shape) == 1:
        assert a.shape[0] <= shape[0]
        b = np.zeros(shape)
        b[:a.shape[0]] = a
    elif len(a.shape) == 2:
        assert a.shape[0] <= shape[0]
        assert a.shape[1] <= shape[1]
        b = np.zeros(shape)
        b[:a.shape[0], :a.shape[1]] = a
    elif len(a.shape) == 3:
        assert a.shape[0] <= shape[0]
        assert a.shape[1] <= shape[1]
        assert a.shape[2] <= shape[2]
        b = np.zeros(shape)
        b[:a.shape[0], :a.shape[1], :a.shape[2]] = a
    return b


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


class PySRBenchmark(BaseBenchmark):
    """PySR benchmark."""

    def __init__(self, timeout_in_seconds=60, n_features=6):
        super().__init__()
        self.timeout_in_seconds = timeout_in_seconds
        self.n_features = n_features
    
    def get_name(self):
        return 'PySR'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        model = PySRRegressor(
        niterations=10e9,
        binary_operators=["+", "*", "/", "-"],
        unary_operators=[
            "sin",
            "exp",
            "logp(x) = log(abs(x)+1.0f-3)",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"logp": lambda x: sympy.log(sympy.Abs(x) + 1e-3)},
        maxsize=max(20,self.n_features * 3),
        timeout_in_seconds=self.timeout_in_seconds,
        deterministic=True,
        random_state=seed,
        procs=0,
        multithreading=False
        )
        return model

    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        model = PySRRegressor(
        niterations=10e9,
        binary_operators=["+", "*", "/", "-"],
        unary_operators=[
            "sin",
            "exp",
            "logp(x) = log(abs(x)+1.0f-3)",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"logp": lambda x: sympy.log(sympy.Abs(x) + 1e-3)},
        maxsize=max(20,self.n_features * 3),
        timeout_in_seconds=self.timeout_in_seconds,
        deterministic=True,
        random_state=seed,
        procs=0,
        multithreading=False
        )
        return model
    
    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):
        
        data_train = dataset.get_single_matrix(train_indices)
        self.X_train = data_train.iloc[:,:-1]    
        self.y_train = data_train.iloc[:,-1].to_numpy()
        
        data_val = dataset.get_single_matrix(val_indices)
        self.X_val = data_val.iloc[:,:-1]
        self.y_val = data_val.iloc[:,-1].to_numpy()

        column_transformer = dataset.get_default_column_transformer()
        y_normalizer = YNormalizer()

        self.X_train = column_transformer.fit_transform(self.X_train)
        self.y_train = y_normalizer.fit_transform(self.y_train)

        self.X_val = column_transformer.transform(self.X_val)
        self.y_val = y_normalizer.transform(self.y_val)

        self.test_samples = []
        for i in test_indices:
            X = dataset.get_single_matrix([i]).iloc[:,:-1]
            X = column_transformer.transform(X)
            y = dataset.get_single_matrix([i]).iloc[:,-1].to_numpy()
            y = y_normalizer.transform(y)
            self.test_samples.append((X,y))

        # Save the transformer using joblib
        os.makedirs(os.path.join(self.benchmarks_dir, self.name), exist_ok=True)
        joblib.dump(column_transformer, os.path.join(self.benchmarks_dir, self.name, 'column_transformer.joblib'))

        # Save y_normalizer
        y_normalizer.save(os.path.join(self.benchmarks_dir, self.name))
    
    def train(self, model, tuning=False):
        """Train model."""
        
        X_train = self.X_train
        y_train = self.y_train


        model.fit(X_train, y_train)

        # Save the model to a pickle file
        if tuning:
            log_dir = os.path.join(self.benchmarks_dir, self.name, 'tuning', 'logs', f'seed_{model.random_state}')
        else:
            log_dir =  os.path.join(self.benchmarks_dir, self.name, 'final', 'logs', f'seed_{model.random_state}')
        os.makedirs(log_dir, exist_ok=True)
        if not tuning:
            with open(os.path.join(log_dir, 'model.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        # Save the equations
        equations_in_latex = model.latex()
        with open(os.path.join(log_dir, 'equations.txt'), 'w') as f:
            f.write(equations_in_latex)

        train_loss = 0
        val_losses = []
        
        n_equations = len(model.equations_)

        for i in range(n_equations):
            val_losses.append(mean_squared_error(self.y_val, model.predict(self.X_val, index=i)))
        
        val_loss = min(val_losses)
        equation_index = np.argmin(val_losses)

        chosen_equation_in_latex = model.latex(index=equation_index)
        with open(os.path.join(log_dir, 'chosen_equation.txt'), 'w') as f:
            f.write(chosen_equation_in_latex)

        
        test_loss = 0
        if not tuning:
            for sample in self.test_samples:
                X, y = sample
                y_pred = model.predict(X,index=equation_index)
                err = mean_squared_error(y, y_pred)
                test_loss += err
            test_loss /= len(self.test_samples)

        return {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}


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
        self.X_train = data_train.iloc[:,:-1]    
        self.y_train = data_train.iloc[:,-1].to_numpy()
        
        data_val = dataset.get_single_matrix(val_indices)
        self.X_val = data_val.iloc[:,:-1]
        self.y_val = data_val.iloc[:,-1].to_numpy()

        column_transformer = dataset.get_default_column_transformer()
        y_normalizer = YNormalizer()

        self.X_train = column_transformer.fit_transform(self.X_train)
        self.y_train = y_normalizer.fit_transform(self.y_train)

        self.X_val = column_transformer.transform(self.X_val)
        self.y_val = y_normalizer.transform(self.y_val)

        self.test_samples = []
        for i in test_indices:
            X = dataset.get_single_matrix([i]).iloc[:,:-1]
            X = column_transformer.transform(X)
            y = dataset.get_single_matrix([i]).iloc[:,-1].to_numpy()
            y = y_normalizer.transform(y)
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


class SINDyBenchmark(BaseBenchmark):
    """SINDy benchmark."""
    def __init__(self):
        super().__init__()
        self.fitted = False
    
    def get_name(self):
        return 'SINDy'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        
        optimizer_threshold = trial.suggest_float('optimizer_threshold', 1e-3, 1e-1, log=True)
        optimizer = ps.STLSQ(threshold=optimizer_threshold)

        differentiation_kind = trial.suggest_categorical('differentiation_kind', ['finite_difference', 'spline', 'trend_filtered'])
        if differentiation_kind == 'finite_difference':
            k = trial.suggest_int('k', 1, 5)
            differentiation_method = ps.SINDyDerivative(kind='finite_difference', k=k)
        elif differentiation_kind == 'spline':
            s = trial.suggest_float('s', 1e-3, 1, log=True)
            differentiation_method = ps.SINDyDerivative(kind='spline', s=s)
        elif differentiation_kind == 'trend_filtered':
            order = trial.suggest_int('order', 0, 2)
            alpha = trial.suggest_float('alpha', 1e-4, 1, log=True)
            differentiation_method = ps.SINDyDerivative(kind='trend_filtered', order=order, alpha=alpha)
        elif differentiation_kind == 'smoothed_finite_difference':
            window_length = trial.suggest_int('window_length', 1, 5)
            differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={"window_length":window_length})
        
        lib_poly = ps.PolynomialLibrary(degree=2, include_interaction=True, include_bias=True)

        model = ps.SINDy(optimizer=optimizer, differentiation_method=differentiation_method, feature_library=lib_poly)

        return model

    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        if parameters is None:
            return ps.SINDy(optimizer=ps.STLSQ(threshold=1e-3), differentiation_method=ps.SINDyDerivative(kind='finite_difference', k=2), feature_library=ps.PolynomialLibrary(degree=2, include_interaction=True, include_bias=True))
        else:
            optimizer_threshold = parameters['optimizer_threshold']
            optimizer = ps.STLSQ(threshold=optimizer_threshold)
            differentiation_kind = parameters['differentiation_kind']
            if differentiation_kind == 'finite_difference':
                k = parameters['k']
                differentiation_method = ps.SINDyDerivative(kind='finite_difference', k=k)
            elif differentiation_kind == 'spline':
                s = parameters['s']
                differentiation_method = ps.SINDyDerivative(kind='spline', s=s)
            elif differentiation_kind == 'trend_filtered':
                order = parameters['order']
                alpha = parameters['alpha']
                differentiation_method = ps.SINDyDerivative(kind='trend_filtered', order=order, alpha=alpha)
            elif differentiation_kind == 'smoothed_finite_difference':
                window_length = parameters['window_length']
                differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={"window_length":window_length})
            lib_poly = ps.PolynomialLibrary(degree=2, include_interaction=True, include_bias=True)
            return ps.SINDy(optimizer=optimizer, differentiation_method=differentiation_method, feature_library=lib_poly)
    
    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):
        
        X, ts, ys = dataset.get_X_ts_ys()

        # Fit the transformer
        column_transformer = dataset.get_default_column_transformer()
        y_normalizer = YNormalizer()

        column_transformer.fit(X.iloc[train_indices,:])
        y_normalizer.fit([ys[i] for i in train_indices])

        self.X_train = []
        self.ts_train = [ts[i] for i in train_indices]
        self.ys_train = y_normalizer.transform([ys[i] for i in train_indices])

        self.X_val = []
        self.ts_val = [ts[i] for i in val_indices]
        self.ys_val = y_normalizer.transform([ys[i] for i in val_indices])

        self.X_test = []
        self.ts_test = [ts[i] for i in test_indices]
        self.ys_test = y_normalizer.transform([ys[i] for i in test_indices])

        for i in train_indices:
            num_measurements = len(ys[i])
            features = column_transformer.transform(X.iloc[[i],:])
            features = np.tile(features, (num_measurements, 1))
            self.X_train.append(features)
        
        for i in val_indices:
            num_measurements = len(ys[i])
            features = column_transformer.transform(X.iloc[[i],:])
            features = np.tile(features, (num_measurements, 1))
            self.X_val.append(features)

        for i in test_indices:
            num_measurements = len(ys[i])
            features = column_transformer.transform(X.iloc[[i],:])
            features = np.tile(features, (num_measurements, 1))
            self.X_test.append(features)

        # Save the transformer using joblib
        os.makedirs(os.path.join(self.benchmarks_dir, self.name), exist_ok=True)
        joblib.dump(column_transformer, os.path.join(self.benchmarks_dir, self.name, 'column_transformer.joblib'))

        # Save y_normalizer
        y_normalizer.save(os.path.join(self.benchmarks_dir, self.name))
    
    def train(self, model, tuning=False):
        """Train model."""

        INF = 1.0e9

        if not tuning:
            if self.fitted:
                return self.results # We can do it because the algorithm is deterministic

        # print("Fitting model")
        try:
            model.fit(self.ys_train, t=self.ts_train,u=self.X_train, multiple_trajectories=True)
        except Exception as e:
            print(e)
            return {'train_loss': INF, 'val_loss': INF, 'test_loss': INF}
        # print("Model fitted")

        # Save the model to a pickle file
        if tuning:
            log_dir = os.path.join(self.benchmarks_dir, self.name, 'tuning', 'logs')
        else:
            log_dir =  os.path.join(self.benchmarks_dir, self.name, 'final', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        if not tuning:
            with open(os.path.join(log_dir, 'model.pkl'), 'wb') as f:
                pickle.dump(model, f)

        def get_control_function(X):

            def u(t):
                return X[0,:]

            return u
        
        # train_losses = []
        # for X, t, y in zip(self.X_train, self.ts_train, self.ys_train):
        #     u = get_control_function(X)
        #     y_pred = model.simulate(y[[0]], t=t, u=u)
        #     train_losses.append(mean_squared_error(y, y_pred))
        # train_loss = np.mean(train_losses)
        
        def clip_to_finite(y):
            return np.clip(y,-INF,INF)

        train_loss = 0

        val_loss = 0
        if tuning:
            val_losses = []
            for X, t, y in zip(self.X_val, self.ts_val, self.ys_val):
                try:
                    # control_input = X[[0],:]
                    # def dydt(yt,t):
                    #     # print(yt.shape)
                    #     res = model.predict(yt.reshape(1,1),u=control_input).flatten()
                    #     print(res)
                    #     return res
                    # y_pred = odeint(dydt, y[0], t)
                    # print("Simulating")
                    u = get_control_function(X)
                    y_pred = clip_to_finite(model.simulate(y[[0]], t=t, u=u, integrator='odeint'))
                except Exception as e:
                    print(e)
                    y_pred = np.zeros_like(y)
                val_losses.append(min(INF,mean_squared_error(y, y_pred)))
            val_loss = np.mean(val_losses)
        
        test_loss = 0
        if not tuning:
            test_losses = []
            for X, t, y in zip(self.X_test, self.ts_test, self.ys_test):
                try:
                    # control_input = X[[0],:]
                    # def dydt(yt,t):
                    #     # print(yt.shape)
                    #     res = model.predict(yt.reshape(1,1),u=control_input).flatten()
                    #     print(res.shape)
                    #     return model.predict(yt.reshape(1,1),u=control_input).flatten()
                    # y_pred = odeint(dydt, y[0], t)
                    u = get_control_function(X)
                    y_pred = clip_to_finite(model.simulate(y[[0]], t=t, u=u, integrator='odeint'))
                except Exception as e:
                    print(e)
                    y_pred = np.zeros_like(y)
                test_losses.append(min(INF,mean_squared_error(y, y_pred)))
            test_loss = np.mean(test_losses)
        if not tuning:
            self.fitted = True
            self.results = {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}
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
        
        data_train = dataset.get_single_matrix(train_indices)
        self.X_train = data_train.iloc[:,:-1]    
        self.y_train = data_train.iloc[:,-1].to_numpy()
        
        data_val = dataset.get_single_matrix(val_indices)
        self.X_val = data_val.iloc[:,:-1]
        self.y_val = data_val.iloc[:,-1].to_numpy()

        column_transformer = dataset.get_default_column_transformer()
        y_normalizer = YNormalizer()

        self.X_train = column_transformer.fit_transform(self.X_train)
        self.y_train = y_normalizer.fit_transform(self.y_train)

        self.X_val = column_transformer.transform(self.X_val)
        self.y_val = y_normalizer.transform(self.y_val)

        self.test_samples = []
        for i in test_indices:
            X = dataset.get_single_matrix([i]).iloc[:,:-1]
            X = column_transformer.transform(X)
            y = dataset.get_single_matrix([i]).iloc[:,-1].to_numpy()
            y = y_normalizer.transform(y)
            self.test_samples.append((X,y))

        # Save the transformer using joblib
        os.makedirs(os.path.join(self.benchmarks_dir, self.name), exist_ok=True)
        joblib.dump(column_transformer, os.path.join(self.benchmarks_dir, self.name, 'column_transformer.joblib'))

        # Save y_normalizer
        y_normalizer.save(os.path.join(self.benchmarks_dir, self.name))
    
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
        if not tuning:
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
        
        data_train = dataset.get_single_matrix(train_indices)
        self.X_train = data_train.iloc[:,:-1]    
        self.y_train = data_train.iloc[:,-1].to_numpy()
        
        data_val = dataset.get_single_matrix(val_indices)
        self.X_val = data_val.iloc[:,:-1]
        self.y_val = data_val.iloc[:,-1].to_numpy()

        column_transformer = dataset.get_default_column_transformer()
        y_normalizer = YNormalizer()

        self.X_train = column_transformer.fit_transform(self.X_train)
        self.y_train = y_normalizer.fit_transform(self.y_train)

        self.X_val = column_transformer.transform(self.X_val)
        self.y_val = y_normalizer.transform(self.y_val)

        self.test_samples = []
        for i in test_indices:
            X = dataset.get_single_matrix([i]).iloc[:,:-1]
            X = column_transformer.transform(X)
            y = dataset.get_single_matrix([i]).iloc[:,-1].to_numpy()
            y = y_normalizer.transform(y)
            self.test_samples.append((X,y))

        # Save the transformer using joblib
        os.makedirs(os.path.join(self.benchmarks_dir, self.name), exist_ok=True)
        joblib.dump(column_transformer, os.path.join(self.benchmarks_dir, self.name, 'column_transformer.joblib'))

        # Save y_normalizer
        y_normalizer.save(os.path.join(self.benchmarks_dir, self.name))
    
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
        if not tuning:
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
    

class CatBoostBenchmark(BaseBenchmark):
    """CatBoost benchmark."""
    
    def get_name(self):
        return 'CatBoost'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        parameters = {
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 10, 50, step=10),
        }

        return CatBoostRegressor(**parameters, random_seed=seed)

    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        if parameters is None:
            return CatBoostRegressor(random_state=seed)
        else:
            return CatBoostRegressor(**parameters, random_state=seed)
    
    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):
        
        data_train = dataset.get_single_matrix(train_indices)
        self.X_train = data_train.iloc[:,:-1]    
        self.y_train = data_train.iloc[:,-1].to_numpy()
        
        data_val = dataset.get_single_matrix(val_indices)
        self.X_val = data_val.iloc[:,:-1]
        self.y_val = data_val.iloc[:,-1].to_numpy()

        column_transformer = dataset.get_default_column_transformer(keep_categorical=True)
        y_normalizer = YNormalizer()

        self.X_train = column_transformer.fit_transform(self.X_train)
        self.y_train = y_normalizer.fit_transform(self.y_train)

        self.X_val = column_transformer.transform(self.X_val)
        self.y_val = y_normalizer.transform(self.y_val)

        self.categorical_indices = [i for i, name in enumerate(dataset.get_feature_names()) if dataset.get_feature_type(name) == 'categorical']

        self.X_train = pd.DataFrame(self.X_train, columns=dataset.get_feature_names() + ['t'])
        for i in self.categorical_indices:
            self.X_train.iloc[:,i] = self.X_train.iloc[:,i].astype(int).astype('category')
        
        self.X_val = pd.DataFrame(self.X_val, columns=dataset.get_feature_names() + ['t'])
        for i in self.categorical_indices:
            self.X_val.iloc[:,i] = self.X_val.iloc[:,i].astype(int).astype('category')
        

        self.test_samples = []
        for i in test_indices:
            X = dataset.get_single_matrix([i]).iloc[:,:-1]
            X = column_transformer.transform(X)
            y = dataset.get_single_matrix([i]).iloc[:,-1].to_numpy()
            y = y_normalizer.transform(y)
            X = pd.DataFrame(X, columns=dataset.get_feature_names() + ['t'])
            for i in self.categorical_indices:
                X.iloc[:,i] = X.iloc[:,i].astype(int).astype('category')
            self.test_samples.append((X,y))

        # Save the transformer using joblib
        os.makedirs(os.path.join(self.benchmarks_dir, self.name), exist_ok=True)
        joblib.dump(column_transformer, os.path.join(self.benchmarks_dir, self.name, 'column_transformer.joblib'))

        # Save y_normalizer
        y_normalizer.save(os.path.join(self.benchmarks_dir, self.name))
    
    def train(self, model, tuning=False):
        """Train model."""
        
        # if not tuning:
        #     X_train = np.concatenate([self.X_train, self.X_val], axis=0)
        #     y_train = np.concatenate([self.y_train, self.y_val], axis=0)
        # else:
        #     X_train = self.X_train
        #     y_train = self.y_train

        model.fit(X=self.X_train, y=self.y_train, cat_features=self.categorical_indices, eval_set=(self.X_val, self.y_val), verbose=False)

        # Save the model to a pickle file
        if tuning:
            log_dir = os.path.join(self.benchmarks_dir, self.name, 'tuning', 'logs', f'seed_{model.random_seed_}')
        else:
            log_dir =  os.path.join(self.benchmarks_dir, self.name, 'final', 'logs', f'seed_{model.random_seed_}')
        os.makedirs(log_dir, exist_ok=True)
        if not tuning:
            with open(os.path.join(log_dir, 'model.pkl'), 'wb') as f:
                pickle.dump(model, f)


        train_loss = mean_squared_error(self.y_train, model.predict(self.X_train))
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

class LGBMBenchmark(BaseBenchmark):
    """LightGBM benchmark."""
    
    def get_name(self):
        return 'LGBM'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        if_alpha = trial.suggest_categorical('if_alpha', [True, False])
        if_lambda = trial.suggest_categorical('if_lambda', [True, False])
        parameters = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 10.0, log=True),
            'n_estimators': trial.suggest_categorical('n_estimators', [10, 50, 100, 200, 500, 1000, 2000, 5000]),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            # 'max_depth': trial.suggest_categorical('max_depth', [-1, 5, 10, 20, 50]),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-7, 10.0, log=True) if if_alpha else 0.0,
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-7, 10.0, log=True) if if_lambda else 0.0,
            'n_jobs': -1,
        }
    
        return LGBMRegressor(**parameters, random_state=seed)

    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        if parameters is None:
            return LGBMRegressor(random_state=seed, n_jobs=-1)
        else:
            if 'if_alpha' in parameters:
                del parameters['if_alpha']
            if 'if_lambda' in parameters:
                del parameters['if_lambda']

            return LGBMRegressor(**parameters, random_state=seed)
    
    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):
        
        data_train = dataset.get_single_matrix(train_indices)
        self.X_train = data_train.iloc[:,:-1]    
        self.y_train = data_train.iloc[:,-1].to_numpy()
        
        data_val = dataset.get_single_matrix(val_indices)
        self.X_val = data_val.iloc[:,:-1]
        self.y_val = data_val.iloc[:,-1].to_numpy()

        column_transformer = dataset.get_default_column_transformer(keep_categorical=True)
        y_normalizer = YNormalizer()

        self.X_train = column_transformer.fit_transform(self.X_train)
        self.y_train = y_normalizer.fit_transform(self.y_train)

        self.X_val = column_transformer.transform(self.X_val)
        self.y_val = y_normalizer.transform(self.y_val)

        self.categorical_indices = [i for i, name in enumerate(dataset.get_feature_names()) if dataset.get_feature_type(name) == 'categorical']

        self.test_samples = []
        for i in test_indices:
            X = dataset.get_single_matrix([i]).iloc[:,:-1]
            X = column_transformer.transform(X)
            y = dataset.get_single_matrix([i]).iloc[:,-1].to_numpy()
            y = y_normalizer.transform(y)
            self.test_samples.append((X,y))

        # Save the transformer using joblib
        os.makedirs(os.path.join(self.benchmarks_dir, self.name), exist_ok=True)
        joblib.dump(column_transformer, os.path.join(self.benchmarks_dir, self.name, 'column_transformer.joblib'))

        # Save y_normalizer
        y_normalizer.save(os.path.join(self.benchmarks_dir, self.name))
    
    def train(self, model, tuning=False):
        """Train model."""
        
        # if not tuning:
        #     X_train = np.concatenate([self.X_train, self.X_val], axis=0)
        #     y_train = np.concatenate([self.y_train, self.y_val], axis=0)
        # else:
        #     X_train = self.X_train
        #     y_train = self.y_train
        callbacks = [lgb.early_stopping(10, verbose=0), lgb.log_evaluation(period=0)]

        model.fit(X=self.X_train, y=self.y_train, categorical_feature=self.categorical_indices, eval_set=(self.X_val, self.y_val), callbacks=callbacks, verbose=0)

        # Save the model to a pickle file
        if tuning:
            log_dir = os.path.join(self.benchmarks_dir, self.name, 'tuning', 'logs', f'seed_{model.get_params()["random_state"]}')
        else:
            log_dir =  os.path.join(self.benchmarks_dir, self.name, 'final', 'logs', f'seed_{model.get_params()["random_state"]}')
        os.makedirs(log_dir, exist_ok=True)
        if not tuning:
            with open(os.path.join(log_dir, 'model.pkl'), 'wb') as f:
                pickle.dump(model, f)


        train_loss = mean_squared_error(self.y_train, model.predict(self.X_train))
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
    


class DecisionTreeBenchmark(BaseBenchmark):
    """Decision Tree benchmark."""
    
    def get_name(self):
        return 'DecisionTree'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        parameters = {
            'max_depth': 5,
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 32),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 32),
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'absolute_error']),
        }
        return DecisionTreeRegressor(**parameters, random_state=seed)

    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        if parameters is None:
            return DecisionTreeRegressor(max_depth=5,random_state=seed)
        else:
            return DecisionTreeRegressor(max_depth=5,**parameters, random_state=seed)
    
    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):
        
        data_train = dataset.get_single_matrix(train_indices)
        self.X_train = data_train.iloc[:,:-1]    
        self.y_train = data_train.iloc[:,-1].to_numpy()
        
        data_val = dataset.get_single_matrix(val_indices)
        self.X_val = data_val.iloc[:,:-1]
        self.y_val = data_val.iloc[:,-1].to_numpy()

        column_transformer = dataset.get_default_column_transformer()
        y_normalizer = YNormalizer()

        self.X_train = column_transformer.fit_transform(self.X_train)
        self.y_train = y_normalizer.fit_transform(self.y_train)

        self.X_val = column_transformer.transform(self.X_val)
        self.y_val = y_normalizer.transform(self.y_val)

        self.test_samples = []
        for i in test_indices:
            X = dataset.get_single_matrix([i]).iloc[:,:-1]
            X = column_transformer.transform(X)
            y = dataset.get_single_matrix([i]).iloc[:,-1].to_numpy()
            y = y_normalizer.transform(y)
            self.test_samples.append((X,y))

        # Save the transformer using joblib
        os.makedirs(os.path.join(self.benchmarks_dir, self.name), exist_ok=True)
        joblib.dump(column_transformer, os.path.join(self.benchmarks_dir, self.name, 'column_transformer.joblib'))

        # Save y_normalizer
        y_normalizer.save(os.path.join(self.benchmarks_dir, self.name))
    
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
        if not tuning:
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
    

class ElasticNetBenchmark(BaseBenchmark):
    """Decision Tree benchmark."""
    
    def get_name(self):
        return 'Linear'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        parameters = {
            'alpha': trial.suggest_float('alpha', 0.0, 1.0),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
        }
        return ElasticNet(**parameters, random_state=seed)

    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        if parameters is None:
            return ElasticNet(random_state=seed)
        else:
            return ElasticNet(**parameters, random_state=seed)
    
    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):
        
        data_train = dataset.get_single_matrix(train_indices)
        self.X_train = data_train.iloc[:,:-1]    
        self.y_train = data_train.iloc[:,-1].to_numpy()
        
        data_val = dataset.get_single_matrix(val_indices)
        self.X_val = data_val.iloc[:,:-1]
        self.y_val = data_val.iloc[:,-1].to_numpy()

        column_transformer = dataset.get_default_column_transformer()
        y_normalizer = YNormalizer()

        self.X_train = column_transformer.fit_transform(self.X_train)
        self.y_train = y_normalizer.fit_transform(self.y_train)

        self.X_val = column_transformer.transform(self.X_val)
        self.y_val = y_normalizer.transform(self.y_val)

        self.test_samples = []
        for i in test_indices:
            X = dataset.get_single_matrix([i]).iloc[:,:-1]
            X = column_transformer.transform(X)
            y = dataset.get_single_matrix([i]).iloc[:,-1].to_numpy()
            y = y_normalizer.transform(y)
            self.test_samples.append((X,y))

        # Save the transformer using joblib
        os.makedirs(os.path.join(self.benchmarks_dir, self.name), exist_ok=True)
        joblib.dump(column_transformer, os.path.join(self.benchmarks_dir, self.name, 'column_transformer.joblib'))

        # Save y_normalizer
        y_normalizer.save(os.path.join(self.benchmarks_dir, self.name))
    
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
        if not tuning:
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
    


class SimpleLinearBenchmark(BaseBenchmark):
    """Simple linear benchmark."""
    
    def get_name(self):
        return 'Linear2'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        return MultiOutputRegressor(LinearRegression())

    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        if parameters is None:
            return MultiOutputRegressor(LinearRegression())
        else:
            return MultiOutputRegressor(LinearRegression())
    
    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):

        X, ts, ys = dataset.get_X_ts_ys()

        # Check if all ys have the same length
        assert len(set([y.shape[0] for y in ys])) == 1

        # Combine ys into a single matrix
        y = np.stack(ys, axis=0)

        y0s = y[:,0]
        y0s = y0s.reshape(-1,1)

        self.X_train = X.iloc[train_indices,:]
        self.y_train = y[train_indices,:]

        self.X_val = X.iloc[val_indices,:]
        self.y_val = y[val_indices,:]

        self.X_test = X.iloc[test_indices,:]
        self.y_test = y[test_indices,:]

        column_transformer = dataset.get_default_column_transformer()
        y_normalizer = YNormalizer()

        self.X_train = column_transformer.fit_transform(self.X_train)
        self.y_train = y_normalizer.fit_transform(self.y_train)
        self.y0s_train = self.y_train[:,0].reshape(-1,1)

        self.X_val = column_transformer.transform(self.X_val)
        self.y_val = y_normalizer.transform(self.y_val)
        self.y0s_val = self.y_val[:,0].reshape(-1,1)

        self.X_test = column_transformer.transform(self.X_test)
        self.y_test = y_normalizer.transform(self.y_test)
        self.y0s_test = self.y_test[:,0].reshape(-1,1)

        # Save the transformer using joblib
        os.makedirs(os.path.join(self.benchmarks_dir, self.name), exist_ok=True)
        joblib.dump(column_transformer, os.path.join(self.benchmarks_dir, self.name, 'column_transformer.joblib'))

        # Save y_normalizer
        y_normalizer.save(os.path.join(self.benchmarks_dir, self.name))
    
    def train(self, model, tuning=False):
        """Train model."""
        
        if not tuning:
            y_train = np.concatenate([self.y_train, self.y_val], axis=0)
            y0s_train = np.concatenate([self.y0s_train, self.y0s_val], axis=0)
        else:
            y_train = self.y_train
            y0s_train = self.y0s_train


        model.fit(y0s_train, y_train)

        train_loss = mean_squared_error(y_train, model.predict(y0s_train))
        val_loss = mean_squared_error(self.y_val, model.predict(self.y0s_val))
        test_loss = mean_squared_error(self.y_test, model.predict(self.y0s_test))
        

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

        self.X_train = X.iloc[train_indices,:]
        self.ts_train = [ts[i] for i in train_indices]
        self.ys_train = [ys[i] for i in train_indices]
        self.X_val = X.iloc[val_indices,:]
        self.ts_val = [ts[i] for i in val_indices]
        self.ys_val = [ys[i] for i in val_indices]
        self.X_test = X.iloc[test_indices,:]
        self.ts_test = [ts[i] for i in test_indices]
        self.ys_test = [ys[i] for i in test_indices]

        # Transform the data
        transformer = dataset.get_default_column_transformer()
        y_normalizer = YNormalizer()

        self.X_train = transformer.fit_transform(self.X_train)
        self.ys_train = y_normalizer.fit_transform(self.ys_train)

        self.X_val = transformer.transform(self.X_val)
        self.ys_val = y_normalizer.transform(self.ys_val)
        self.X_test = transformer.transform(self.X_test)
        self.ys_test = y_normalizer.transform(self.ys_test)

        # Save the transformer using joblib
        os.makedirs(os.path.join(self.benchmarks_dir, self.name), exist_ok=True)
        joblib.dump(transformer, os.path.join(self.benchmarks_dir, self.name, 'column_transformer.joblib'))

        # Save y_normalizer
        y_normalizer.save(os.path.join(self.benchmarks_dir, self.name))

        if self.config.n_basis_tunable:
            # That means that we cannot precompute the internal knots and instantiate the datasets
            # Instead we will have to do it in the train function
            return
        else: 
            if self.config.internal_knots is None:
                # We need to find the internal knots

                n_internal_knots = self.config.n_basis - 2

                internal_knots = calculate_knot_placement(self.ts_train, self.ys_train, n_internal_knots, T=self.config.T, seed=self.config.seed)
                print(f'Found internal knots: {internal_knots}')

                self.config.internal_knots = internal_knots

            self.train_dataset = TTSDataset(self.config, (self.X_train, self.ts_train, self.ys_train))
            self.val_dataset = TTSDataset(self.config, (self.X_val, self.ts_val, self.ys_val))
            self.test_dataset = TTSDataset(self.config, (self.X_test, self.ts_test, self.ys_test))
        

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

        if not tuning:
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
    


class RNNBenchmark(BaseBenchmark):
    """RNN benchmark."""

    def __init__(self, config):
        self.config = config
        super().__init__()

    def get_name(self):
        return 'RNN'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        config = RNNTuningConfig(
                            trial,
                            decoder_type=self.config.decoder_type,
                            n_features=self.config.n_features,
                            max_len = self.config.max_len,
                            seed=self.config.seed,
                            device=self.config.device,
                            num_epochs=self.config.num_epochs)
        litmodel = LitRNN(config)
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
            decoder = {
                'input_dim': parameters['hidden_size_2'],
                'hidden_dim': parameters['decoder_hidden_dim'],
                'output_dim': 1,
                'num_layers': parameters['decoder_num_layers'],
                'dropout_p': parameters['decoder_dropout_p']
            }
            training = {
                'optimizer': 'adam',
                'batch_size': parameters['batch_size'],
                'lr': parameters['lr'],
                'weight_decay': parameters['weight_decay'],
            }

            config = RNNConfig(decoder_type=self.config.decoder_type,
                            n_features=self.config.n_features,
                            seed=seed,
                            max_len=self.config.max_len,
                            encoder=encoder,
                            decoder=decoder,
                            training=training,
                            device=self.config.device,
                            num_epochs=self.config.num_epochs
                            )
        else:
            config = self.config

        litmodel = LitRNN(config)
        return litmodel

    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):
        X, ts, ys = dataset.get_X_ts_ys()

        # Check if every y has the same number of elements
        y_lengths = [len(y) for y in ys]
        if not all([self.config.max_len == y_length for y_length in y_lengths]):
            raise ValueError("All y values must have the same length if you want to use RNN. Same as max_len parameter of RNN.")

        self.X_train = X.iloc[train_indices,:]
        self.ts_train = [ts[i] for i in train_indices]
        self.ys_train = [ys[i] for i in train_indices]
        self.X_val = X.iloc[val_indices,:]
        self.ts_val = [ts[i] for i in val_indices]
        self.ys_val = [ys[i] for i in val_indices]
        self.X_test = X.iloc[test_indices,:]
        self.ts_test = [ts[i] for i in test_indices]
        self.ys_test = [ys[i] for i in test_indices]

        # Transform the data
        transformer = dataset.get_default_column_transformer()
        y_normalizer = YNormalizer()

        self.X_train = transformer.fit_transform(self.X_train)
        self.ys_train = y_normalizer.fit_transform(self.ys_train)

        self.X_val = transformer.transform(self.X_val)
        self.ys_val = y_normalizer.transform(self.ys_val)
        self.X_test = transformer.transform(self.X_test)
        self.ys_test = y_normalizer.transform(self.ys_test)

        # Save the transformer using joblib
        os.makedirs(os.path.join(self.benchmarks_dir, self.name), exist_ok=True)
        joblib.dump(transformer, os.path.join(self.benchmarks_dir, self.name, 'column_transformer.joblib'))

        # Save y_normalizer
        y_normalizer.save(os.path.join(self.benchmarks_dir, self.name))

        self.train_dataset = torch.utils.data.TensorDataset(torch.tensor(self.X_train, dtype=torch.float32),torch.stack([torch.tensor(y, dtype=torch.float32) for y in self.ys_train], dim=0))
        self.val_dataset = torch.utils.data.TensorDataset(torch.tensor(self.X_val, dtype=torch.float32),torch.stack([torch.tensor(y, dtype=torch.float32) for y in self.ys_val], dim=0))
        self.test_dataset = torch.utils.data.TensorDataset(torch.tensor(self.X_test, dtype=torch.float32),torch.stack([torch.tensor(y, dtype=torch.float32) for y in self.ys_test], dim=0))
        

    def train(self, model, tuning=False):
        """Train model."""
        if tuning:
            tuning_callback = model[1]
            model = model[0]
            log_dir = os.path.join(self.benchmarks_dir, self.name, 'tuning', 'logs', f'seed_{model.config.seed}')
        else:
            log_dir =  os.path.join(self.benchmarks_dir, self.name, 'final', 'logs', f'seed_{model.config.seed}')

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

        gen = torch.Generator()
        seed = model.config.seed
        gen.manual_seed(seed)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, generator=gen)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, generator=gen)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, generator=gen)

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


class DeltaTRNNBenchmark(BaseBenchmark):
    """RNN benchmark."""

    def __init__(self, config):
        self.config = config
        super().__init__()

    def get_name(self):
        return 'DeltaTRNN'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        config = RNNTuningConfig(
                            trial,
                            decoder_type=self.config.decoder_type,
                            n_features=self.config.n_features,
                            max_len = self.config.max_len,
                            seed=self.config.seed,
                            device=self.config.device,
                            num_epochs=self.config.num_epochs)
        litmodel = LitDeltaTRNN(config)
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
            decoder = {
                'input_dim': parameters['hidden_size_2'],
                'hidden_dim': parameters['decoder_hidden_dim'],
                'output_dim': 1,
                'num_layers': parameters['decoder_num_layers'],
                'dropout_p': parameters['decoder_dropout_p']
            }
            training = {
                'optimizer': 'adam',
                'batch_size': parameters['batch_size'],
                'lr': parameters['lr'],
                'weight_decay': parameters['weight_decay'],
            }

            config = RNNConfig(decoder_type=self.config.decoder_type,
                            n_features=self.config.n_features,
                            seed=seed,
                            max_len=self.config.max_len,
                            encoder=encoder,
                            decoder=decoder,
                            training=training,
                            device=self.config.device,
                            num_epochs=self.config.num_epochs
                            )
        else:
            config = self.config

        litmodel = LitDeltaTRNN(config)
        return litmodel

    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):
        X, ts, ys = dataset.get_X_ts_ys()

        self.X_train = X.iloc[train_indices,:]
        self.ts_train = [ts[i] for i in train_indices]
        self.ys_train = [ys[i] for i in train_indices]
        self.X_val = X.iloc[val_indices,:]
        self.ts_val = [ts[i] for i in val_indices]
        self.ys_val = [ys[i] for i in val_indices]
        self.X_test = X.iloc[test_indices,:]
        self.ts_test = [ts[i] for i in test_indices]
        self.ys_test = [ys[i] for i in test_indices]

        # Transform the data
        transformer = dataset.get_default_column_transformer()
        y_normalizer = YNormalizer()

        self.X_train = transformer.fit_transform(self.X_train)
        self.ys_train = y_normalizer.fit_transform(self.ys_train)

        self.X_val = transformer.transform(self.X_val)
        self.ys_val = y_normalizer.transform(self.ys_val)
        self.X_test = transformer.transform(self.X_test)
        self.ys_test = y_normalizer.transform(self.ys_test)

        # Save the transformer using joblib
        os.makedirs(os.path.join(self.benchmarks_dir, self.name), exist_ok=True)
        joblib.dump(transformer, os.path.join(self.benchmarks_dir, self.name, 'column_transformer.joblib'))

        # Save y_normalizer
        y_normalizer.save(os.path.join(self.benchmarks_dir, self.name))

        # Create datasets

        def prepare_tensor_dataset(X, ts, ys):
            """
            Calculates dt from t
            Pads dt and y with zeros to max_len
            Creates mask
            """
            mask = np.zeros((len(ts), self.config.max_len))
            dts = []
            padded_ys = []
            for i in range(len(ts)):
                t = ts[i]
                y = ys[i]
                dt = np.diff(t)
                dt = np.insert(dt, 0, 0)
                n = len(dt)
                if n > self.config.max_len:
                    dt = dt[:self.config.max_len]
                    y = y[:self.config.max_len]
                else:
                    dt = _pad_to_shape(dt, (self.config.max_len,))
                    y = _pad_to_shape(y, (self.config.max_len,))
                mask[i,0:n] = 1
                dts.append(dt)
                padded_ys.append(y)
            
            X = torch.tensor(X, dtype=torch.float32)
            dts = torch.tensor(np.stack(dts, axis=0), dtype=torch.float32)
            padded_ys = torch.tensor(np.stack(padded_ys, axis=0), dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)

            return torch.utils.data.TensorDataset(X, dts, padded_ys, mask)


        self.train_dataset = prepare_tensor_dataset(self.X_train, self.ts_train, self.ys_train)
        self.val_dataset = prepare_tensor_dataset(self.X_val, self.ts_val, self.ys_val)
        self.test_dataset = prepare_tensor_dataset(self.X_test, self.ts_test, self.ys_test)
        

    def train(self, model, tuning=False):
        """Train model."""
        if tuning:
            tuning_callback = model[1]
            model = model[0]
            log_dir = os.path.join(self.benchmarks_dir, self.name, 'tuning', 'logs', f'seed_{model.config.seed}')
        else:
            log_dir =  os.path.join(self.benchmarks_dir, self.name, 'final', 'logs', f'seed_{model.config.seed}')

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

        gen = torch.Generator()
        seed = model.config.seed
        gen.manual_seed(seed)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, generator=gen)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, generator=gen)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, generator=gen)

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
    