from datetime import datetime
import os
import joblib
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

from tts.data import TTSDataset, create_dataloader, BaseDataset
from tts.config import TuningConfig, Config
from tts.model import TTS
from tts.lit_module import LitTTS
import pytorch_lightning as pl
import torch
from datasets import load_dataset, save_dataset
from baselines import GAMBenchmark, XGBBenchmark, TTSBenchmark, BaseBenchmark, get_baseline

import argparse

import time

def load_column_transformer(timestamp, baseline='TTS', benchmarks_dir='benchmarks'):
    path = os.path.join(benchmarks_dir, timestamp, baseline, 'column_transformer.joblib')
    column_transformer = joblib.load(path)
    return column_transformer


def generate_indices(n, train_size, val_size, seed=0):
    gen = np.random.default_rng(seed)
    train_indices = gen.choice(n, int(n*train_size), replace=False)
    train_indices = [i.item() for i in train_indices]
    val_indices = gen.choice(list(set(range(n)) - set(train_indices)), int(n*val_size), replace=False)
    val_indices = [i.item() for i in val_indices]
    test_indices = list(set(range(n)) - set(train_indices) - set(val_indices))
    return train_indices, val_indices, test_indices


def create_benchmark_datasets_if_not_exist(dataset_description_path='dataset_descriptions'):

    datasets = [
        {
        'dataset_name': 'synthetic_tumor_wilkerson_1',
        'dataset_builder': 'SyntheticTumorDataset',
        'dataset_dictionary': {
            'n_samples': 2000,
            'n_time_steps': 20,
            'time_horizon': 1.0,
            'noise_std': 0.0,
            'seed': 0,
            'equation': 'wilkerson'}
        },
        {
        'dataset_name': 'tumor',
        'dataset_builder': 'TumorDataset',
        'dataset_dictionary': {}
        },
        {
        'dataset_name': 'mimic_0.1',
        'dataset_builder': 'MIMICDataset',
        'dataset_dictionary': {
            'subset': 0.1,
            'seed': 0}
        }
    ]

    # Check if the dataset description directory exists and create it if not
    if not os.path.exists(dataset_description_path):
        os.mkdir(dataset_description_path)
    
    # Check if the dataset description file exists and if not create it
    for dataset in datasets:
        dataset_name = dataset['dataset_name']
        dataset_path = os.path.join(dataset_description_path, f'{dataset_name}.json')
        if not os.path.exists(dataset_path):
            save_dataset(**dataset, dataset_description_path=dataset_description_path)

            
def run_benchmarks(dataset_name, benchmarks: dict, dataset_split = [0.7,0.15,0.15], n_trials=10, n_tune=100, seed=0, benchmarks_dir='benchmarks', dataset_description_path='dataset_descriptions'):
    """
    Runs a set of benchmarks on a dataset
    Args:
        dataset_name (str): The name of the dataset. Should be the same as the name of the dataset description file.
        benchmarks (dict): A dictionary of benchmarks to run. The keys are the names of the benchmarks and the values are the parameter dictionaries.
    """

    # Load the dataset
    dataset = load_dataset(dataset_name, dataset_description_path=dataset_description_path)

    # Check if there exists a file summary.csv in the benchmarks directory
    if os.path.exists(os.path.join(benchmarks_dir, 'summary.csv')):
        # Load into a DataFrame
        df = pd.read_csv(os.path.join(benchmarks_dir, 'summary.csv'), index_col=0)
    else:
        # Create a DataFrame
        df = pd.DataFrame(columns=['timestamp','dataset_name','n_trials', 'n_tune', 'train_size', 'val_size', 'seed'])

    # Add a row to the DataFrame
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    
    df = pd.concat([df,pd.DataFrame({'timestamp':[timestamp], 'dataset_name':[dataset_name], 'n_trials':[n_trials], 'n_tune':[n_tune], 'train_size':[dataset_split[0]], 'val_size':[dataset_split[1]], 'seed':[seed]})], ignore_index=True)
    
    # Save the DataFrame
    df.to_csv(os.path.join(benchmarks_dir, 'summary.csv'))

    # Generate train, validation, and test indices
    train_indices, val_indices, test_indices = generate_indices(len(dataset), dataset_split[0], dataset_split[1], seed=seed)

    results = {'name':[],'mean':[],'std':[], 'time_elapsed':[]}

    benchmarks_dir = os.path.join(benchmarks_dir, timestamp)
    # Check if the benchmarks directory exists and create it if not
    if not os.path.exists(benchmarks_dir):
        os.mkdir(benchmarks_dir)

    # Save the benchmarks to a pickle file
    with open(os.path.join(benchmarks_dir, 'baselines.pkl'), 'wb') as f:
        pickle.dump(benchmarks, f)
    
    for baseline_name, parameter_dict in benchmarks.items():
        time_start = time.time()
        benchmark = get_baseline(baseline_name, parameter_dict)
        losses = benchmark.run(dataset, train_indices, val_indices, test_indices, n_trials=n_trials, n_tune=n_tune, seed=seed, benchmarks_dir=benchmarks_dir, timestamp=timestamp)
        time_end = time.time()
        results['name'].append(benchmark.name)
        results['mean'].append(np.mean(losses))
        results['std'].append(np.std(losses))
        results['time_elapsed'].append(time_end - time_start)

    # Create a dataframe with the results
    df = pd.DataFrame(results)

    # Save the results
    df.to_csv(os.path.join(benchmarks_dir, 'results.csv'), index=False)

    return df

def repeat_benchmark(timestamp, benchmarks=None, benchmarks_dir='benchmarks', dataset_description_path='dataset_descriptions'):

    # Load the summary.csv file
    df = pd.read_csv(os.path.join(benchmarks_dir, 'summary.csv'), index_col=0)

    # Get the row corresponding to the timestamp
    row = df.loc[df['timestamp'] == timestamp]

    # Extract the parameters
    dataset_name = row['dataset_name'].values[0]
    n_trials = row['n_trials'].values[0]
    n_tune = row['n_tune'].values[0]
    train_size = row['train_size'].values[0]
    val_size = row['val_size'].values[0]
    seed = row['seed'].values[0]

    dataset_split = [train_size, val_size, 1-train_size-val_size]

    if benchmarks is None:
        # Load the baselines used in the benchmark from a pickle file
        with open(os.path.join(benchmarks_dir, timestamp, 'baselines.pkl'), 'rb') as f:
            benchmarks = pickle.load(f)

    run_benchmarks(dataset_name, benchmarks, n_trials=n_trials, n_tune=n_tune, dataset_split=dataset_split, seed=seed, benchmarks_dir=benchmarks_dir, dataset_description_path=dataset_description_path)

def get_dataset_and_indices_from_seed(dataset_name, seed, dataset_split=[0.7,0.15,0.15], dataset_description_path='dataset_descriptions'):
    
    # Load the dataset
    dataset = load_dataset(dataset_name, dataset_description_path=dataset_description_path)

    # Generate train, validation, and test indices
    train_indices, val_indices, test_indices = generate_indices(len(dataset), dataset_split[0], dataset_split[1], seed=seed)

    return dataset, train_indices, val_indices, test_indices


def get_dataset_and_indices_from_benchmark(timestamp, benchmarks_dir='benchmarks', dataset_description_path='dataset_descriptions'):

    # Load the summary.csv file
    df = pd.read_csv(os.path.join(benchmarks_dir, 'summary.csv'), index_col=0)

    # Get the row corresponding to the timestamp
    row = df.loc[df['timestamp'] == timestamp]

    # Extract the parameters
    dataset_name = row['dataset_name'].values[0]
    train_size = row['train_size'].values[0]
    val_size = row['val_size'].values[0]
    seed = row['seed'].values[0]

    # Load the dataset
    dataset = load_dataset(dataset_name, dataset_description_path=dataset_description_path)

    # Generate train, validation, and test indices
    train_indices, val_indices, test_indices = generate_indices(len(dataset), train_size, val_size, seed=seed)

    return dataset, train_indices, val_indices, test_indices



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run benchmarks.')
    parser.add_argument('--datasets', nargs='+', help='List of datasets to run benchmarks on.')
    parser.add_argument('--baselines', nargs='+', help='List of baselines to run.')
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials to run.')
    parser.add_argument('--n_tune', type=int, default=1, help='Number of tuning trials to run.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    args = parser.parse_args()

    global_seed = args.seed

    dataset_description_path = 'dataset_descriptions'
    benchmarks_dir = 'benchmarks'

    create_benchmark_datasets_if_not_exist(dataset_description_path=dataset_description_path)

    dataset_names = args.datasets


    benchmark_options = {
        'n_trials': args.n_trials,
        'n_tune': args.n_tune,
        'seed': global_seed,
        'dataset_split': [0.7,0.15,0.15],
        'benchmarks_dir': benchmarks_dir,
        'dataset_description_path': dataset_description_path
    }

    # Synthetic tumor Wilkerson 1
    if "synthetic_tumor_wilkerson_1" in dataset_names:
        dataset_name = 'synthetic_tumor_wilkerson_1'

        benchmarks = {}

        if 'XGB' in args.baselines:
            benchmarks['XGB'] = {}
        if 'GAM' in args.baselines:
            benchmarks['GAM'] = {}
        if 'TTS' in args.baselines:
            tts_config = Config(n_features=4, n_basis=5, T=1, seed=global_seed, dataloader_type='iterative', num_epochs=400, device='gpu')
            benchmarks['TTS'] = {'config': tts_config}

        run_benchmarks(dataset_name, benchmarks, **benchmark_options)

    if 'tumor' in dataset_names:
        dataset_name = 'tumor'

        benchmarks = {}

        if 'XGB' in args.baselines:
            benchmarks['XGB'] = {}
        if 'GAM' in args.baselines:
            benchmarks['GAM'] = {}
        if 'TTS' in args.baselines:
            tts_config = Config(n_features=2, n_basis=5, T=365, seed=global_seed, dataloader_type='iterative', num_epochs=400, device='gpu')
            benchmarks['TTS'] = {'config': tts_config}

        run_benchmarks(dataset_name, benchmarks, **benchmark_options)


    if 'mimic_0.1' in dataset_names:
        dataset_name = 'mimic_0.1'

        benchmarks = {}

        if 'XGB' in args.baselines:
            benchmarks['XGB'] = {}
        if 'GAM' in args.baselines:
            benchmarks['GAM'] = {}
        if 'TTS' in args.baselines:
            tts_config = Config(n_features=43, n_basis=5, T=18, seed=global_seed, dataloader_type='tensor', num_epochs=400, device='gpu')
            benchmarks['TTS'] = {'config': tts_config}

        run_benchmarks(dataset_name, benchmarks, **benchmark_options)

    







