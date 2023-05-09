import copy
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
from baseline_implementations.rnn.config import RNNConfig

import argparse

import time

def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def make_json_serializable(dictionary):
    if is_json_serializable(dictionary):
        return dictionary
    else:
        for key, value in dictionary.items():
            if is_json_serializable(value):
                continue
            elif isinstance(value, dict):
                dictionary[key] = make_json_serializable(value)
            else:
                dictionary[key] = {
                    'class': value.__class__.__name__,
                    'value': make_json_serializable(value.__dict__) if hasattr(value, '__dict__') else str(value)
                }
    return dictionary

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
        # {
        # 'dataset_name': 'mimic_0.1',
        # 'dataset_builder': 'MIMICDataset',
        # 'dataset_dictionary': {
        #     'subset': 0.1,
        #     'seed': 0}
        # }
        {
            'dataset_name': 'airfoil_log',
            'dataset_builder': 'AirfoilDataset',
            'dataset_dictionary': {"log_t": True}
        },
        {
            'dataset_name': 'celgene',
            'dataset_builder': 'CelgeneDataset',
            'dataset_dictionary': {} 
        },
        {
            'dataset_name': 'flchain_1000',
            'dataset_builder': 'FLChainDataset',
            'dataset_dictionary': {"subset": 1000}
        },
        {
            'dataset_name': 'stress-strain-lot-max-0.2',
            'dataset_builder': 'StressStrainDataset',
            "dataset_dictionary": {
                "lot": "all",
                "include_lot_as_feature": True,
                "downsample": True,
                "more_samples": 0,
                "specimen": "all",
                "max_strain": 0.2}
        },
        {
            'dataset_name': 'tacrolimus_visit_12',
            'dataset_builder': 'TacrolimusDataset',
            'dataset_dictionary': {
                "granularity": "visit",
                "normalize": False,
                "max_t": 12.5}
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

            
def run_benchmarks(dataset_name, benchmarks: dict, dataset_split = [0.7,0.15,0.15], n_trials=10, n_tune=100, seed=0, benchmarks_dir='benchmarks', dataset_description_path='dataset_descriptions', notes=''):
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

    # Check if there exists a file summary.json in the benchmarks directory
    if os.path.exists(os.path.join(benchmarks_dir, 'summary.json')):
        # Load
        with open(os.path.join(benchmarks_dir, 'summary.json'), 'r') as f:
            summary = json.load(f)
    else:
        # Create
        summary = []
        # Save the summary
        with open(os.path.join(benchmarks_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
    
    summary.append(
        {
            'timestamp': timestamp,
            'dataset_name': dataset_name,
            'n_trials': n_trials,
            'n_tune': n_tune,
            'train_size': dataset_split[0],
            'val_size': dataset_split[1],
            'seed': seed,
            'results': {},
            'notes': notes
        }
    )

    # Save the summary
    with open(os.path.join(benchmarks_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)


    # Generate train, validation, and test indices
    train_indices, val_indices, test_indices = generate_indices(len(dataset), dataset_split[0], dataset_split[1], seed=seed)

    results = {'name':[],'mean':[],'std':[], 'time_elapsed':[]}

    old_benchmarks_dir = benchmarks_dir

    benchmarks_dir = os.path.join(benchmarks_dir, timestamp)
    # Check if the benchmarks directory exists and create it if not
    if not os.path.exists(benchmarks_dir):
        os.mkdir(benchmarks_dir)

    # Save the benchmarks to a pickle file
    with open(os.path.join(benchmarks_dir, 'baselines.pkl'), 'wb') as f:
        pickle.dump(benchmarks, f)
    # Save the benchmarks as a json file
    with open(os.path.join(benchmarks_dir, 'baselines.json'), 'w') as f:
        benchmarks_to_save = copy.deepcopy(benchmarks)
        json.dump(make_json_serializable(benchmarks_to_save), f, indent=4)
    
    
    for baseline_name, parameter_dict in benchmarks.items():
        time_start = time.time()
        benchmark = get_baseline(baseline_name, parameter_dict)
        losses = benchmark.run(dataset, train_indices, val_indices, test_indices, n_trials=n_trials, n_tune=n_tune, seed=seed, benchmarks_dir=benchmarks_dir, timestamp=timestamp)
        time_end = time.time()
        results['name'].append(benchmark.name)
        results['mean'].append(np.mean(losses))
        results['std'].append(np.std(losses))
        results['time_elapsed'].append(time_end - time_start)

        summary[-1]['results'][benchmark.name] = {'mean': np.mean(losses), 'std': np.std(losses), 'time_elapsed': time_end - time_start}

        # Save the summary
        with open(os.path.join(old_benchmarks_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

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
    parser.add_argument('--n_basis', type=int, default=5, help='Number of basis functions to use for TTS')
    parser.add_argument('--device', type=str, default='gpu', help='Device to run on.')
    parser.add_argument('--validate', action='store_true', help='Whether to validate first', default=False)
    parser.add_argument('--rnn_type', type=str, choices=['lstm', 'rnn'], default='lstm', help='RNN type to use')

    args = parser.parse_args()

    global_seed = args.seed

    dataset_description_path = 'dataset_descriptions'
    benchmarks_dir = 'benchmarks'

    create_benchmark_datasets_if_not_exist(dataset_description_path=dataset_description_path)

    dataset_names = args.datasets

    notes = f"n_basis={args.n_basis},rnn_type={args.rnn_type}"

    if args.n_basis == 0:
        n_basis_tunable = True
        args.n_basis = 5
    else:
        n_basis_tunable = False


    benchmark_options = {
        'n_trials': args.n_trials,
        'n_tune': args.n_tune,
        'seed': global_seed,
        'dataset_split': [0.7,0.15,0.15],
        'benchmarks_dir': benchmarks_dir,
        'dataset_description_path': dataset_description_path,
        'notes': notes,
    }

    tts_T = {
        'synthetic_tumor_wilkerson_1': 1.0,
        'tumor': 1.0,
        'airfoil_log': 4.7,
        'celgene': 1.0,
        'flchain_1000': 1.0,
        'stress-strain-lot-max-0.2': 1.0,
        'tacrolimus_visit_12': 12.5
    }

    tts_n_features = {
        'synthetic_tumor_wilkerson_1': 4,
        'tumor': 1,
        'airfoil_log': 4,
        'celgene': 11,
        'flchain_1000': 16,
        'stress-strain-lot-max-0.2': 10,
        'tacrolimus_visit_12': 9
    }

    rnn_max_len = {
        'synthetic_tumor_wilkerson_1': 20,
        'flchain_1000': 20,
    }

    if args.validate:

        print('Validating benchmarks...')

        validate_benchmark_options = {
            'n_trials': 1,
            'n_tune': 1,
            'seed': global_seed,
            'dataset_split': [0.7,0.15,0.15],
            'benchmarks_dir': benchmarks_dir,
            'dataset_description_path': dataset_description_path,
            'notes': notes,
        }

        for dataset_name in dataset_names:

            print('Running validation on dataset: {}'.format(dataset_name))

            benchmarks = {}

            if 'TTS' in args.baselines:
                tts_config = Config(n_features=tts_n_features[dataset_name], n_basis=args.n_basis, T=tts_T[dataset_name], seed=global_seed, dataloader_type='tensor', num_epochs=1000, device=args.device, n_basis_tunable=n_basis_tunable, dynamic_bias=True)
                benchmarks['TTS'] = {'config': tts_config}
            if 'XGB' in args.baselines:
                benchmarks['XGB'] = {}
            if 'GAM' in args.baselines:
                benchmarks['GAM'] = {}
            if 'Mean' in args.baselines:
                benchmarks['Mean'] = {}
            if 'RNN' in args.baselines:
                rnn_config = RNNConfig(args.rnn_type, n_features=tts_n_features[dataset_name], seed=global_seed, max_len=rnn_max_len[dataset_name], num_epochs=1000, device=args.device)
                benchmarks['RNN'] = {'config': rnn_config}

            run_benchmarks(dataset_name, benchmarks, **validate_benchmark_options)
        
        print('Validation complete.')

    print('Running benchmarks...')

    for dataset_name in dataset_names:

        print('Running benchmarks on dataset: {}'.format(dataset_name))

        benchmarks = {}

        if 'TTS' in args.baselines:
            tts_config = Config(n_features=tts_n_features[dataset_name], n_basis=args.n_basis, T=tts_T[dataset_name], seed=global_seed, dataloader_type='tensor', num_epochs=1000, device=args.device, n_basis_tunable=n_basis_tunable, dynamic_bias=True)
            benchmarks['TTS'] = {'config': tts_config}
        if 'XGB' in args.baselines:
            benchmarks['XGB'] = {}
        if 'GAM' in args.baselines:
            benchmarks['GAM'] = {}
        if 'Mean' in args.baselines:
            benchmarks['Mean'] = {}
        if 'RNN' in args.baselines:
            rnn_config = RNNConfig(args.rnn_type, n_features=tts_n_features[dataset_name], seed=global_seed, max_len=rnn_max_len[dataset_name], num_epochs=1000, device=args.device)
            benchmarks['RNN'] = {'config': rnn_config}


        run_benchmarks(dataset_name, benchmarks, **benchmark_options)

    







