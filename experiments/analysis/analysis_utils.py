# This file containts utils for analysis of the experiments
import sys
sys.path.append('../../')
import os
import json
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from experiments.benchmark import load_column_transformer
from experiments.baselines import YNormalizer
from experiments.datasets import load_dataset


# This function finds experiment results with given characteristics
def find_results(dataset_name, model_name, results_dir='../benchmarks', summary_filename="summary.json", filter_dict={}):
    found_timestamps = []
    
    # Open the summary file
    with open(os.path.join(results_dir, summary_filename), 'r') as summary_file:
        summary = json.load(summary_file)

    for result in summary:
        if result['dataset_name'] == dataset_name:
            match = True
            for key, value in filter_dict.items():
                if key not in result or result[key] != value:
                    match = False
                    break
            if not match:
                continue

            model_names = result['results'].keys()
            if model_name in model_names:
                found_timestamps.append(result['timestamp'])
    
    return found_timestamps

def load_result_from_timestamp(timestamp, results_dir='../benchmarks', summary_filename="summary.json"):
    with open(os.path.join(results_dir, summary_filename), 'r') as summary_file:
        summary = json.load(summary_file)

    for result in summary:
        if result['timestamp'] == timestamp:
            return result
    
              
# Creates dataset if does not exist
def check_if_dataset_description_exists(dataset_name, dataset_descriptions_path="../dataset_descriptions"):

    # Check if dataset description file exists
    dataset_path = os.path.join(dataset_descriptions_path, dataset_name)
    if not os.path.exists(dataset_path):
        raise ValueError("Dataset description for {} does not exist".format(dataset_name))
    return

def create_shap_plot(model_name, timestamp, dataset_name, dataset_title, timesteps, cmap_scale=0.4, results_dir='../benchmarks', data_dir=None, dataset_description_path="../dataset_descriptions"):

    dataset = load_dataset(dataset_name, data_folder=data_dir, dataset_description_path=dataset_description_path)
    column_transformer = load_column_transformer(timestamp, model_name, benchmarks_dir=results_dir)
    y_normalizer = YNormalizer.load_from_benchmark(timestamp, model_name, benchmark_dir=results_dir)

    # Load the baseline
    path = os.path.join(results_dir, timestamp, model_name, 'final', 'logs')

    def _get_seed_number(path):
        seeds = [os.path.basename(path).split("_")[1] for path in glob.glob(os.path.join(path, '*'))]
        seed = seeds[0]
        return seed

    seed = _get_seed_number(path)

    full_path = os.path.join(path, f'seed_{seed}', 'model.pkl')

    # Load from pickle
    import pickle
    with open(full_path, 'rb') as f:
        model = pickle.load(f)



    X, ts, ys = dataset.get_X_ts_ys()

    explainer = shap.TreeExplainer(model)

    # explain the model's predictions using SHAP values
    all_shap_values = []
    for t in timesteps:
        X['t'] = t
        shap_values = explainer.shap_values(X)
        all_shap_values.append(shap_values)
    all_shap_values = np.stack(all_shap_values, axis=2)
    all_shap_values = np.mean(all_shap_values, axis=0)

    # convert back to original scale
    scale = y_normalizer.y_std
    all_shap_values = all_shap_values * scale

    feature_names = dataset.get_feature_names() + ['t']

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(all_shap_values, cmap='plasma')

    # Add a small colorbar next to the plot where c_size controls the size of the colorbar
    plt.colorbar(shrink=cmap_scale)


    # Add feature names as y labels
    plt.yticks(range(len(feature_names)), feature_names, fontsize=14)
    # Add xticks for every other time step (to avoid overcrowding) and round to 2 decimal places
    plt.xticks(range(0, len(timesteps), 2), np.round(timesteps[::2], 2), fontsize=14)
    plt.xlabel('Time (hours)', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(f'SHAP values for each feature over time ({dataset_title} dataset)', fontsize=14)
    return fig



     

  
    