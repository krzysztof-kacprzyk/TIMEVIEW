import os
import optuna
import json
import pickle
from timeview.config import TuningConfig
from timeview.training import training
from timeview.data import synthetic_tumor_data, TTSDataset


def tuning(seed: int, tuning_dir: str,  n_trials: int):
    '''
    Hyperparameter tuning script
    Arguments:
        seed: int - seed for reproducibility
        tuning_dir: str - directory to store tuning results
        n_trials: int - number of trials to run, 100 is generally a good starting point, but you may need less
    '''

    def objective(trial):
        config = TuningConfig(
            trial=trial,
            n_features=4,
            n_basis=5,
            T=1,
            seed=seed
        )
        # you can abstract this out, so that it can be passed in as an argument to tuning()
        # e.g. tuning(seed, tuning_dir, dataset=(X, ts, ys))
        # and then here do dataset = TTSDataset(config, *dataset)
        # this will give you a more generalized tuning function
        X, ts, ys = synthetic_tumor_data(
            2000, 20, 1.0, 0.0, seed=seed, equation='wilkerson')
        dataset = TTSDataset(config, (X, ts, ys))

        # you have to pass in trial, to tell training() to perform optuna hyperopt
        best_val_loss = training(seed, config, dataset, trial)

        return best_val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_trial = study.best_trial
    best_hyperparameters = best_trial.params

    print('[Best hyperparameter configuration]:')
    print(best_hyperparameters)

    tuning_dir = os.path.join(tuning_dir, f'seed_{seed}')
    os.makedirs(tuning_dir, exist_ok=True)

    # save best hyperparameters
    hyperparam_save_path = os.path.join(tuning_dir, f'hyperparameters.json')
    with open(hyperparam_save_path, 'w') as f:
        json.dump(best_hyperparameters, f)

    # save optuna study
    study_save_path = os.path.join(tuning_dir, f'study_{seed}.pkl')
    with open(study_save_path, 'wb') as f:
        pickle.dump(study, f)

    # save optuna visualizations
    fig = optuna.visualization.plot_intermediate_values(study)
    fig.write_image(os.path.join(tuning_dir, 'intermediate_values.png'))

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(os.path.join(tuning_dir, 'optimization_history.png'))

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(tuning_dir, 'param_importance.png'))

    print(f'[Tuning complete], saved tuning results to {tuning_dir}')

    return
