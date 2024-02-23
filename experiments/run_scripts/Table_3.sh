# Get the argument --debug to run the experiments in debug mode. This will run the experiments with a smaller number of trials and tune iterations.
# Example: bash Table_3.sh --debug

# Check if --debug is in the arguments and set the number of trials and tune iterations accordingly.
if [[ " $@ " =~ " --debug " ]]; then
    n_trials=1
    n_tune=1
else
    n_trials=10
    n_tune=100
fi


# Comment this part if you already have run TIMEVIEW_interface_only.sh
if [[ " $@ " =~ " --debug " ]]; then
    bash ./run_scripts/TIMEVIEW_interface_only.sh --debug
else
    bash ./run_scripts/TIMEVIEW_interface_only.sh
fi

python benchmark.py --datasets airfoil_log flchain_1000 stress-strain-lot-max-0.2 --baselines TTS XGB GAM SINDy DeltaTRNN CatBoost LGBM DecisionTree ElasticNet --n_trials $n_trials --n_tune $n_tune --seed 0 --device gpu --n_basis 9 --rnn_type lstm

# Run RNN only when regularly sampled dataset
python benchmark.py --datasets flchain_1000 --baselines RNN --n_trials $n_trials --n_tune $n_tune --seed 0 --device gpu --n_basis 9 --rnn_type lstm

# Do not run TIMEVIEW on Sine, Beta and Tumour as we have already run it in TIMEVIEW_interface_only.sh
python benchmark.py --datasets synthetic_tumor_wilkerson_1 --baselines XGB GAM RNN SINDy DeltaTRNN CatBoost LGBM DecisionTree ElasticNet --n_trials $n_trials --n_tune $n_tune --seed 0 --device gpu --n_basis 9 --rnn_type lstm
python benchmark.py --datasets sine_trans_200_20 beta_900_20 --baselines XGB GAM RNN SINDy DeltaTRNN CatBoost LGBM DecisionTree ElasticNet --n_trials $n_trials --n_tune $n_tune --seed 0 --device gpu --n_basis 5 --rnn_type lstm


# PySR requires a different timeout for each dataset. The timeout is set to the time it took to run TIMEVIEW on the dataset divided by 10 (becuase we run 10 times). Appendix D.3.
python benchmark.py --datasets airfoil_log --baselines PySR --n_trials $n_trials --n_tune 0 --seed 0 --timeout 19
python benchmark.py --datasets flchain_1000 --baselines PySR --n_trials $n_trials --n_tune 0 --seed 0 --device gpu --timeout 98
python benchmark.py --datasets stress-strain-lot-max-0.2 --baselines PySR --n_trials $n_trials --n_tune 0 --seed 0 --timeout 19
python benchmark.py --datasets synthetic_tumor_wilkerson_1 --baselines PySR --n_trials $n_trials --n_tune 0 --seed 0 --timeout 270
python benchmark.py --datasets sine_trans_200_20 --baselines PySR --n_trials $n_trials --n_tune 0 --seed 0 --timeout 100
python benchmark.py --datasets beta_900_20 --baselines PySR --n_trials $n_trials --n_tune 0 --seed 0 --timeout 190

# Uncomment only if you have the Tacrolimus dataset as reported in the paper.
# python benchmark.py --datasets tacrolimus_visit_12 --baselines TTS XGB GAM RNN SINDy DeltaTRNN CatBoost LGBM DecisionTree ElasticNet --n_trials $n_trials --n_tune $n_tune --seed 0 --device gpu --n_basis 9 --rnn_type lstm
# python benchmark.py --datasets tacrolimus_visit_12 --baselines PySR --n_trials $n_trials --n_tune $n_tune --seed 0 --timeout 13