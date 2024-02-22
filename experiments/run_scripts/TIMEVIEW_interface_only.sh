# This script runs TIMEVIEW model on datasets Sine, Beta, Tumour and Tacrolimus
# This data is needed for Figure 1, Figure 4, Figure 7, Figure 8, Figure 9, Figure 10, Figure 15

# Get the argument --debug to run the experiments in debug mode. This will run the experiments with a smaller number of tune iterations.
# Example: bash TIMEVIEW_interface_only.sh --debug
if [[ " $@ " =~ " --debug " ]]; then
    n_tune=1
else
    n_tune=100
fi

python benchmark.py --datasets sine_trans_200_20 beta_900_20 --baselines TTS --n_trials 10 --n_tune $n_tune --seed 0 --device gpu --n_basis 5 --rnn_type lstm
python benchmark.py --datasets synthetic_tumor_wilkerson_1 --baselines TTS --n_trials 10 --n_tune $n_tune --seed 0 --device gpu --n_basis 9 --rnn_type lstm

