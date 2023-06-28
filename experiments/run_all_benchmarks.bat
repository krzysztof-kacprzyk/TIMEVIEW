python benchmark.py --datasets synthetic_tumor_wilkerson_1 airfoil_log flchain_1000 stress-strain-lot-max-0.2 tacrolimus_visit_12 --baselines TTS XGB GAM RNN SINDy --n_trials 10 --n_tune 100 --seed 0 --device gpu --n_basis 9 --rnn_type lstm
python benchmark.py --datasets sine_trans_200_20 beta_900_20 --baselines TTS XGB GAM RNN SINDy --n_trials 10 --n_tune 100 --seed 0 --device gpu --n_basis 5




python benchmark.py --datasets synthetic_tumor_wilkerson_1 flchain_1000 --baselines RNN --n_trials 10 --n_tune 100 --seed 0 --device gpu --rnn_type lstm
