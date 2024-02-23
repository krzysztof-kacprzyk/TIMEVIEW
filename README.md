# TIMEVIEW

This is the original repository for the paper "Towards Transparent Time Series Forecasting".

## Clone the repository
Clone the repository using
```
git clone https://github.com/krzysztof-kacprzyk/TIMEVIEW.git
```

## Dependencies
You can install all required dependencies using conda and the following command
```
conda env create -n timeview --file environment.yml
```
This will also install `timeview` (the main module) in editable mode.

## Running all experiments
To run all experiments navigate to `experiments` using
```
cd experiments
``` 
and run
```
./run_scripts/run_all.sh
```
Or you can call the scripts individually in `run_scripts`.

The results will be saved in
```
experiments/benchmarks/{timestamp}/
experiments/benchmarks/summary.json
```

## Figures and tables
Jupyter notebooks used to create all figures and tables in the paper can be found in `experiments/analysis`.

## Other information
To properly install PySR follow instructions on https://github.com/MilesCranmer/PySR

## Citations
If you use this code, please cite using the following information.

*Kacprzyk, K., Liu, T. & van der Schaar, M. Towards Transparent Time Series Forecasting. The Twelfth International Conference on Learning Representations (2024).*


```
@inproceedings{Kacprzyk.TransparentTimeSeries.2024,
  title = {Towards Transparent Time Series Forecasting},
  booktitle = {The {{Twelfth International Conference}} on {{Learning Representations}}},
  author = {Kacprzyk, Krzysztof and Liu, Tennison and {van der Schaar}, Mihaela},
  year = {2024},
}
```

