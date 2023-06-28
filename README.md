# Transparent Time Series

## Dependencies
You can install all required dependencies using conda and the following command
```
conda env create -n tts --file environment.yml
```
This will also install tts (the main module) in editable mode.

## Running all experiments
To run all experiments run the commands in the following two files
```
experiments/run_all_benchmarks.bat
experiments/run_sensitivity.bat
```
The results will be saved in
```
experiments/benchmarks/{timestamp}/
experiments/benchmarks/summary.json
```

## Visualization
To check out the visualization tool used for the examples in the paper, open jupyter notebook
```
experiments/visualize_all.ipynb
```

## Other information
To properly install PySR follow instructions on https://github.com/MilesCranmer/PySR

