# Informed Meta-Learning with INPs

This repository contains the code to reproduce the results of the regression exepriments presented in the paper.

## Synthetic Data
`jobs/run_sinusoids.sh` contais commands that need to be run to reproduce the experiments with synthetic data 

After training the models, results can be analyzed with the following two notebooks:
- `evaluation/evaluate_sinusoids.ipynb` contains the analysis of the base experiments 
- `evaluation/evaluate_sinusoids_dist_shift.ipynb` contains the analysis of the train/test distribution shift experiment

## Tempereatures
`jobs/run_temperatures.sh` contais commands that need to be run to reproduce the experiments with the tempereatures datasets 

After training the models, results can be analyzes with `evaluation/evaluate_temperature.ipynb`