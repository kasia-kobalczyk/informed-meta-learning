# Informed Meta-Learning with INPs

This repository contains the code to reproduce the results of the exepriments presented in the paper:

[Towards Automated Knowledge Integration From Human-Interpretable Representations](https://openreview.net/forum?id=NTHMw8S1Ow) published at ICLR 2025

<img src="https://github.com/kasia-kobalczyk/informed-meta-learning/blob/main/figure1.png?raw=true" width="800"/>

For citations, use the following:
```
@inproceedings{
kobalczyk2025towards,
title={Towards Automated Knowledge Integration From Human-Interpretable Representations},
author={Katarzyna Kobalczyk and Mihaela van der Schaar},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=NTHMw8S1Ow}
}
```

## Setup

The `environment.yaml` lists the required packages to reproduce the experiments presented in the paper. To install the evnironment run:

`conda env create -f environment.yaml`

`conda activate inps`

## Experiments with Synthetic Data
`jobs/run_sinusoids.sh` contais commands that need to be run to reproduce the experiments with synthetic data

After training the models, results can be analyzed with the following two notebooks:
- `evaluation/evaluate_sinusoids.ipynb` contains the analysis of the base experiments
- `evaluation/evaluate_sinusoids_dist_shift.ipynb` contains the analysis of the train/test distribution shift experiment

## Experiments with the Tempereatures Dataset
`jobs/run_temperatures.sh` contais commands that need to be run to reproduce the experiments with the tempereatures datasets

After training the models, results can be analyzes with `evaluation/evaluate_temperature.ipynb`
