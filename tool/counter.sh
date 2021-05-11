#!/bin/sh

## uncomment for slurm
##SBATCH -p quadro
##SBATCH --gres=gpu:8
##SBATCH -c 80

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
conda activate ptEnv  # pytorch env
PYTHON=python

export PYTHONPATH=./
$PYTHON -u ./tool/counter.py
