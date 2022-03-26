#!/bin/bash

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
conda activate torchEnv  # pytorch env
PYTHON=python

export PYTHONPATH=./
$PYTHON -u "$@"
