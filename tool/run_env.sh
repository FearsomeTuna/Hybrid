#!/bin/sh

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
conda activate ptEnv  # pytorch env
PYTHON=python

export PYTHONPATH=./
$PYTHON -u "$@"
